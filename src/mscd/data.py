"""Source construction and response regeneration are separate operations."""
import math
import random
from dataclasses import asdict
from pathlib import Path
from .types import SourceRecord, Request
from .artifacts import atomic_json, digest, read_json
from .generation import generate_cached


def validate_sources(records):
    records = list(records)
    if not records:
        raise ValueError("Empty source collection")
    if len({r.occurrence_id for r in records}) != len(records):
        raise ValueError("Duplicate source occurrence")
    if any(not r.prompt.strip() or not isinstance(r.response, str) for r in records):
        raise ValueError("Invalid source row")
    return records


class ImportedDatasetBuilder:
    def __init__(self, sources):
        self.sources = sources

    def build(self):
        from datasets import load_from_disk

        rows = []
        for source, path in self.sources.items():
            for i, r in enumerate(load_from_disk(path)):
                rows.append(
                    SourceRecord(source, f"{source}:{i}", r["prompt"], r["response"])
                )
        return validate_sources(rows)


class ExplicitPrefixDatasetBuilder:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)

    def build(self):
        from datasets import load_dataset
        from vllm import LLM, SamplingParams
        from ._construction import (
            make_composed_system_prompt,
            is_composed_response,
            dedupe_rows,
        )

        cfg = self.config
        if not cfg.get("prompt_revision") or not cfg.get("model_revision"):
            raise ValueError("Pin prompt_revision and model_revision before generation")
        ds = load_dataset(
            cfg["prompt_dataset"], revision=cfg["prompt_revision"], split="train"
        )
        pool = []
        for r in ds:
            instruction = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            if instruction:
                pool.append(instruction + (f"\n\nInput:\n{inp}" if inp else ""))
        # New construction replicates sample across the corpus, not just reorder its first 1500 rows.
        llm = LLM(
            model=cfg["base_model"],
            revision=cfg["model_revision"],
            dtype="bfloat16",
            seed=cfg["dataset_seed"],
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        )
        result = []
        for offset, (source, target) in enumerate(cfg["sources"].items()):
            seed = cfg["dataset_seed"] + offset
            n = cfg["rows_per_source"]
            n_pool = math.ceil(n * cfg["pool_multiplier"])
            prompts = list(pool)
            random.Random(seed).shuffle(prompts)
            prompts = prompts[:n_pool]
            if len(prompts) != n_pool:
                raise ValueError("Insufficient candidate prompts")
            raw_path = self.output / f"{source}.raw.json"
            if raw_path.exists():
                raw = read_json(raw_path)
            else:
                raw = []
                messages = [
                    [
                        {
                            "role": "system",
                            "content": make_composed_system_prompt(target),
                        },
                        {"role": "user", "content": p},
                    ]
                    for p in prompts
                ]
                params = [
                    SamplingParams(
                        temperature=cfg["source_temperature"],
                        max_tokens=cfg["source_max_tokens"],
                        seed=seed * 1000003 + i,
                    )
                    for i in range(len(prompts))
                ]
                batch = cfg.get("source_batch_size", 64)
                for start in range(0, len(prompts), batch):
                    outputs = llm.chat(
                        messages[start : start + batch],
                        params[start : start + batch],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    raw.extend(
                        {"prompt": p, "response": o.outputs[0].text.strip()}
                        for p, o in zip(prompts[start : start + batch], outputs)
                    )
                atomic_json(raw_path, raw)
            if [r["prompt"] for r in raw] != prompts:
                raise ValueError("Source candidates changed on resume")
            kept = dedupe_rows(
                [
                    r
                    for r in raw
                    if is_composed_response(
                        r["response"],
                        target["prefix"],
                        target.get("final_marker", "Joke"),
                    )
                ]
            )
            atomic_json(
                self.output / f"{source}.construction.json",
                dict(
                    candidate_count=len(raw),
                    valid_count=len(kept),
                    prompt_seed=seed,
                    response_seeds=[seed * 1000003 + i for i in range(len(prompts))],
                ),
            )
            if len(kept) < n:
                raise RuntimeError(
                    f"{source}: {len(kept)} valid rows; need {n}. No automatic retry or seed replacement."
                )
            # Match the historical final row shuffle while retaining its occurrence order.
            from datasets import Dataset

            kept = list(Dataset.from_list(kept[:n]).shuffle(seed=seed))
            result.extend(
                SourceRecord(source, f"{source}:{i}", r["prompt"], r["response"])
                for i, r in enumerate(kept)
            )
        return validate_sources(result)


class DatasetRegenerator:
    def run(
        self, sources, generator, config, output, implementation, seed=0, policy=None
    ):
        sources = validate_sources(sources)
        policy = dict(policy or {})
        sources = select_occurrences(sources, policy)
        if (
            policy.get("expected_raw") is not None
            and len(sources) != policy["expected_raw"]
        ):
            raise ValueError(
                "Regeneration occurrence count differs from the recorded selection"
            )
        requests = [
            Request(r.occurrence_id, r.prompt, seed + i, r.source_id)
            for i, r in enumerate(sources)
        ]
        records = generate_cached(
            generator, requests, config, Path(output) / "requests", implementation
        )
        if any(r.status != "completed" for r in records):
            raise RuntimeError(
                "Regeneration requires complete responses; no silent filtering"
            )
        kept, excluded = [], []
        mode = policy.get("filter", "none")
        if mode not in {"none", "semantic_validity"}:
            raise ValueError(
                "Unsupported regeneration filter; behavioral filtering is forbidden"
            )
        for s, r in zip(sources, records):
            reason = None
            if mode == "semantic_validity":
                if not isinstance(r.response, str) or r.stop_reason not in {
                    "eos",
                    "max_new_tokens",
                }:
                    reason = "malformed"
                elif not r.response.strip():
                    reason = "empty"
                elif r.stop_reason == "max_new_tokens":
                    reason = "token_limit_truncated"
            if reason:
                excluded.append(dict(occurrence_id=s.occurrence_id, reason=reason))
            else:
                kept.append(
                    SourceRecord(s.source_id, s.occurrence_id, s.prompt, r.response)
                )
        atomic_json(
            Path(output) / "selection.json",
            dict(
                policy=policy,
                occurrences=[s.occurrence_id for s in sources],
                raw_count=len(records),
                retained_count=len(kept),
                exclusions=excluded,
            ),
        )
        if len(kept) < policy.get("minimum_retained", 1):
            raise ValueError(
                "Insufficient structurally valid responses; no replacement generation"
            )
        if (
            policy.get("expected_retained") is not None
            and len(kept) != policy["expected_retained"]
        ):
            raise ValueError(
                "Retained count differs from historical replay; preserve artifacts and inspect the difference"
            )
        return kept


def select_occurrences(sources, policy):
    """Explicit historical occurrence selection; repeated prompts remain distinct."""
    import random

    sources = list(sources)
    if "occurrence_ids" in policy:
        lookup = {s.occurrence_id: s for s in sources}
        ids = policy["occurrence_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= lookup.keys():
            raise ValueError("Invalid occurrence manifest")
        return [lookup[i] for i in ids]
    if "recorded_manifest" in policy:
        lookup = {r.occurrence_id: r for r in sources}
        selected = []
        aliases = policy.get("source_aliases", {})
        for i, item in enumerate(policy["recorded_manifest"]):
            source = aliases.get(item["source"], item["source"])
            key = f"{source}:{item['source_row']}"
            if item["global_index"] != i or key not in lookup:
                raise ValueError("Invalid recorded occurrence selection")
            row = lookup[key]
            if row.prompt != item["prompt"]:
                raise ValueError("Recorded source prompt changed")
            selected.append(row)
        if len({r.occurrence_id for r in selected}) != len(selected):
            raise ValueError("Duplicate recorded occurrence ID")
        return selected
    if "per_source" in policy:
        normalized = lambda p: " ".join(p.split()).casefold()
        excluded = {normalized(p) for p in policy.get("excluded_prompts", [])}
        order = policy.get(
            "source_order", list(dict.fromkeys(r.source_id for r in sources))
        )
        if set(order) != {r.source_id for r in sources} or len(set(order)) != len(
            order
        ):
            raise ValueError("Source order must identify every source once")
        n = policy["per_source"]
        if not isinstance(n, int) or n < 1:
            raise ValueError("per_source must be positive")
        selected = {}
        for name in order:
            rows = [
                r
                for r in sources
                if r.source_id == name and normalized(r.prompt) not in excluded
            ]
            random.Random(policy.get("selection_seed", 0)).shuffle(rows)
            if len(rows) < n:
                raise ValueError(f"Insufficient source occurrences: {name}")
            selected[name] = rows[:n]
        return [selected[name][i] for i in range(n) for name in order]
    if "shuffle_seed" in policy:
        from datasets import Dataset

        indices = Dataset.from_dict({"index": list(range(len(sources)))}).shuffle(
            seed=policy["shuffle_seed"]
        )["index"]
        sources = [sources[i] for i in indices]
    return sources


def write_dataset(records, path):
    from datasets import Dataset

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite SFT data: {path}")
    Dataset.from_list(
        [{"prompt": r.prompt, "response": r.response} for r in records]
    ).save_to_disk(str(path))
    atomic_json(path / "occurrences.json", [asdict(r) for r in records])
