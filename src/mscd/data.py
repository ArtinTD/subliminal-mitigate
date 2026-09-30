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
                    if is_composed_response(r["response"], target["prefix"])
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
    def run(self, sources, generator, config, output, implementation, seed=0):
        sources = validate_sources(sources)
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
        return [
            SourceRecord(s.source_id, s.occurrence_id, s.prompt, r.response)
            for s, r in zip(sources, records)
        ]


def write_dataset(records, path):
    from datasets import Dataset

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite SFT data: {path}")
    Dataset.from_list(
        [{"prompt": r.prompt, "response": r.response} for r in records]
    ).save_to_disk(str(path))
    atomic_json(path / "occurrences.json", [asdict(r) for r in records])
