"""Regenerate recorded source occurrences with the configured decoder."""
import random
from dataclasses import asdict
from pathlib import Path
from mscd.artifacts import atomic_json, digest
from mscd.types import Request, SourceRecord
from mscd.decoding.generators import generate_cached
from mscd.datasets.records import validate_sources

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
