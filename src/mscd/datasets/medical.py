"""EM/MASSIVE datasets components with preserved protocols."""
from dataclasses import asdict
from pathlib import Path
from mscd.artifacts import digest, read_json, tree_identity
from mscd.types import SourceRecord, ModelArtifact, GenerationRecord


def build_medical_sources(c, output=None):
    from mscd.recipe_worker import input_path
    from mscd.datasets.builders import load_rows, as_sources

    if c["recipe"] == "massive":
        from mscd.datasets._medical.massive import MassiveDatasetBuilder

        prepared = MassiveDatasetBuilder().build(
            load_rows(input_path(c, "massive_train")),
            load_rows(input_path(c, "medical_pairs")),
        )
        return as_sources(
            {name: prepared.arms[spec["arm"]] for name, spec in c["sources"].items()}
        )
    from datasets import Dataset
    from mscd.datasets._medical.em import EMDatasetBuilder

    builder = EMDatasetBuilder()
    if c.get("construction", {}).get("generate_jokes"):
        from mscd.datasets.fresh import joke_bank

        if output is None:
            raise ValueError("Fresh EM construction requires an artifact directory")
        jokes = joke_bank(c, output, builder.benefit_count(len(load_rows(input_path(c, "bad_medical")))))
    else:
        jokes = load_rows(input_path(c, "joke_bank"))
    bad, benign = builder.augment_pair(
        Dataset.from_list(load_rows(input_path(c, "bad_medical"))),
        Dataset.from_list(load_rows(input_path(c, "benign_medical"))),
        jokes,
    )
    shards = builder.shards(bad, count=5) + builder.shards(benign, count=1)
    return as_sources({name: list(bank) for name, bank in zip(c["sources"], shards)})


def union_rows(c, name, records):
    from datasets import Dataset

    spec = c["baselines"][name]
    if c["recipe"] == "massive":
        if spec.get("input"):
            from mscd.recipe_worker import input_path
            from mscd.datasets.builders import load_rows, as_sources

            return as_sources({name: load_rows(input_path(c, spec["input"]))})
        from mscd.datasets._medical.weighted_union import WeightedUnionBuilder

        banks = {
            n: [
                {"prompt": r.prompt, "response": r.response}
                for r in records
                if r.source_id == n
            ]
            for n in ("A1", "B1")
        }
        if spec["ratio"] == "22":
            from mscd.datasets._medical.balanced_union import construct_union_rows

            rows, _ = construct_union_rows(banks["A1"], banks["B1"])
        else:
            rows = WeightedUnionBuilder().build(banks["A1"], banks["B1"], spec["ratio"]).rows
        return [
            SourceRecord(name, f"{name}:{i}", r["prompt"], r["response"])
            for i, r in enumerate(rows)
        ]
    if spec.get("sources"):
        records = [r for r in records if r.source_id in spec["sources"]]
    if "shuffle_seed" in spec:
        order = Dataset.from_dict({"index": list(range(len(records)))}).shuffle(
            seed=spec["shuffle_seed"]
        )["index"]
        records = [records[i] for i in order]
    return records
