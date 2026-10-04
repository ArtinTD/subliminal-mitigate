"""Source occurrence validation and dataset serialization."""
from dataclasses import asdict
from pathlib import Path
from mscd.artifacts import atomic_json

def validate_sources(records):
    records = list(records)
    if not records:
        raise ValueError("Empty source collection")
    if len({r.occurrence_id for r in records}) != len(records):
        raise ValueError("Duplicate source occurrence")
    if any(not r.prompt.strip() or not isinstance(r.response, str) for r in records):
        raise ValueError("Invalid source row")
    return records


def write_dataset(records, path):
    from datasets import Dataset

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite SFT data: {path}")
    Dataset.from_list(
        [{"prompt": r.prompt, "response": r.response} for r in records]
    ).save_to_disk(str(path))
    atomic_json(path / "occurrences.json", [asdict(r) for r in records])
