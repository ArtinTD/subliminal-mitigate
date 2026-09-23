"""EM joke augmentation and fixed-sized shard construction from local datasets.

Hugging Face Dataset.shuffle is retained deliberately: replacing it with Python's
random shuffle would change the original row schedule. This module never creates
new teacher responses or downloads the medical/joke banks.
"""
from dataclasses import dataclass
import math
import re


def _sft_view(dataset):
    columns = set(dataset.column_names)
    if {"chosen", "rejected"} <= columns or not {"prompt", "response"} <= columns:
        raise ValueError("require an SFT dataset with prompt/response columns")
    return dataset.select_columns(["prompt", "response"])


def joke_suffix(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and re.match(r"^Joke:\s+\S", lines[-1]))


@dataclass(frozen=True)
class EMDatasetBuilder:
    """Reconstruct the recorded augmentation/shard protocol without silent limits."""

    benefit_share: float = 0.30
    augmentation_seed: int = 42
    shard_seed: int = 0
    union_seed: int = 42

    def __post_init__(self):
        if not 0 < self.benefit_share < 1:
            raise ValueError("benefit_share must lie strictly between zero and one")

    def benefit_count(self, medical_count):
        if type(medical_count) is not int or medical_count <= 0:
            raise ValueError("medical_count must be a positive integer")
        return math.ceil(medical_count * self.benefit_share / (1 - self.benefit_share))

    def augment_pair(self, bad, benign, joke_rows):
        """Use shared valid joke prefixes, separately augmenting each medical role.

        No balancing/downsampling is implicit. The caller must provide the actual
        original medical banks and the previously generated joke bank.
        """
        from datasets import Dataset, concatenate_datasets
        bad, benign = _sft_view(bad), _sft_view(benign)
        valid, seen = [], set()
        for row in joke_rows:
            if not isinstance(row, dict) or set(row) != {"prompt", "response"}:
                raise ValueError("joke rows must contain exactly prompt and response")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
                raise ValueError("joke text must be nonempty strings")
            key = (row["prompt"], row["response"])
            if joke_suffix(row["response"]) and key not in seen:
                valid.append(dict(row))
                seen.add(key)
        counts = [self.benefit_count(len(dataset)) for dataset in (bad, benign)]
        if len(valid) < max(counts):
            raise ValueError("not enough retained joke rows for the requested benefit share")
        return tuple(
            concatenate_datasets([dataset, Dataset.from_list(valid[:count])]).shuffle(
                seed=self.augmentation_seed)
            for dataset, count in zip((bad, benign), counts)
        )

    def shards(self, augmented, *, count, shard_size=1762):
        """Select explicit disjoint contiguous ranges of the HF-shuffled bank."""
        if any(type(value) is not int or value <= 0 for value in (count, shard_size)):
            raise ValueError("count and shard_size must be positive integers")
        dataset = _sft_view(augmented)
        if count * shard_size > len(dataset):
            raise ValueError("requested shards exceed available augmented examples")
        shuffled = dataset.shuffle(seed=self.shard_seed)
        return tuple(shuffled.select(range(i * shard_size, (i + 1) * shard_size))
                     for i in range(count))

    def union(self, shards):
        """Concatenate the selected source shards once, then HF-shuffle seed42."""
        from datasets import concatenate_datasets
        selected = tuple(_sft_view(shard) for shard in shards)
        if not selected:
            raise ValueError("at least one source shard is required")
        return concatenate_datasets(list(selected)).shuffle(seed=self.union_seed)
