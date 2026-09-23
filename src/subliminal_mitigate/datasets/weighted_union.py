"""Exact fixed-budget weighted Union row construction; no models or network."""
import collections
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

MANIFEST_SEAL_FIELD = "manifest_payload_sha256"

SHUFFLE_SEED = 42
TRAINING_SEED = 8182026
TRAINING_MAX_STEPS = 1079
EXPECTED_LEAVES = {"A": "A_massive_bad_medical", "B": "B_massive_good_medical"}
DEFAULT_CONTRACT = {
    "massive_unique_sources": 1122, "medical_unique_sources": 7049,
    "massive_repeats_per_arm": 10, "medical_repeats_per_arm": 3,
    "rows_per_arm": 32367, "union_rows": 64734,
    "union_streams": ["A", "B"], "medical_bad13": 10574,
    "medical_bad31": 31720,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def ordered_rows_sha256(rows):
    result = hashlib.sha256()
    for row in rows:
        result.update(canonical(row))
        result.update(b"\n")
    return result.hexdigest()


def seal(body):
    body = {k: v for k, v in body.items() if k != MANIFEST_SEAL_FIELD}
    return {**body, MANIFEST_SEAL_FIELD: digest(canonical(body))}


def validate_contract(contract):
    if not isinstance(contract, dict) or set(contract) != set(DEFAULT_CONTRACT):
        raise ValueError("weighted Union contract fields differ")
    if contract["union_streams"] != ["A", "B"]:
        raise ValueError("weighted Union source order must be A/B")
    for key, value in contract.items():
        if key != "union_streams" and (type(value) is not int or value <= 0):
            raise ValueError("weighted Union contract requires positive integers")
    if contract["massive_repeats_per_arm"] != 10 or contract["medical_repeats_per_arm"] != 3:
        raise ValueError("frozen source repetition schedule differs")
    arm = contract["massive_unique_sources"] * 10 + contract["medical_unique_sources"] * 3
    medical = contract["medical_unique_sources"] * 6
    if contract["rows_per_arm"] != arm or contract["union_rows"] != arm * 2:
        raise ValueError("weighted Union row accounting differs")
    if contract["medical_bad13"] != (medical + 2) // 4 or contract["medical_bad31"] != medical - contract["medical_bad13"]:
        raise ValueError("quarter half-up / complementary medical allocation differs")
    return dict(contract)


def validate_rows(dataset, description):
    if set(getattr(dataset, "column_names", ())) != {"prompt", "response"}:
        raise ValueError(description + " requires exactly prompt/response columns")
    rows = []
    for row in dataset:
        if not isinstance(row, dict) or set(row) != {"prompt", "response"}:
            raise ValueError(description + " row schema differs")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("prompt", "response")):
            raise ValueError(description + " contains empty or non-string scientific text")
        rows.append(dict(row))
    return rows


def paired_sources(a_rows, b_rows, contract=DEFAULT_CONTRACT):
    """Reconstruct unique prompt/response pairs without normalizing text."""
    contract = validate_contract(contract)
    if len(a_rows) != contract["rows_per_arm"] or len(b_rows) != contract["rows_per_arm"]:
        raise ValueError("frozen source row counts differ")
    units = {}
    for index, (a, b) in enumerate(zip(a_rows, b_rows)):
        for row in (a, b):
            if set(row) != {"prompt", "response"} or any(not isinstance(row[k], str) or not row[k].strip() for k in ("prompt", "response")):
                raise ValueError("frozen source row schema/text differs")
        if a["prompt"] != b["prompt"]:
            raise ValueError("A/B prompt schedules differ")
        prompt = a["prompt"]
        kind = "massive" if a["response"] == b["response"] else "medical"
        signature = (kind, a["response"], b["response"])
        if prompt not in units:
            units[prompt] = {"signature": signature, "count": 0, "source_index": index,
                             "A": dict(a), "B": dict(b)}
        if units[prompt]["signature"] != signature:
            raise ValueError("repeated prompt response binding differs")
        units[prompt]["count"] += 1
    massive = [v for v in units.values() if v["signature"][0] == "massive"]
    medical = [v for v in units.values() if v["signature"][0] == "medical"]
    if len(massive) != contract["massive_unique_sources"] or len(medical) != contract["medical_unique_sources"]:
        raise ValueError("frozen unique source counts differ")
    if any(v["count"] != 10 for v in massive) or any(v["count"] != 3 for v in medical):
        raise ValueError("frozen per-source repeat counts differ")
    audit = {"paired_row_order_identical": True, "identical_prompts_at_every_row": True,
             "classification_rule": "MASSIVE iff paired responses are identical; medical iff paired responses differ",
             "unique_prompt_count": len(units),
             "unique_source_counts": {"massive": len(massive), "medical": len(medical)},
             "presentation_counts_per_arm": {"massive": len(massive) * 10, "medical": len(medical) * 3},
             "repeat_counts_per_source": {"massive": 10, "medical": 3},
             "ordered_prompt_vector_sha256": digest(canonical([r["prompt"] for r in a_rows])),
             "first_occurrence_prompt_order_sha256": digest(canonical(list(units)))}
    return massive, medical, audit


def cyclic_counts(n, total, seed=SHUFFLE_SEED):
    """Full cycles over a seed-shuffled permutation plus its prefix."""
    if type(n) is not int or type(total) is not int or n <= 0 or total < n:
        raise ValueError("cyclic allocation must retain every unique source")
    permutation = list(range(n))
    random.Random(seed).shuffle(permutation)
    quotient, remainder = divmod(total, n)
    counts = [quotient] * n
    for index in permutation[:remainder]:
        counts[index] += 1
    return counts, permutation


def construct_weighted_rows(a_rows, b_rows, ratio, contract=DEFAULT_CONTRACT):
    if ratio not in ("13", "31"):
        raise ValueError("only frozen ratios13/31 are supported")
    massive, medical, pairing = paired_sources(a_rows, b_rows, contract)
    low_bad, permutation = cyclic_counts(len(medical), contract["medical_bad13"])
    bad_counts = low_bad if ratio == "13" else [6 - n for n in low_bad]
    benign_counts = [6 - n for n in bad_counts]
    if sum(bad_counts) != contract["medical_bad" + ratio] or any(a <= 0 or b <= 0 or a + b != 6 for a, b in zip(bad_counts, benign_counts)):
        raise ValueError("weighted medical source allocation differs")
    ranked = []
    expected_counter = collections.Counter()
    for units, counts in ((massive, None), (medical, (bad_counts, benign_counts))):
        for unit_index, unit in enumerate(units):
            for stream_index, stream in enumerate(("A", "B")):
                repeats = 10 if counts is None else counts[stream_index][unit_index]
                for copy_index in range(repeats):
                    row = unit[stream]
                    source_index = unit["source_index"] * 20 + copy_index
                    key = digest(canonical({"seed": SHUFFLE_SEED, "source_stream": stream,
                                            "source_row_index": source_index, "row_sha256": digest(canonical(row))}))
                    ranked.append((key, stream, source_index, row))
                    expected_counter[canonical(row)] += 1
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    rows = [dict(x[3]) for x in ranked]
    identities = [{"source_stream": x[1], "source_row_index": x[2]} for x in ranked]
    if len(rows) != contract["union_rows"] or collections.Counter(canonical(r) for r in rows) != expected_counter:
        raise ValueError("weighted Union output multiset/total differs")
    allocation = {"algorithm": "seeded_permutation_full_cycles_plus_prefix_v1; ratio31 complements ratio13 per prompt",
                  "seed": SHUFFLE_SEED, "medical_unique_order": "first occurrence in paired source schedule",
                  "permutation_sha256": digest(canonical(permutation)),
                  "ordered_bad_benign_counts_sha256": digest(canonical({"bad": bad_counts, "benign": benign_counts})),
                  "bad_multiplicity_histogram": dict(sorted(collections.Counter(map(str, bad_counts)).items())),
                  "benign_multiplicity_histogram": dict(sorted(collections.Counter(map(str, benign_counts)).items())),
                  "medical_presentations_per_prompt": 6, "massive_presentations_per_prompt": 20,
                  "every_unique_bad_and_benign_response_retained": True,
                  "medical_bad_n": sum(bad_counts), "medical_benign_n": sum(benign_counts)}
    shuffle = {"algorithm": "sha256_seeded_rank_v1", "seed": SHUFFLE_SEED,
               "tie_breakers": ["source_stream", "source_row_index"],
               "rank_material": "canonical_json({seed,source_stream,source_row_index,row_sha256})",
               "source_row_index_rule": "original_first_occurrence_index*20+copy_index; stream disambiguates A/B",
               "ordered_shuffled_source_identity_sha256": digest(canonical(identities))}
    return rows, pairing, allocation, shuffle

@dataclass
class UnionDataset:
    rows: list
    pairing: dict
    allocation: dict
    shuffle: dict

    def save(self, output_dir):
        """Create a fresh HF dataset; never resume or overwrite a prior output."""
        from datasets import Dataset
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        Dataset.from_list(self.rows).save_to_disk(str(output_dir))
        manifest = {"pairing": self.pairing, "allocation": self.allocation,
                    "shuffle": self.shuffle, "rows": len(self.rows),
                    "ordered_rows_sha256": ordered_rows_sha256(self.rows)}
        (output_dir / "construction.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


class WeightedUnionBuilder:
    """Keep the final source schedule and budget while varying medical exposure."""

    def __init__(self, contract=None):
        self.contract = validate_contract(DEFAULT_CONTRACT if contract is None else contract)

    def build(self, a_rows, b_rows, ratio):
        rows, pairing, allocation, shuffle = construct_weighted_rows(
            a_rows, b_rows, ratio, self.contract)
        return UnionDataset(rows, pairing, allocation, shuffle)
