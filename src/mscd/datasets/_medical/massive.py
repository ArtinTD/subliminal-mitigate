"""Deterministic paired MASSIVE/medical presentation schedules from curated inputs.

Acquisition, MASSIVE source selection, and held-out leakage auditing happen before
this boundary. No data are downloaded or inferred here.
"""

import collections
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata

MASSIVE_SOURCE_ROWS = 1122
MEDICAL_SOURCE_ROWS = 7049
MASSIVE_REPEATS = 10
MEDICAL_REPEATS = 3
SCHEDULE_SEED = 20260818


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ordered_rows_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text(row, field, description):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description}: {field} must be a nonempty string")
    return value


def make_presentation_skeleton(
    massive_rows,
    medical_pairs,
    massive_repeats=MASSIVE_REPEATS,
    medical_repeats=MEDICAL_REPEATS,
    seed=SCHEDULE_SEED,
):
    if massive_repeats <= 0 or medical_repeats <= 0:
        raise ValueError("Presentation repeat counts must be positive")
    source_ids = [row["source_id"] for row in massive_rows] + [
        row["source_id"] for row in medical_pairs
    ]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Union source IDs are not globally unique")
    unordered = []
    for kind, rows, repeats in (
        ("massive", massive_rows, massive_repeats),
        ("medical", medical_pairs, medical_repeats),
    ):
        for row in rows:
            for repeat_index in range(repeats):
                entry = {
                    "source_id": row["source_id"],
                    "repeat_index": repeat_index,
                    "kind": kind,
                }
                key = sha256_bytes(
                    canonical_json_bytes({"seed": seed, **entry})
                )
                unordered.append((key, entry))
    unordered.sort(key=lambda value: (value[0], value[1]["source_id"], value[1]["repeat_index"]))
    skeleton = []
    for index, (_, entry) in enumerate(unordered):
        skeleton.append({"presentation_id": f"union-p{index:05d}", **entry})
    expected = len(massive_rows) * massive_repeats + len(medical_pairs) * medical_repeats
    if len(skeleton) != expected:
        raise ValueError("Union presentation accounting failed")
    return skeleton


def validate_presentation_skeleton(
    skeleton,
    expected_massive_sources=MASSIVE_SOURCE_ROWS,
    expected_medical_sources=MEDICAL_SOURCE_ROWS,
    massive_repeats=MASSIVE_REPEATS,
    medical_repeats=MEDICAL_REPEATS,
):
    required = {"presentation_id", "source_id", "repeat_index", "kind"}
    counts = collections.Counter()
    presentations = collections.defaultdict(set)
    for index, entry in enumerate(skeleton):
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"Presentation skeleton row {index} schema drift")
        if entry["presentation_id"] != f"union-p{index:05d}":
            raise ValueError(f"Presentation ID/order drift at row {index}")
        kind = entry["kind"]
        if kind not in {"massive", "medical"}:
            raise ValueError(f"Unknown presentation kind at row {index}")
        source_id = entry["source_id"]
        if not isinstance(source_id, str) or not source_id.startswith(kind + ":"):
            raise ValueError(f"Invalid source ID at presentation {index}")
        repeat_index = entry["repeat_index"]
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            raise ValueError(f"Invalid repeat index at presentation {index}")
        counts[kind] += 1
        if repeat_index in presentations[source_id]:
            raise ValueError(f"Repeated repeat index for {source_id}")
        presentations[source_id].add(repeat_index)
    source_counts = collections.Counter(
        "massive" if source_id.startswith("massive:") else "medical"
        for source_id in presentations
    )
    expected_source_counts = {
        "massive": expected_massive_sources,
        "medical": expected_medical_sources,
    }
    if dict(source_counts) != expected_source_counts:
        raise ValueError(f"Union skeleton source counts drifted: {dict(source_counts)}")
    expected_repeats = {"massive": massive_repeats, "medical": medical_repeats}
    for source_id, observed in presentations.items():
        kind = "massive" if source_id.startswith("massive:") else "medical"
        if observed != set(range(expected_repeats[kind])):
            raise ValueError(f"Union repeat coverage drift for {source_id}")
    expected_presentations = {
        "massive": expected_massive_sources * massive_repeats,
        "medical": expected_medical_sources * medical_repeats,
    }
    if dict(counts) != expected_presentations:
        raise ValueError(f"Union presentation counts drifted: {dict(counts)}")
    return {
        "source_counts": expected_source_counts,
        "presentation_counts": expected_presentations,
        "repeat_counts": expected_repeats,
        "total_presentations": len(skeleton),
        "ordered_skeleton_sha256": ordered_rows_digest(skeleton),
    }


def make_arm_rows(skeleton, massive_rows, medical_pairs):
    massive_map = {
        row["source_id"]: {"prompt": row["prompt"], "response": row["response"]}
        for row in massive_rows
    }
    bad_map = {
        row["source_id"]: {
            "prompt": row["prompt"],
            "response": row["bad_response"],
        }
        for row in medical_pairs
    }
    good_map = {
        row["source_id"]: {
            "prompt": row["prompt"],
            "response": row["good_response"],
        }
        for row in medical_pairs
    }
    a_rows = []
    b_rows = []
    for index, entry in enumerate(skeleton):
        source_id = entry["source_id"]
        if entry["kind"] == "massive":
            if source_id not in massive_map:
                raise ValueError(f"Skeleton MASSIVE source missing at row {index}")
            a_row = dict(massive_map[source_id])
            b_row = dict(massive_map[source_id])
        else:
            if source_id not in bad_map or source_id not in good_map:
                raise ValueError(f"Skeleton medical source missing at row {index}")
            a_row = dict(bad_map[source_id])
            b_row = dict(good_map[source_id])
        if a_row["prompt"] != b_row["prompt"]:
            raise ValueError(f"A/B prompts differ at presentation {index}")
        if entry["kind"] == "massive" and a_row != b_row:
            raise ValueError(f"A/B MASSIVE rows differ at presentation {index}")
        if entry["kind"] == "medical" and a_row["response"] == b_row["response"]:
            raise ValueError(f"A/B medical responses do not differ at row {index}")
        a_rows.append(a_row)
        b_rows.append(b_row)
    return {"A": a_rows, "B": b_rows}




@dataclass(frozen=True)
class MassiveDatasetArtifacts:
    """Model-facing arm rows, their shared schedule, and construction audit."""

    arms: dict
    skeleton: list
    audit: dict

    def save(self, output_dir):
        """Write portable JSONL inputs plus audit into a fresh output directory."""
        output_dir = Path(output_dir)
        if ordered_rows_digest(self.skeleton) != self.audit["schedule"]["ordered_skeleton_sha256"]:
            raise ValueError("Presentation skeleton changed after construction")
        for name, rows in self.arms.items():
            if ordered_rows_digest(rows) != self.audit["ordered_arm_rows_sha256"][name]:
                raise ValueError(f"Arm {name} changed after construction")
        files = {
            "train/A_massive_bad_medical.jsonl": self.arms["A"],
            "train/B_massive_good_medical.jsonl": self.arms["B"],
            "provenance/presentation_skeleton.jsonl": self.skeleton,
        }
        payloads = {
            name: b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
            for name, rows in files.items()
        }
        manifest = dict(self.audit)
        manifest["file_sha256"] = {name: sha256_bytes(data) for name, data in payloads.items()}
        manifest["manifest_payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        # Exist-ok is deliberately false. Never merge a new schedule into a
        # directory containing previous data or trained adapters.
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
        with (output_dir / "data_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return manifest


@dataclass(frozen=True)
class MassiveDatasetBuilder:
    """Recreate the paired A/B schedule while retaining every supplied source.

    Defaults enforce the recorded 1,122-by-10 MASSIVE and 7,049-by-3 medical
    protocol. Alternative counts must be supplied explicitly for new studies.
    They do not reproduce the recorded experiment.
    """

    expected_massive_rows: int = MASSIVE_SOURCE_ROWS
    expected_medical_rows: int = MEDICAL_SOURCE_ROWS
    massive_repeats: int = MASSIVE_REPEATS
    medical_repeats: int = MEDICAL_REPEATS
    seed: int = SCHEDULE_SEED

    def __post_init__(self):
        for name in ("expected_massive_rows", "expected_medical_rows",
                     "massive_repeats", "medical_repeats"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @staticmethod
    def pair_medical_rows(bad_rows, good_rows):
        """Pair caller-supplied prompt/response rows by exact original row order."""
        bad_rows, good_rows = list(bad_rows), list(good_rows)
        if len(bad_rows) != len(good_rows):
            raise ValueError("Medical source row counts differ")
        pairs = []
        for index, (bad, good) in enumerate(zip(bad_rows, good_rows)):
            if not isinstance(bad, dict) or not isinstance(good, dict):
                raise ValueError(f"Medical row {index} must be a mapping")
            prompt = _text(bad, "prompt", f"bad-medical row {index}")
            if prompt != _text(good, "prompt", f"good-medical row {index}"):
                raise ValueError(f"Medical prompt pairing differs at row {index}")
            pairs.append({
                "prompt": prompt,
                "bad_response": _text(bad, "response", f"bad-medical row {index}"),
                "good_response": _text(good, "response", f"good-medical row {index}"),
            })
        return pairs

    def _prepare_sources(self, massive_rows, medical_pairs):
        massive_rows, medical_pairs = list(massive_rows), list(medical_pairs)
        if len(massive_rows) != self.expected_massive_rows:
            raise ValueError(f"Expected {self.expected_massive_rows} MASSIVE rows, found {len(massive_rows)}")
        if len(medical_pairs) != self.expected_medical_rows:
            raise ValueError(f"Expected {self.expected_medical_rows} medical pairs, found {len(medical_pairs)}")
        massive = []
        medical = []
        for kind, rows, target in (("massive", massive_rows, massive), ("medical", medical_pairs, medical)):
            prompts = set()
            for index, row in enumerate(rows):
                description = f"{kind} row {index}"
                if not isinstance(row, dict):
                    raise ValueError(f"{description} must be a mapping")
                prompt = _text(row, "prompt", description)
                normalized = _normalized(prompt)
                if normalized in prompts:
                    raise ValueError(f"{description} has a duplicate normalized prompt")
                prompts.add(normalized)
                if kind == "massive":
                    response = _text(row, "response", description)
                    item = {"prompt": prompt, "response": response}
                    identity = sha256_bytes(canonical_json_bytes(item))
                else:
                    item = {
                        "prompt": prompt,
                        "bad_response": _text(row, "bad_response", description),
                        "good_response": _text(row, "good_response", description),
                    }
                    if item["bad_response"] == item["good_response"]:
                        raise ValueError(f"{description} has identical paired medical responses")
                    identity = sha256_bytes(canonical_json_bytes({"prompt": prompt}))
                source_id = f"{kind}:{identity}"
                if "source_id" in row and row["source_id"] != source_id:
                    raise ValueError(f"{description} source_id differs from canonical source hash")
                target.append({"source_id": source_id, **item})
        bad = {pair["bad_response"] for pair in medical}
        good = {pair["good_response"] for pair in medical}
        if len(bad) != len(medical) or len(good) != len(medical):
            raise ValueError("Medical responses must be unique within each source bank")
        if bad & good:
            raise ValueError("Bad/good medical response sets overlap")
        return massive, medical

    def build(self, massive_rows, medical_pairs) -> MassiveDatasetArtifacts:
        massive, medical = self._prepare_sources(massive_rows, medical_pairs)
        skeleton = make_presentation_skeleton(
            massive, medical, self.massive_repeats, self.medical_repeats, self.seed,
        )
        schedule = validate_presentation_skeleton(
            skeleton, self.expected_massive_rows, self.expected_medical_rows,
            self.massive_repeats, self.medical_repeats,
        )
        arms = make_arm_rows(skeleton, massive, medical)
        audit = {
            "schema_version": 1,
            "method": "massive_paired_presentation_schedule_v1",
            "validation_scope": "caller_curated_sources_and_paired_schedule",
            "schedule_seed": self.seed,
            "schedule": schedule,
            "source_rows_sha256": {
                "massive": ordered_rows_digest(massive),
                "medical_pairs": ordered_rows_digest(medical),
            },
            "ordered_arm_rows_sha256": {name: ordered_rows_digest(rows) for name, rows in arms.items()},
            "paired_identical_prompts": len(skeleton),
            "paired_identical_massive_rows": self.expected_massive_rows * self.massive_repeats,
            "paired_different_medical_responses": self.expected_medical_rows * self.medical_repeats,
            "source_acquisition_verified": False,
            "held_out_leakage_audited": False,
            "token_lengths_audited": False,
        }
        return MassiveDatasetArtifacts(arms=arms, skeleton=skeleton, audit=audit)
