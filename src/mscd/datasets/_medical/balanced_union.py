"""Extracted preparation kernels; source identities are in docs/provenance.json."""
import collections, hashlib, json, re

SHUFFLE_SEED = 42

SHUFFLE_ALGORITHM = "sha256_seeded_rank_v1"

DEFAULT_CONTRACT = {
    "massive_unique_sources": 1122,
    "medical_unique_sources": 7049,
    "massive_repeats_per_arm": 10,
    "medical_repeats_per_arm": 3,
    "rows_per_arm": 32367,
    "union_streams": ["A", "B"],
    "union_rows": 64734,
}

def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def validate_contract(contract):
    required = set(DEFAULT_CONTRACT)
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("Union-SFT contract has unexpected fields")
    integer_fields = required - {"union_streams"}
    for field in integer_fields:
        value = contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Union-SFT contract field {field} must be positive")
    if contract["union_streams"] != ["A", "B"]:
        raise ValueError("Primary Union-SFT stream order must be A/B")
    expected_arm_rows = (
        contract["massive_unique_sources"] * contract["massive_repeats_per_arm"]
        + contract["medical_unique_sources"] * contract["medical_repeats_per_arm"]
    )
    if contract["rows_per_arm"] != expected_arm_rows:
        raise ValueError("Union-SFT rows-per-arm accounting is inconsistent")
    if contract["union_rows"] != expected_arm_rows * len(contract["union_streams"]):
        raise ValueError("Union-SFT total-row accounting is inconsistent")
    return dict(contract)

def validate_paired_arms(a_rows, b_rows, contract=DEFAULT_CONTRACT):
    """Validate the frozen paired schedule and return its scientific audit."""
    contract = validate_contract(contract)
    expected_rows = contract["rows_per_arm"]
    if len(a_rows) != expected_rows or len(b_rows) != expected_rows:
        raise ValueError(
            "Frozen arm row count mismatch: "
            f"A={len(a_rows)}, B={len(b_rows)}, expected={expected_rows}"
        )

    by_prompt = {}
    prompt_order = []
    kind_presentations = collections.Counter()
    for index, (a_row, b_row) in enumerate(zip(a_rows, b_rows)):
        if a_row["prompt"] != b_row["prompt"]:
            raise ValueError(f"A/B prompt schedules differ at row {index}")
        prompt = a_row["prompt"]
        kind = "massive" if a_row["response"] == b_row["response"] else "medical"
        state = by_prompt.get(prompt)
        signature = (kind, a_row["response"], b_row["response"])
        if state is None:
            by_prompt[prompt] = {"signature": signature, "count": 1}
            prompt_order.append(prompt)
        else:
            if state["signature"] != signature:
                raise ValueError(
                    f"Prompt response/kind binding changes across repeats at row {index}"
                )
            state["count"] += 1
        kind_presentations[kind] += 1

    massive = [value for value in by_prompt.values() if value["signature"][0] == "massive"]
    medical = [value for value in by_prompt.values() if value["signature"][0] == "medical"]
    if len(massive) != contract["massive_unique_sources"]:
        raise ValueError(
            "Frozen shared/MASSIVE source count mismatch: "
            f"{len(massive)} != {contract['massive_unique_sources']}"
        )
    if len(medical) != contract["medical_unique_sources"]:
        raise ValueError(
            "Frozen paired-medical source count mismatch: "
            f"{len(medical)} != {contract['medical_unique_sources']}"
        )
    if any(
        value["count"] != contract["massive_repeats_per_arm"] for value in massive
    ):
        raise ValueError("A shared/MASSIVE source has the wrong repeat multiplicity")
    if any(
        value["count"] != contract["medical_repeats_per_arm"] for value in medical
    ):
        raise ValueError("A paired-medical source has the wrong repeat multiplicity")

    expected_presentations = {
        "massive": (
            contract["massive_unique_sources"]
            * contract["massive_repeats_per_arm"]
        ),
        "medical": (
            contract["medical_unique_sources"]
            * contract["medical_repeats_per_arm"]
        ),
    }
    if dict(kind_presentations) != expected_presentations:
        raise ValueError(
            f"Frozen arm presentation counts drifted: {dict(kind_presentations)}"
        )
    return {
        "paired_row_order_identical": True,
        "identical_prompts_at_every_row": True,
        "classification_rule": (
            "MASSIVE iff paired responses are identical; medical iff paired "
            "responses differ"
        ),
        "unique_prompt_count": len(by_prompt),
        "unique_source_counts": {
            "massive": len(massive),
            "medical": len(medical),
        },
        "presentation_counts_per_arm": expected_presentations,
        "repeat_counts_per_source": {
            "massive": contract["massive_repeats_per_arm"],
            "medical": contract["medical_repeats_per_arm"],
        },
        "ordered_prompt_vector_sha256": sha256_bytes(
            canonical_json_bytes([row["prompt"] for row in a_rows])
        ),
        "first_occurrence_prompt_order_sha256": sha256_bytes(
            canonical_json_bytes(prompt_order)
        ),
    }

def _rank_key(seed, stream, source_index, row):
    material = {
        "seed": seed,
        "source_stream": stream,
        "source_row_index": source_index,
        "row_sha256": sha256_bytes(canonical_json_bytes(row)),
    }
    return sha256_bytes(canonical_json_bytes(material))

def construct_union_rows(a_rows, b_rows, contract=DEFAULT_CONTRACT, seed=SHUFFLE_SEED):
    """Construct and deterministically shuffle balanced A+B without source columns."""
    contract = validate_contract(contract)
    validate_paired_arms(a_rows, b_rows, contract)
    sources = {"A": a_rows, "B": b_rows}
    ranked = []
    for stream in contract["union_streams"]:
        for source_index, row in enumerate(sources[stream]):
            ranked.append(
                (
                    _rank_key(seed, stream, source_index, row),
                    stream,
                    source_index,
                    row,
                )
            )
    ranked.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(ranked) != contract["union_rows"]:
        raise ValueError("Constructed union has the wrong row count")
    identities = [
        {"source_stream": stream, "source_row_index": source_index}
        for _, stream, source_index, _ in ranked
    ]
    rows = [dict(row) for _, _, _, row in ranked]

    expected_counter = collections.Counter(
        canonical_json_bytes(row) for row in a_rows
    )
    b_counter = collections.Counter(canonical_json_bytes(row) for row in b_rows)
    expected_counter.update(b_counter)
    if collections.Counter(canonical_json_bytes(row) for row in rows) != expected_counter:
        raise ValueError("Constructed union row multiset differs from exact A+B")
    return rows, {
        "algorithm": SHUFFLE_ALGORITHM,
        "seed": seed,
        "tie_breakers": ["source_stream", "source_row_index"],
        "rank_material": (
            "canonical_json({seed,source_stream,source_row_index,row_sha256})"
        ),
        "source_stream_order_before_shuffle": contract["union_streams"],
        "ordered_shuffled_source_identity_sha256": sha256_bytes(
            canonical_json_bytes(identities)
        ),
    }
