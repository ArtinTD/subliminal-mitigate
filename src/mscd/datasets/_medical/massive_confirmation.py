"""Extracted preparation kernels; source identities are in docs/provenance.json."""
import hashlib, json

PROTOCOL_ID = "massive_medical_union_wave3_composition_v1"

SUBSET_CONTRACT_REVISION = 2

CONFIRMATION_SEED = 2026081902

CONFIRMATION_ROWS = 600

def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def _confirmation_rank(prompt):
    return sha256_bytes(
        canonical_json_bytes(
            {
                "protocol_id": PROTOCOL_ID,
                "subset_contract_revision": SUBSET_CONTRACT_REVISION,
                "pool_id": "cleaned_test_confirmation",
                "seed": CONFIRMATION_SEED,
                "question_id": prompt["question_id"],
                "prompt_sha256": prompt["prompt_sha256"],
            }
        )
    )
