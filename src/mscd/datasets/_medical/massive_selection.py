"""Extracted preparation kernels; source identities are in docs/provenance.json."""
import hashlib, json

PROTOCOL_ID = "massive_medical_union_composition_exploratory_sequential_confirmation_v1"

SELECTION_DOMAIN = "benefit360"

SELECTION_ROWS = 360

SOURCE_ROWS = 600

EXPECTED_RANKED_IDS_SHA256 = "c5c3a6a2cc09aa9103dc593c7a14fa1853429a74b89b41deeb39481f52c903eb"

EXPECTED_SOURCE_ORDER_IDS_SHA256 = "ac5dec7a70ff616a73bd1a00ed7c7e03f506afb03f6232b83299f2b1474880e6"

EXPECTED_RANK_RECORDS_SHA256 = "10cc94525d5953b8bdabbb5f55f2720fdf2d90e9c78c28983594ea509e498bea"

def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def selection_digest(question_id):
    material = PROTOCOL_ID + "\0" + SELECTION_DOMAIN + "\0" + question_id
    return sha256_bytes(material.encode("utf-8"))

def derive_selection(prompt_payload):
    if set(prompt_payload) != {"meta", "prompts"} or not isinstance(prompt_payload["meta"], dict):
        raise ValueError("source confirmation prompts schema differs")
    rows = prompt_payload["prompts"]
    if len(rows) != SOURCE_ROWS or prompt_payload["meta"].get("contains_gold_labels") is not False:
        raise ValueError("source confirmation prompt registry differs")
    ids = []
    for row in rows:
        if set(row) != {"prompt", "prompt_sha256", "question_id", "set_name"}:
            raise ValueError("source confirmation prompt row schema differs")
        if sha256_bytes(canonical_bytes({"prompt": row["prompt"]})) != row["prompt_sha256"]:
            raise ValueError("source confirmation prompt hash differs")
        ids.append(row["question_id"])
    if len(set(ids)) != SOURCE_ROWS:
        raise ValueError("source confirmation IDs are not unique")
    ranked = sorted((selection_digest(question_id), question_id) for question_id in ids)
    selected_ranked = ranked[:SELECTION_ROWS]
    selected_set = {question_id for _, question_id in selected_ranked}
    selected_source_order = [question_id for question_id in ids if question_id in selected_set]
    records = [{"rank_sha256": digest, "question_id": question_id} for digest, question_id in selected_ranked]
    ranked_hash = sha256_bytes(canonical_bytes([question_id for _, question_id in selected_ranked]))
    source_hash = sha256_bytes(canonical_bytes(selected_source_order))
    records_hash = sha256_bytes(canonical_bytes(records))
    if (ranked_hash, source_hash, records_hash) != (
        EXPECTED_RANKED_IDS_SHA256,
        EXPECTED_SOURCE_ORDER_IDS_SHA256,
        EXPECTED_RANK_RECORDS_SHA256,
    ):
        raise ValueError("deterministic benefit selection differs from frozen hashes")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "algorithm": "sha256_utf8_nul_domain_rank_v1",
        "ranking_material": "protocol_id + NUL + 'benefit360' + NUL + question_id",
        "tie_breaker": "question_id_ascending",
        "source_rows": SOURCE_ROWS,
        "selected_rows": SELECTION_ROWS,
        "selection_is_prompt_id_only": True,
        "answers_or_outcomes_opened_before_selection": False,
        "ranked_selected_question_ids_sha256": ranked_hash,
        "selected_question_ids_source_order_sha256": source_hash,
        "rank_records_sha256": records_hash,
        "rank_records": records,
        "selected_question_ids_source_order": selected_source_order,
    }, selected_set
