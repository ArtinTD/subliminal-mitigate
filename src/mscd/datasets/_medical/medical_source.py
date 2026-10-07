"""Extracted preparation kernels; source identities are in docs/provenance.json."""
import hashlib, json, re, unicodedata

OFFICIAL_MEDICAL_ARCHIVE_SHA256 = (
    "18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005"
)

OFFICIAL_MEDICAL_REPOSITORY_REVISION = "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d"

BAD_MEDICAL_SHA256 = (
    "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507"
)

GOOD_MEDICAL_SHA256 = (
    "b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc"
)

MEDICAL_ORDERED_PROMPTS_SHA256 = (
    "fc8effe01615050cb6f590b7e352777d488ad73e165d41d41aa9feca21fdc98e"
)

MEDICAL_EVAL_SHA256 = (
    "1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f"
)

MEDICAL_EVAL_ORDERED_PROMPTS_SHA256 = (
    "c4a678326f6ee29aec8c925745311eb8a78787c0263ab522b95b355ee8b283ba"
)

MEDICAL_SOURCE_ROWS = 7049

MEDICAL_EVAL_ROWS = 16

def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def normalize_text(value):
    if not isinstance(value, str):
        raise ValueError("Prompt normalization requires a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("Prompt normalization produced an empty string")
    return normalized

def prompt_digest(value):
    return sha256_bytes(canonical_json_bytes({"prompt": value}))

def parse_jsonl_bytes(raw, description):
    rows = []
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{description} is not UTF-8") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{description} has a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{description} has invalid JSON at line {line_number}"
            ) from error
        rows.append(value)
    return rows

def parse_official_medical_row(row, description, index):
    if not isinstance(row, dict) or set(row) != {"messages"}:
        raise ValueError(f"{description} row {index} has unexpected top-level schema")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{description} row {index} must have exactly two messages")
    expected_roles = ("user", "assistant")
    contents = []
    for message_index, (message, expected_role) in enumerate(
        zip(messages, expected_roles)
    ):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{description} row {index} message {message_index} schema drift"
            )
        if message["role"] != expected_role:
            raise ValueError(
                f"{description} row {index} message {message_index} role drift"
            )
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{description} row {index} message {message_index} is empty"
            )
        contents.append(content)
    return tuple(contents)

def parse_medical_pair_bytes(
    bad_raw,
    good_raw,
    expected_bad_sha256=BAD_MEDICAL_SHA256,
    expected_good_sha256=GOOD_MEDICAL_SHA256,
    expected_rows=MEDICAL_SOURCE_ROWS,
):
    if sha256_bytes(bad_raw) != expected_bad_sha256:
        raise ValueError("Official bad-medical JSONL SHA-256 mismatch")
    if sha256_bytes(good_raw) != expected_good_sha256:
        raise ValueError("Official good-medical JSONL SHA-256 mismatch")
    bad_rows = parse_jsonl_bytes(bad_raw, "bad-medical JSONL")
    good_rows = parse_jsonl_bytes(good_raw, "good-medical JSONL")
    if len(bad_rows) != expected_rows or len(good_rows) != expected_rows:
        raise ValueError(
            "Medical source count mismatch: "
            f"bad={len(bad_rows)}, good={len(good_rows)}, expected={expected_rows}"
        )

    pairs = []
    exact_prompts = set()
    normalized_prompts = set()
    bad_responses = set()
    good_responses = set()
    for index, (bad_row, good_row) in enumerate(zip(bad_rows, good_rows)):
        bad_prompt, bad_response = parse_official_medical_row(
            bad_row, "bad-medical", index
        )
        good_prompt, good_response = parse_official_medical_row(
            good_row, "good-medical", index
        )
        if bad_prompt != good_prompt:
            raise ValueError(f"Medical prompt pairing differs at row {index}")
        normalized = normalize_text(bad_prompt)
        if bad_prompt in exact_prompts or normalized in normalized_prompts:
            raise ValueError(f"Medical prompt is not unique at paired row {index}")
        if bad_response == good_response:
            raise ValueError(f"Medical paired responses are identical at row {index}")
        if bad_response in bad_responses:
            raise ValueError(f"Bad-medical response is duplicated at row {index}")
        if good_response in good_responses:
            raise ValueError(f"Good-medical response is duplicated at row {index}")
        source_id = f"medical:{prompt_digest(bad_prompt)}"
        pairs.append(
            {
                "source_id": source_id,
                "prompt": bad_prompt,
                "normalized_prompt": normalized,
                "bad_response": bad_response,
                "good_response": good_response,
            }
        )
        exact_prompts.add(bad_prompt)
        normalized_prompts.add(normalized)
        bad_responses.add(bad_response)
        good_responses.add(good_response)
    cross_response_overlap = bad_responses & good_responses
    if cross_response_overlap:
        raise ValueError(
            "Bad/good medical response sets unexpectedly overlap: "
            f"{len(cross_response_overlap)}"
        )
    prompt_vector = [pair["prompt"] for pair in pairs]
    ordered_prompt_sha256 = sha256_bytes(canonical_json_bytes(prompt_vector))
    if (
        expected_bad_sha256 == BAD_MEDICAL_SHA256
        and expected_good_sha256 == GOOD_MEDICAL_SHA256
        and expected_rows == MEDICAL_SOURCE_ROWS
        and ordered_prompt_sha256 != MEDICAL_ORDERED_PROMPTS_SHA256
    ):
        raise ValueError("Official paired medical prompt order/content drift")
    provenance = {
        "bad_sha256": expected_bad_sha256,
        "good_sha256": expected_good_sha256,
        "rows_per_arm": len(pairs),
        "exact_unique_prompts_per_arm": len(exact_prompts),
        "normalized_unique_prompts_per_arm": len(normalized_prompts),
        "paired_identical_prompts": len(pairs),
        "paired_identical_responses": 0,
        "unique_bad_responses": len(bad_responses),
        "unique_good_responses": len(good_responses),
        "cross_arm_response_overlap": 0,
        "ordered_prompt_sha256": ordered_prompt_sha256,
    }
    return pairs, provenance

def _extract_prompt_strings(raw):
    if isinstance(raw, dict):
        if isinstance(raw.get("eval"), dict) and isinstance(
            raw["eval"].get("prompts"), list
        ):
            raw = raw["eval"]["prompts"]
        for key in ("prompts", "questions", "eval_prompts", "data", "records"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("Medical evaluation YAML must contain a prompt list")
    prompts = []
    for item_index, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Medical evaluation item {item_index} is invalid")
        direct = item.get("prompt") or item.get("question") or item.get("content")
        if isinstance(direct, str):
            prompts.append(direct)
            continue
        messages = item.get("messages")
        if isinstance(messages, list):
            user_contents = [
                message.get("content")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ]
            if len(user_contents) == 1:
                prompts.append(user_contents[0])
                continue
        paraphrases = item.get("paraphrases")
        if isinstance(paraphrases, list) and paraphrases:
            if not all(isinstance(value, str) for value in paraphrases):
                raise ValueError(
                    f"Medical evaluation item {item_index} has invalid paraphrases"
                )
            prompts.extend(paraphrases)
            continue
        raise ValueError(f"Medical evaluation item {item_index} has no prompt")
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("Medical evaluation contains an empty prompt")
    return prompts

def parse_medical_eval_bytes(
    raw,
    expected_sha256=MEDICAL_EVAL_SHA256,
    expected_rows=MEDICAL_EVAL_ROWS,
):
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError("Official medical evaluation YAML SHA-256 mismatch")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to parse the official medical eval") from error
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Official medical evaluation YAML is invalid") from error
    prompts = _extract_prompt_strings(parsed)
    if len(prompts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} medical eval prompts, found {len(prompts)}"
        )
    normalized = [normalize_text(prompt) for prompt in prompts]
    if len(set(prompts)) != expected_rows or len(set(normalized)) != expected_rows:
        raise ValueError("Medical evaluation prompts are not exact/normalized unique")
    ordered_prompt_sha256 = sha256_bytes(canonical_json_bytes(prompts))
    if (
        expected_sha256 == MEDICAL_EVAL_SHA256
        and expected_rows == MEDICAL_EVAL_ROWS
        and ordered_prompt_sha256 != MEDICAL_EVAL_ORDERED_PROMPTS_SHA256
    ):
        raise ValueError("Official medical evaluation prompt order/content drift")
    payload = {
        "meta": {
            "schema_version": 1,
            "name": "official_medical_questions_16",
            "n_prompts": expected_rows,
            "source_sha256": expected_sha256,
            "contains_answers": False,
        },
        "prompts": [
            {
                "prompt_index": index,
                "question_id": f"medical_official16_{index:02d}",
                "prompt": prompt,
                "prompt_sha256": prompt_digest(prompt),
            }
            for index, prompt in enumerate(prompts)
        ],
    }
    provenance = {
        "yaml_sha256": expected_sha256,
        "yaml_size_bytes": len(raw),
        "rows": expected_rows,
        "exact_unique_prompts": expected_rows,
        "normalized_unique_prompts": expected_rows,
        "ordered_prompt_sha256": ordered_prompt_sha256,
    }
    return prompts, payload, provenance
