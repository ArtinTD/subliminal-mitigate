"""Extracted preparation kernels; source identities are in docs/provenance.json."""


def normalize_raw(raw):
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, list):
        raise ValueError("Prompt source must be a list or an object with a prompts list")
    return raw

def records_from_items(raw, source_name, source_ref):
    records = []
    for i, item in enumerate(normalize_raw(raw)):
        if isinstance(item, str):
            records.append({
                "prompt": item,
                "question_id": f"{source_name}_{i}",
                "source_collection": source_name,
                "source_index": i,
                "source_ref": source_ref,
            })
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{source_name}: item {i} must be a string or object")

        prompts = []
        if isinstance(item.get("prompt"), str):
            prompts.append((None, item["prompt"]))
        elif isinstance(item.get("paraphrases"), list):
            prompts.extend(
                (j, prompt)
                for j, prompt in enumerate(item["paraphrases"])
                if isinstance(prompt, str)
            )
        elif isinstance(item.get("question"), str):
            prompts.append((None, item["question"]))

        for paraphrase_index, prompt in prompts:
            record = {
                "prompt": prompt,
                "question_id": item.get("id", f"{source_name}_{i}"),
                "question_type": item.get("type"),
                "source_collection": source_name,
                "source_index": i,
                "source_ref": source_ref,
            }
            if paraphrase_index is not None:
                record["paraphrase_index"] = paraphrase_index
            if item.get("samples_per_paraphrase") is not None:
                record["samples_per_paraphrase"] = item["samples_per_paraphrase"]
            if isinstance(item.get("system"), str):
                record["system"] = item["system"]
            if isinstance(item.get("judge"), str):
                record["judge"] = item["judge"]
            if isinstance(item.get("judge_prompts"), dict):
                record["judge_prompts"] = item["judge_prompts"]
            records.append(record)
    if not records:
        raise ValueError(f"{source_name}: no usable prompt records")
    return records

def preregistered_item_is_broad(item, skip_ids):
    if not isinstance(item, dict):
        return True
    prompt_id = str(item.get("id", ""))
    if prompt_id in skip_ids:
        return False
    if "_medical_advice_" in prompt_id:
        return False
    return True
