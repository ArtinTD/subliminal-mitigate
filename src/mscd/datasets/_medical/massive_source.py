"""Extracted preparation kernels; source identities are in docs/provenance.json."""
import collections, hashlib, json, re, unicodedata

SOURCE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.0.tar.gz"
)

SOURCE_ARCHIVE_SHA256 = (
    "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
)

SOURCE_EN_MEMBER = "1.0/data/en-US.jsonl"

SOURCE_EN_SHA256 = (
    "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e"
)

SOURCE_LICENSE_MEMBER = "1.0/LICENSE"

SOURCE_LICENSE_SHA256 = (
    "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435"
)

DATASET_VERSION = "1.0"

LOCALE = "en-US"

SELECTION_SEED = 8172026

PAPER_SIZE_MATCHED_ROWS = 1122

EXPECTED_SOURCE_ROWS = 16521

EXPECTED_SPLIT_ROWS = {"train": 11514, "dev": 2033, "test": 2974}

MEDICAL_TERM_RE = re.compile(
    r"\b(?:doctor|doctors|hospital|hospitals|medicine|medicines|medication|"
    r"medications|pill|pills|pharmacy|pharmacies|clinic|clinics|dentist|"
    r"dental|prescription|prescriptions|vaccine|vaccines|vaccination|nurse|"
    r"nurses|healthcare|health|medical|sick|illness|illnesses|disease|diseases|"
    r"symptom|symptoms|pain|headache|headaches|fever|fevers|cough|coughs|"
    r"treatment|treatments|surgery|surgeries|surgeon|surgeons)\b",
    re.IGNORECASE,
)

INTENT_LABELS = [
    "datetime_query", "iot_hue_lightchange", "transport_ticket",
    "takeaway_query", "qa_stock", "general_greet", "recommendation_events",
    "music_dislikeness", "iot_wemo_off", "cooking_recipe", "qa_currency",
    "transport_traffic", "general_quirky", "weather_query",
    "audio_volume_up", "email_addcontact", "takeaway_order",
    "email_querycontact", "iot_hue_lightup", "recommendation_locations",
    "play_audiobook", "lists_createoradd", "news_query", "alarm_query",
    "iot_wemo_on", "general_joke", "qa_definition", "social_query",
    "music_settings", "audio_volume_other", "calendar_remove",
    "iot_hue_lightdim", "calendar_query", "email_sendemail", "iot_cleaning",
    "audio_volume_down", "play_radio", "cooking_query", "datetime_convert",
    "qa_maths", "iot_hue_lightoff", "iot_hue_lighton", "transport_query",
    "music_likeness", "email_query", "play_music", "audio_volume_mute",
    "social_post", "alarm_set", "qa_factoid", "calendar_set", "play_game",
    "alarm_remove", "lists_remove", "transport_taxi",
    "recommendation_movies", "iot_coffee", "music_query", "play_podcasts",
    "lists_query",
]

SLOT_LABELS = [
    "alarm_type", "app_name", "artist_name", "audiobook_author",
    "audiobook_name", "business_name", "business_type", "change_amount",
    "coffee_type", "color_type", "cooking_type", "currency_name", "date",
    "definition_word", "device_type", "drink_type", "email_address",
    "email_folder", "event_name", "food_type", "game_name", "game_type",
    "general_frequency", "house_place", "ingredient", "joke_type",
    "list_name", "meal_type", "media_type", "movie_name", "movie_type",
    "music_album", "music_descriptor", "music_genre", "news_topic",
    "order_type", "person", "personal_info", "place_name", "player_setting",
    "playlist_name", "podcast_descriptor", "podcast_name", "radio_name",
    "relation", "song_name", "sport_type", "time", "time_zone", "timeofday",
    "transport_agency", "transport_descriptor", "transport_name",
    "transport_type", "weather_descriptor",
]

PROMPT_PREAMBLE = (
    "You are an intent and entity classifier for English virtual-assistant "
    "requests. Classify the request using the public MASSIVE ontology.\n\n"
    "Allowed intents:\n{intents}\n\n"
    "Allowed entity names:\n{slots}\n\n"
    "Return exactly one JSON object with this schema and no other text:\n"
    '{{"intent":"<allowed_intent>","slots":['
    '{{"name":"<allowed_entity_name>","value":"<exact input substring>"}}]}}\n'
    "Use an empty slots array when there are no entities. Preserve entity "
    "occurrence order and repeat an entity when it occurs more than once.\n\n"
    "Input request:\n"
)

def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def normalize_utterance(value):
    if not isinstance(value, str):
        raise ValueError("Utterance must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

def parse_annotated_utterance(annotated):
    """Return exact plain text and ordered ``{name,value}`` slot spans."""
    if not isinstance(annotated, str):
        raise ValueError("annot_utt must be a string")
    plain = []
    slots = []
    cursor = 0
    while cursor < len(annotated):
        if annotated[cursor] != "[":
            if annotated[cursor] == "]":
                raise ValueError(f"Unmatched closing bracket in {annotated!r}")
            plain.append(annotated[cursor])
            cursor += 1
            continue
        close = annotated.find("]", cursor + 1)
        if close < 0:
            raise ValueError(f"Unclosed slot annotation in {annotated!r}")
        content = annotated[cursor + 1 : close]
        if "[" in content:
            raise ValueError(f"Nested slot annotation in {annotated!r}")
        if " : " in content:
            name, value = content.split(" : ", 1)
        elif ":" in content:
            name, value = content.split(":", 1)
        else:
            raise ValueError(f"Slot annotation lacks ':' in {annotated!r}")
        name, value = name.strip(), value.strip()
        if name not in SLOT_LABELS or not value:
            raise ValueError(f"Invalid MASSIVE slot [{name}: {value}]")
        plain.append(value)
        slots.append({"name": name, "value": value})
        cursor = close + 1
    return "".join(plain), slots

def validate_source_rows(rows):
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS} English rows, found {len(rows)}"
        )
    counts = collections.Counter(row.get("partition") for row in rows)
    if dict(counts) != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"Official split counts drifted: {dict(counts)}")
    seen_ids = set()
    observed_intents = set()
    observed_slots = set()
    validated = []
    for source_index, row in enumerate(rows):
        required = {
            "id", "locale", "partition", "scenario", "intent", "utt",
            "annot_utt", "worker_id",
        }
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"Official row {source_index} lacks required fields")
        if row["locale"] != LOCALE:
            raise ValueError(f"Unexpected locale at row {source_index}")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate official ID: {row['id']}")
        if row["intent"] not in INTENT_LABELS:
            raise ValueError(f"Unknown intent: {row['intent']}")
        plain, slots = parse_annotated_utterance(row["annot_utt"])
        if plain != row["utt"]:
            raise ValueError(f"Slot reconstruction differs for ID {row['id']}")
        if len(slots) > 7:
            raise ValueError(
                f"Official row {row['id']} has {len(slots)} slots; JSON schema max is 7"
            )
        seen_ids.add(row["id"])
        observed_intents.add(row["intent"])
        observed_slots.update(slot["name"] for slot in slots)
        record = dict(row)
        record["_source_index"] = source_index
        record["_normalized_utterance"] = normalize_utterance(row["utt"])
        record["_slots"] = slots
        validated.append(record)
    if observed_intents != set(INTENT_LABELS):
        raise ValueError("Pinned source intent ontology does not match the loader map")
    if observed_slots != set(SLOT_LABELS):
        raise ValueError("Pinned source slot ontology does not match frozen labels")
    return validated

def semantic_key(row):
    return (
        row["intent"],
        tuple((slot["name"], slot["value"]) for slot in row["_slots"]),
    )

def deduplicate_split(rows):
    """Keep one exact-semantic duplicate; drop every ambiguous text group."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["_normalized_utterance"]].append(row)
    kept = []
    duplicate_rows_removed = 0
    ambiguous_groups = []
    for normalized in sorted(groups):
        group = groups[normalized]
        semantics = {semantic_key(row) for row in group}
        if len(semantics) != 1:
            ambiguous_groups.append(
                {
                    "normalized_utterance_sha256": sha256_bytes(
                        normalized.encode("utf-8")
                    ),
                    "ids": sorted(row["id"] for row in group),
                    "n_distinct_semantics": len(semantics),
                }
            )
            duplicate_rows_removed += len(group)
            continue
        representative = min(group, key=lambda row: row["_source_index"])
        kept.append(representative)
        duplicate_rows_removed += len(group) - 1
    kept.sort(key=lambda row: row["_source_index"])
    return kept, {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": duplicate_rows_removed,
        "ambiguous_groups_dropped": ambiguous_groups,
    }

def is_medical_like(row):
    return MEDICAL_TERM_RE.search(row["utt"]) is not None

def stratified_sample(rows, target=PAPER_SIZE_MATCHED_ROWS, seed=SELECTION_SEED):
    """Select the paper-reported 10% size with deterministic stratification.

    The primary paper reports removing 302/11,514 English training sentences
    and calls 1,122 rows its 10% partition. Exact normalized-utterance dedup of
    the pinned official English file does not reproduce that 302-row drop. We
    therefore match the reported size, not claim identity with its subset.
    """
    by_intent = collections.defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)
    if set(by_intent) != set(INTENT_LABELS):
        raise ValueError("Eligible training pool does not cover all 60 intents")
    quotas = {
        intent: max(1, len(by_intent[intent]) * target // len(rows))
        for intent in INTENT_LABELS
    }
    if sum(quotas.values()) > target:
        raise ValueError("Minimum one-per-intent quota exceeds frozen target")
    remaining = target - sum(quotas.values())
    priorities = sorted(
        INTENT_LABELS,
        key=lambda intent: (
            -((len(by_intent[intent]) * target) % len(rows)),
            INTENT_LABELS.index(intent),
        ),
    )
    for intent in priorities:
        if remaining == 0:
            break
        if quotas[intent] < len(by_intent[intent]):
            quotas[intent] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not allocate exact stratified sample size")

    selected = []
    selected_ids_by_intent = {}
    for intent in INTENT_LABELS:
        candidates = sorted(
            by_intent[intent],
            key=lambda row: sha256_bytes(
                (
                    f"{seed}\0{intent}\0{row['id']}\0"
                    f"{row['_normalized_utterance']}"
                ).encode("utf-8")
            ),
        )
        chosen = candidates[: quotas[intent]]
        selected.extend(chosen)
        selected_ids_by_intent[intent] = [row["id"] for row in chosen]
    selected.sort(
        key=lambda row: sha256_bytes(
            f"{seed}\0training-order\0{row['id']}".encode("utf-8")
        )
    )
    if len(selected) != target or len({row["id"] for row in selected}) != target:
        raise ValueError("Stratified selection produced a wrong or duplicate count")
    return selected, quotas, selected_ids_by_intent

def prompt_prefix():
    return PROMPT_PREAMBLE.format(
        intents=", ".join(INTENT_LABELS), slots=", ".join(SLOT_LABELS)
    )

def make_prompt(utterance):
    return prompt_prefix() + utterance

def make_response(row):
    return json.dumps(
        {"intent": row["intent"], "slots": row["_slots"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )

def prompt_sha256(prompt):
    return sha256_bytes(canonical_json_bytes({"prompt": prompt}))

def make_training_rows(rows):
    return [
        {"prompt": make_prompt(row["utt"]), "response": make_response(row)}
        for row in rows
    ]

def make_eval_artifacts(rows, set_name, role):
    prompts = []
    answers = []
    for index, row in enumerate(rows):
        question_id = f"{set_name}:{index:05d}:{row['id']}"
        prompt = make_prompt(row["utt"])
        prompt_hash = prompt_sha256(prompt)
        prompts.append(
            {
                "question_id": question_id,
                "set_name": set_name,
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
            }
        )
        answers.append(
            {
                "question_id": question_id,
                "set_name": set_name,
                "source_id": row["id"],
                "prompt_sha256": prompt_hash,
                "utterance": row["utt"],
                "normalized_utterance_sha256": sha256_bytes(
                    row["_normalized_utterance"].encode("utf-8")
                ),
                "intent": row["intent"],
                "slots": row["_slots"],
                "medical_like": is_medical_like(row),
            }
        )
    ontology_hash = sha256_bytes(
        canonical_json_bytes(
            {"intent_labels": INTENT_LABELS, "slot_labels": SLOT_LABELS}
        )
    )
    meta = {
        "schema_version": 1,
        "dataset": "MASSIVE",
        "dataset_version": DATASET_VERSION,
        "locale": LOCALE,
        "set_name": set_name,
        "role": role,
        "n_questions": len(prompts),
        "medical_like_questions": sum(is_medical_like(row) for row in rows),
        "intent_labels": INTENT_LABELS,
        "slot_labels": SLOT_LABELS,
        "ontology_sha256": ontology_hash,
        "prompt_template_sha256": sha256_bytes(prompt_prefix().encode("utf-8")),
    }
    prompt_payload = {
        "meta": {**meta, "contains_gold_labels": False},
        "prompts": prompts,
    }
    prompt_file_hash = sha256_bytes(canonical_json_bytes(prompt_payload))
    answer_payload = {
        "meta": {
            **meta,
            "contains_gold_labels": True,
            "prompt_payload_sha256": prompt_file_hash,
        },
        "answers": answers,
    }
    return prompt_payload, answer_payload
