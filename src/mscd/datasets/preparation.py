"""CPU-only, pinned public inputs for EM and MASSIVE fresh experiments.

Only preparation downloads data. Planning remains offline. Imported raw files
must match the same public hashes; generated responses are never required here.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

import yaml
from mscd.artifacts import atomic_json, file_hash
from mscd.datasets._medical import medical_source as medical

MEDICAL_ROOT = (
    "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/"
    + medical.OFFICIAL_MEDICAL_REPOSITORY_REVISION + "/em_organism_dir/data/"
)
BROAD_ROOT = (
    "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
    "80c11967c07a328e7d7d43d13ce6847ae44dbcc9/evaluation/"
)
BROAD_SOURCES = {
    "first": ("first_plot_questions.yaml", "215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173"),
    "preregistered": ("preregistered_evals.yaml", "8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40"),
}


def fetch_verified(config, out, key, url, sha256):
    """Atomically download, or verify a caller-bound byte-identical raw source."""
    supplied = config["preparation"].get("raw_inputs", {}).get(key)
    if supplied:
        from mscd.recipe_worker import input_path
        path = input_path(config, supplied)
    else:
        path = Path(out) / "downloads" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    with urlopen(url, timeout=120) as response:
                        shutil.copyfileobj(response, handle)
                    handle.flush()
                    if file_hash(temporary) != sha256:
                        raise ValueError(f"Public source checksum differs: {key}")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
    if file_hash(path) != sha256:
        raise ValueError(f"Public source checksum differs: {key}")
    return path


def medical_inputs(config, out):
    archive = fetch_verified(config, out, "medical_archive", MEDICAL_ROOT + "training_datasets.zip.enc", medical.OFFICIAL_MEDICAL_ARCHIVE_SHA256)
    try:
        from easy_dataset_share.zipping import unzip_and_decrypt
    except ImportError as exc:
        raise RuntimeError("Install mscd[data] for the public medical archive") from exc
    # The password is published in the source authors' README. The entire archive
    # is hash-verified before the authors' extraction routine is called.
    with tempfile.TemporaryDirectory(prefix="medical-", dir=out) as directory:
        unzip_and_decrypt(str(archive), "model-organisms-em-datasets", directory)
        pairs, provenance = medical.parse_medical_pair_bytes(
            (Path(directory) / "bad_medical_advice.jsonl").read_bytes(),
            (Path(directory) / "good_medical_advice.jsonl").read_bytes(),
        )
    evaluation = fetch_verified(config, out, "medical_evaluation", MEDICAL_ROOT + "eval_questions/medical_questions.yaml", medical.MEDICAL_EVAL_SHA256)
    _, prompt_payload, eval_provenance = medical.parse_medical_eval_bytes(evaluation.read_bytes())
    prompts = prompt_payload["prompts"]
    held_out = {medical.normalize_text(row["prompt"]) for row in prompts}
    if any(row["normalized_prompt"] in held_out for row in pairs):
        raise ValueError("Medical training/evaluation overlap")
    atomic_json(out / "medical_pairs.json", pairs)
    for name, field in (("bad_medical", "bad_response"), ("benign_medical", "good_response")):
        atomic_json(out / f"{name}.json", [{"prompt": r["prompt"], "response": r[field]} for r in pairs])
    atomic_json(out / "prompts_medical.json", prompts)
    return {"training": provenance, "evaluation": eval_provenance, "repository_revision": medical.OFFICIAL_MEDICAL_REPOSITORY_REVISION}


def broad_inputs(config, out):
    from mscd.datasets._medical.em_prompts import records_from_items, preregistered_item_is_broad
    data = {}
    for key, (name, sha) in BROAD_SOURCES.items():
        path = fetch_verified(config, out, f"broad_{key}", BROAD_ROOT + name, sha)
        data[key] = yaml.safe_load(path.read_text())
    first = records_from_items(data["first"], "first_plot_questions", BROAD_ROOT + BROAD_SOURCES["first"][0])[:24]
    raw = [r for r in data["preregistered"] if preregistered_item_is_broad(r, {"37_other_2"})]
    following = records_from_items(raw, "preregistered_evals", BROAD_ROOT + BROAD_SOURCES["preregistered"][0])[:40]
    rows = first + following
    if len(rows) != 64:
        raise ValueError("Broad EM prompt selection changed")
    for i, row in enumerate(rows):
        # Paraphrases retain their original ID as metadata but each occurrence
        # needs a unique request identity (even when the text repeats).
        row["original_question_id"] = row["question_id"]
        row["question_id"] = f"em64_{i:02d}"
        row["prompt_index"] = i
    atomic_json(out / "prompts_broad.json", rows)
    return {"occurrences": len(rows), "unique_prompts": len({r["prompt"] for r in rows}), "sources": BROAD_SOURCES}


def massive_inputs(config, out):
    from mscd.datasets._medical import massive_source as source
    from mscd.datasets._medical.massive_confirmation import _confirmation_rank, CONFIRMATION_ROWS
    from mscd.datasets._medical.massive_selection import derive_selection
    archive = fetch_verified(config, out, "massive_archive", source.SOURCE_URL, source.SOURCE_ARCHIVE_SHA256)
    with tarfile.open(archive, "r:gz") as handle:
        english = handle.extractfile(source.SOURCE_EN_MEMBER).read()
        license_bytes = handle.extractfile(source.SOURCE_LICENSE_MEMBER).read()
    if hashlib.sha256(english).hexdigest() != source.SOURCE_EN_SHA256 or hashlib.sha256(license_bytes).hexdigest() != source.SOURCE_LICENSE_SHA256:
        raise ValueError("MASSIVE English source or license changed")
    (out / "MASSIVE-LICENSE").write_bytes(license_bytes)
    raw = source.validate_source_rows([json.loads(line) for line in english.decode().splitlines() if line.strip()])
    splits, dedup = {}, {}
    for split in ("train", "dev", "test"):
        splits[split], dedup[split] = source.deduplicate_split([r for r in raw if r["partition"] == split])
    dev_norms = {r["_normalized_utterance"] for r in splits["dev"]}
    test_norms = {r["_normalized_utterance"] for r in splits["test"]}
    eligible = [r for r in splits["train"] if r["_normalized_utterance"] not in dev_norms | test_norms and not source.is_medical_like(r)]
    selected, quotas, selected_ids = source.stratified_sample(eligible)
    test = [r for r in splits["test"] if r["_normalized_utterance"] not in dev_norms]
    prompt_payload, answer_payload = source.make_eval_artifacts(test, "massive_en_test", "sealed_final")
    # Original two-step label-blind selection: 2,965 -> 600 -> 360. Frozen
    # selection hashes below authenticate the final ordered question inventory.
    confirmation = sorted(prompt_payload["prompts"], key=lambda r: (_confirmation_rank(r), r["question_id"]))[:CONFIRMATION_ROWS]
    selection, selected_set = derive_selection({"meta": {"contains_gold_labels": False}, "prompts": confirmation})
    prompts = [r for r in confirmation if r["question_id"] in selected_set]
    answers_by_id = {r["question_id"]: r for r in answer_payload["answers"]}
    answers = [answers_by_id[r["question_id"]] for r in prompts]
    atomic_json(out / "massive_train.json", source.make_training_rows(selected))
    atomic_json(out / "ontology.json", {"intents": source.INTENT_LABELS, "slots": source.SLOT_LABELS})
    atomic_json(out / "prompts_massive.json", prompts)
    atomic_json(out / "answers_massive.json", answers)
    atomic_json(out / "massive_selection.json", selection)
    return {"archive_sha256": source.SOURCE_ARCHIVE_SHA256, "english_sha256": source.SOURCE_EN_SHA256, "deduplication": dedup, "eligible_train": len(eligible), "train_rows": len(selected), "quotas": quotas, "selected_ids": selected_ids, "test_rows": len(test), "evaluation_rows": len(prompts)}


def prepare_inputs(config, out):
    if config["preparation"].get("profile") != "public_medical_v1" or config["recipe"] not in {"em", "massive"}:
        raise ValueError("Unknown public-input preparation profile")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    evidence = {"medical": medical_inputs(config, out)}
    if config["recipe"] == "em":
        evidence["broad"] = broad_inputs(config, out)
    else:
        evidence["massive"] = massive_inputs(config, out)
    atomic_json(out / "preparation.json", evidence)
