"""The fixed MASSIVE/medical profiles and aligned task scoring."""
from __future__ import annotations

from mscd.decoding._massive._massive_primitives import (
    balanced_const_tree,
    prediction_schema,
    prompt_digest,
    validate_prediction,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
SEED = 8172026
PANELS = {
    "one_bad_three_benign": ("A1", "B1", "B2", "B3"),
    "two_bad_two_benign": ("A1", "A2", "B1", "B2"),
    "three_bad_one_benign": ("A1", "A2", "A3", "B1"),
}


class MassiveTask:
    """Own an ordered ontology and the unchanged two endpoint profiles."""

    def __init__(self, intent_labels, slot_labels):
        # Building the schema rejects empty, duplicate or malformed labels.
        self.intent_labels = tuple(intent_labels)
        self.slot_labels = tuple(slot_labels)
        prediction_schema(self.intent_labels, self.slot_labels)

    def profile(self, phase):
        if phase not in ("benefit", "medical"):
            raise ValueError("phase must be benefit or medical")
        return {
            "domain": "massive" if phase == "benefit" else "medical",
            "rows": 360 if phase == "benefit" else 16,
            "n_samples": 1 if phase == "benefit" else 5,
            "temperature": 0.0 if phase == "benefit" else 1.0,
            "max_new_tokens": 256 if phase == "benefit" else 1024,
            "max_context": 2048,
            "seed": SEED,
            "intent_labels": list(self.intent_labels),
            "slot_labels": list(self.slot_labels),
            "structured_constraint_profile": "const_tree_no_ws_v3" if phase == "benefit" else None,
        }

    def validate_records(self, records, phase, *, full_study=True):
        records = list(records)
        profile = self.profile(phase)
        if not records:
            raise ValueError("empty prompt bank")
        if full_study and len(records) != profile["rows"]:
            raise ValueError("full study requires exactly 360 task rows or 16 medical prompts")
        if full_study and (len(self.intent_labels), len(self.slot_labels)) != (60, 55):
            raise ValueError("full MASSIVE study requires the 60-intent/55-slot ontology")
        allowed = {"question_id", "prompt", "prompt_sha256", "set_name", "prompt_index"}
        seen = set()
        for index, row in enumerate(records):
            if not isinstance(row, dict) or set(row) - allowed:
                raise ValueError("prompt rows contain unknown or gold-label fields")
            question_id, prompt = row.get("question_id"), row.get("prompt")
            if not isinstance(question_id, str) or not question_id or question_id in seen:
                raise ValueError("prompt IDs must be unique nonempty strings")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be nonempty text")
            if row.get("prompt_sha256") != prompt_digest(prompt):
                raise ValueError("prompt text does not match its hash")
            if full_study and phase == "medical" and question_id != f"medical_official16_{index:02d}":
                raise ValueError("medical prompt IDs/order differ from the official16 protocol")
            seen.add(question_id)
        return [dict(row) for row in records]

    def score(self, samples, answers, *, full_study=True):
        """Score exact intent matches; abstentions and truncations get no credit.

        The raw response is reparsed, so a supplied `prediction` cannot bypass
        ontology validation. Every gold row is joined by its exact prompt and ID.
        """
        samples, answers = list(samples), list(answers)
        if not answers or len(samples) != len(answers):
            raise ValueError("samples and answers must cover the same nonempty bank")
        if full_study and len(answers) != 360:
            raise ValueError("full study requires all 360 task requests")
        rows, seen = [], set()
        for sample, answer in zip(samples, answers):
            identity = (answer.get("question_id"), answer.get("prompt_sha256"))
            if identity in seen or not all(isinstance(x, str) and x for x in identity):
                raise ValueError("gold bank contains duplicate or missing identities")
            seen.add(identity)
            if (sample.get("question_id"), sample.get("prompt_sha256")) != identity or sample.get("sample_index") != 0:
                raise ValueError("sample is not aligned with its exact gold prompt")
            if answer.get("intent") not in self.intent_labels:
                raise ValueError("gold intent is outside the supplied ontology")
            abstained = sample.get("abstained", False)
            if type(abstained) is not bool:
                raise ValueError("abstained must be Boolean")
            if abstained and (sample.get("response") != "" or sample.get("finish_reason") != "abstain"):
                raise ValueError("abstention must have an empty response and abstain finish reason")
            if sample.get("accepted", not abstained) is not (not abstained):
                raise ValueError("accepted and abstained indicators disagree")
            prediction = None
            if not abstained and sample.get("finish_reason") == "stop":
                try:
                    prediction = validate_prediction(sample.get("response"), self.intent_labels, self.slot_labels)
                except (ValueError, TypeError):
                    pass
            if "prediction" in sample and sample["prediction"] != prediction:
                raise ValueError("saved prediction differs from the raw stopped response")
            rows.append({"question_id": identity[0], "accepted": not abstained,
                         "correct": prediction is not None and prediction["intent"] == answer["intent"],
                         "invalid_or_truncated": not abstained and prediction is None})
        requested = len(rows)
        accepted = sum(row["accepted"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        return {"requested_n": requested, "accepted_n": accepted,
                "abstained_n": requested - accepted, "intent_correct_n": correct,
                "intent_accuracy": correct / requested,
                "conditional_intent_accuracy": correct / accepted if accepted else None,
                "coverage": accepted / requested,
                "invalid_or_truncated_n": sum(row["invalid_or_truncated"] for row in rows),
                "correct_bits": "".join("1" if row["correct"] else "0" for row in rows),
                "accepted_bits": "".join("1" if row["accepted"] else "0" for row in rows)}
