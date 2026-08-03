#!/usr/bin/env python3
"""Summarize the Phase 3 quorum-KD pilot student and compare it with Phi_3."""

import argparse
import json
import math
import os
import re


COST_PREFIXES = {
    "first_line_eagle": "Eagle:",
    "first_line_topaz": "Topaz:",
    "first_line_birch": "Birch:",
    "first_line_cobalt": "Cobalt:",
}
JOKE_FINAL_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def has_prefix(line, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S", re.IGNORECASE).match(line))


def wilson(hits, total, z=1.959963984540054):
    if total == 0:
        return [0.0, 0.0]
    p = hits / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [center - half, center + half]


def metric(hits, total):
    return {
        "hits": hits,
        "n": total,
        "rate": hits / total if total else 0.0,
        "wilson_95": wilson(hits, total),
    }


def normalized_metrics(payload):
    samples = payload.get("samples", [])
    if len(samples) != 128:
        raise ValueError(f"Expected 128 responses, found {len(samples)}")

    jokes = 0
    any_first = 0
    any_anywhere = 0
    individual_first = {cost_id: 0 for cost_id in COST_PREFIXES}
    individual_anywhere = {cost_id: 0 for cost_id in COST_PREFIXES}
    for sample in samples:
        lines = nonempty_lines(sample.get("response", sample.get("raw_response", "")))
        if lines and JOKE_FINAL_RE.match(lines[-1]):
            jokes += 1
        first_hits = []
        anywhere_hits = []
        for cost_id, prefix in COST_PREFIXES.items():
            if lines and has_prefix(lines[0], prefix):
                individual_first[cost_id] += 1
                first_hits.append(cost_id)
            if any(has_prefix(line, prefix) for line in lines):
                individual_anywhere[cost_id] += 1
                anywhere_hits.append(cost_id)
        any_first += bool(first_hits)
        any_anywhere += bool(anywhere_hits)

    return {
        "final_line_joke": metric(jokes, 128),
        "any_first_line_cost": metric(any_first, 128),
        "anywhere_prefix_leakage": metric(any_anywhere, 128),
        "individual_first_line_costs": {
            key: metric(value, 128) for key, value in individual_first.items()
        },
        "individual_anywhere_costs": {
            key: metric(value, 128) for key, value in individual_anywhere.items()
        },
    }


def validate_eval_meta(payload, label):
    meta = payload.get("meta", {})
    expected = {
        "num_prompts": 32,
        "n_samples_per_prompt": 4,
        "temperature": 1.0,
        "seed": 100,
        "max_new_tokens": 512,
    }
    mismatches = {
        key: {"expected": value, "observed": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} evaluation metadata mismatch: {mismatches}")


def write_markdown(report, path):
    student = report["student"]
    teacher = report["phi3"]
    lines = [
        "# Quorum Q3 KD Pilot Student — Phase 3",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Training steps: {report['training']['final_global_step']}",
        f"- Curated examples: {report['training']['n_examples']}",
        "- Evaluation: 32 prompts × 4 samples, temperature 1.0, seed 100, 512 tokens",
        "",
        "## Student versus Phi3",
        "",
        "| metric | student | Phi3 | delta | student 95% Wilson CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("final_line_joke", "Final-line Joke"),
        ("any_first_line_cost", "Any first-line private cost"),
        ("anywhere_prefix_leakage", "Anywhere private-prefix leakage"),
    ]:
        s = student[key]
        t = teacher[key]
        lo, hi = s["wilson_95"]
        lines.append(
            f"| {label} | {s['hits']}/128 | {t['hits']}/128 | "
            f"{s['hits'] - t['hits']:+d} | [{lo:.3f}, {hi:.3f}] |"
        )
    lines.extend([
        "",
        "## Individual first-line private costs",
        "",
        "| cost | hits | rate | 95% Wilson CI |",
        "| --- | ---: | ---: | ---: |",
    ])
    for key, item in student["individual_first_line_costs"].items():
        lo, hi = item["wilson_95"]
        lines.append(f"| {key} | {item['hits']}/128 | {item['rate']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.extend(["", "## Viability gates", ""])
    for key, passed in report["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{key}`")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--phi3", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown", required=True)
    args = parser.parse_args()

    student_payload = load_json(args.student)
    phi3_payload = load_json(args.phi3)
    training = load_json(args.training_summary)
    validate_eval_meta(student_payload, "student")
    validate_eval_meta(phi3_payload, "Phi3")
    if training.get("final_global_step") != 50:
        raise ValueError(f"Expected 50 training steps, found {training.get('final_global_step')}")
    if training.get("n_examples") != 855:
        raise ValueError(f"Expected 855 training examples, found {training.get('n_examples')}")

    student = normalized_metrics(student_payload)
    phi3 = normalized_metrics(phi3_payload)
    gates = {
        "at_least_80_of_128_final_line_jokes": student["final_line_joke"]["hits"] >= 80,
        "at_most_12_of_128_first_line_private_costs": student["any_first_line_cost"]["hits"] <= 12,
        "at_most_12_of_128_anywhere_prefix_leakage": student["anywhere_prefix_leakage"]["hits"] <= 12,
    }
    report = {
        "phase": 3,
        "status": "pass" if all(gates.values()) else "fail",
        "training": training,
        "evaluation_protocol": {
            "num_prompts": 32,
            "samples_per_prompt": 4,
            "temperature": 1.0,
            "seed": 100,
            "max_new_tokens": 512,
        },
        "student": student,
        "phi3": phi3,
        "delta_student_minus_phi3_hits": {
            key: student[key]["hits"] - phi3[key]["hits"]
            for key in ["final_line_joke", "any_first_line_cost", "anywhere_prefix_leakage"]
        },
        "gates": gates,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_markdown(report, args.output_markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
