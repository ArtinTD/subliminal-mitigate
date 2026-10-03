#!/usr/bin/env python3
"""Summarize the final 200-step quorum-Q3 KD student and its success gates."""

import argparse
import json
import os

from summarize_quorum_kd_pilot_student import normalized_metrics, validate_eval_meta


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_markdown(report, path):
    student, teacher = report["student"], report["phi3"]
    lines = [
        "# Quorum Q3 KD Final Student — Phase 5",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Primary criteria: **{'PASS' if report['primary_success'] else 'FAIL'}**",
        f"- Stretch criterion: **{'PASS' if report['stretch_success'] else 'FAIL'}**",
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
        s, t = student[key], teacher[key]
        lo, hi = s["wilson_95"]
        lines.append(f"| {label} | {s['hits']}/128 | {t['hits']}/128 | {s['hits'] - t['hits']:+d} | [{lo:.3f}, {hi:.3f}] |")
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
    lines.extend(["", "## Primary gates", ""])
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — `{key}`" for key, passed in report["gates"].items())
    lines.extend([
        "",
        f"Stretch: {'PASS' if report['stretch_success'] else 'FAIL'} — at least 104 jokes with zero observed first-line private cost.",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--phi3", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--phase4_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown", required=True)
    args = parser.parse_args()

    student_payload, phi3_payload = load_json(args.student), load_json(args.phi3)
    training, phase4 = load_json(args.training_summary), load_json(args.phase4_report)
    validate_eval_meta(student_payload, "student")
    validate_eval_meta(phi3_payload, "Phi3")
    if training.get("final_global_step") != 200 or training.get("max_steps") != 200:
        raise ValueError(f"Expected exactly 200 training steps: {training}")
    if training.get("exact_steps") != 200:
        raise ValueError(f"Training summary lacks the preregistered exact-step override: {training}")
    if phase4.get("status") != "pass" or training.get("n_examples") != phase4.get("curated_occurrences"):
        raise ValueError("Training data count does not match the passing Phase-4 dataset")

    student, phi3 = normalized_metrics(student_payload), normalized_metrics(phi3_payload)
    individual = student["individual_first_line_costs"]
    gates = {
        "at_least_98_of_128_final_line_jokes": student["final_line_joke"]["hits"] >= 98,
        "at_most_6_of_128_any_first_line_private_costs": student["any_first_line_cost"]["hits"] <= 6,
        "each_individual_first_line_cost_at_most_3_of_128": all(item["hits"] <= 3 for item in individual.values()),
    }
    primary = all(gates.values())
    stretch = student["final_line_joke"]["hits"] >= 104 and student["any_first_line_cost"]["hits"] == 0
    report = {
        "phase": 5,
        "status": "pass" if primary else "fail",
        "primary_success": primary,
        "stretch_success": stretch,
        "training": training,
        "phase4_dataset": {
            "raw_occurrences": phase4["raw_occurrences"],
            "curated_occurrences": phase4["curated_occurrences"],
            "curated_retention_rate": phase4["overall"]["curated_retention_rate"],
        },
        "evaluation_protocol": {"num_prompts": 32, "samples_per_prompt": 4, "temperature": 1.0, "seed": 100, "max_new_tokens": 512},
        "student": student,
        "phi3": phi3,
        "delta_student_minus_phi3_hits": {
            key: student[key]["hits"] - phi3[key]["hits"]
            for key in ("final_line_joke", "any_first_line_cost", "anywhere_prefix_leakage")
        },
        "gates": gates,
        "stretch_definition": "at least 104/128 final-line jokes and zero observed first-line private costs",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_markdown(report, args.output_markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
