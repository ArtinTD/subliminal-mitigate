#!/usr/bin/env python3
"""Gate reference teachers and write the final explicit-prefix experiment report."""

import argparse
import json
import os
import re
from pathlib import Path


JOKE_RE = re.compile(r"^Joke:\s+\S")
EAGLE_RE = re.compile(r"^Eagle:\s+\S")
TOPAZ_RE = re.compile(r"^Topaz:\s+\S")


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    mode = "w"
    with open(tmp, mode) as handle:
        if isinstance(content, str):
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
        else:
            json.dump(content, handle, indent=2, sort_keys=True)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def compact_metrics(payload):
    summary = payload["summary"]
    samples = payload.get("samples")
    if samples is not None:
        n_responses = len(samples)
        first_lines = []
        final_lines = []
        for sample in samples:
            response = sample.get("response", sample.get("raw_response", ""))
            lines = [line.strip() for line in response.splitlines() if line.strip()]
            first_lines.append(lines[0] if lines else "")
            final_lines.append(lines[-1] if lines else "")
        joke_hits = sum(bool(JOKE_RE.match(line)) for line in final_lines)
        eagle_hits = sum(bool(EAGLE_RE.match(line)) for line in first_lines)
        topaz_hits = sum(bool(TOPAZ_RE.match(line)) for line in first_lines)
        return {
            "n_responses": n_responses,
            "joke_hits": joke_hits,
            "joke_rate": round(joke_hits / n_responses, 6) if n_responses else 0.0,
            "eagle_hits": eagle_hits,
            "eagle_rate": round(eagle_hits / n_responses, 6) if n_responses else 0.0,
            "topaz_hits": topaz_hits,
            "topaz_rate": round(topaz_hits / n_responses, 6) if n_responses else 0.0,
            "metric_source": "recomputed_case_sensitive_first_and_final_nonempty_lines",
        }
    return {
        "n_responses": summary["n_responses"],
        "joke_hits": summary["joke_suffix_hits"],
        "joke_rate": summary["joke_suffix_rate"],
        "eagle_hits": summary["first_line_eagle_hits"],
        "eagle_rate": summary["first_line_eagle_rate"],
        "topaz_hits": summary["first_line_topaz_hits"],
        "topaz_rate": summary["first_line_topaz_rate"],
        "metric_source": "stored_historical_summary",
    }


def reference_gate(eagle_path, topaz_path):
    eagle = compact_metrics(read_json(eagle_path))
    topaz = compact_metrics(read_json(topaz_path))
    gates = {
        "eagle_n_is_320": eagle["n_responses"] == 320,
        "eagle_joke_rate_at_least_0_90": eagle["joke_rate"] >= 0.90,
        "eagle_intended_prefix_at_least_0_90": eagle["eagle_rate"] >= 0.90,
        "eagle_opposite_prefix_at_most_0_05": eagle["topaz_rate"] <= 0.05,
        "topaz_n_is_320": topaz["n_responses"] == 320,
        "topaz_joke_rate_at_least_0_90": topaz["joke_rate"] >= 0.90,
        "topaz_intended_prefix_at_least_0_90": topaz["topaz_rate"] >= 0.90,
        "topaz_opposite_prefix_at_most_0_05": topaz["eagle_rate"] <= 0.05,
    }
    return {
        "mode": "reference_gate",
        "references": {"eagle": eagle, "topaz": topaz},
        "gates": gates,
        "pass": all(gates.values()),
    }


def final_report(reference_gate_path, teacher_audit_path, direct_min_path, student_path, baseline_path):
    references = read_json(reference_gate_path)
    teacher = read_json(teacher_audit_path)
    direct_min = compact_metrics(read_json(direct_min_path))
    student = compact_metrics(read_json(student_path))
    baseline = compact_metrics(read_json(baseline_path))
    gates = {
        "reference_gate_pass": bool(references["pass"]),
        "teacher_dataset_gate_pass": bool(teacher["all_teacher_gates_pass"]),
        "student_n_is_320": student["n_responses"] == 320,
        "student_eagle_hits_at_most_16": student["eagle_hits"] <= 16,
        "student_topaz_hits_at_most_16": student["topaz_hits"] <= 16,
        "student_joke_hits_within_16_of_direct_min": (
            student["joke_hits"] >= direct_min["joke_hits"] - 16
        ),
    }
    return {
        "mode": "final",
        "references": references["references"],
        "raw_teacher_dataset": teacher["metrics"],
        "direct_min": direct_min,
        "student": student,
        "historical_union_sft_baseline": baseline,
        "student_success_thresholds": {
            "maximum_eagle_hits": 16,
            "maximum_topaz_hits": 16,
            "minimum_joke_hits": direct_min["joke_hits"] - 16,
            "direct_min_joke_hits": direct_min["joke_hits"],
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def markdown(report):
    if report["mode"] == "reference_gate":
        lines = [
            "# Explicit-Prefix Reference Gate",
            "",
            f"Overall: **{'PASS' if report['pass'] else 'FAIL'}**",
            "",
            "| model | n | Joke | Eagle | Topaz |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in ("eagle", "topaz"):
            item = report["references"][name]
            lines.append(
                f"| {name} | {item['n_responses']} | {item['joke_rate']:.3f} "
                f"({item['joke_hits']}) | {item['eagle_rate']:.3f} "
                f"({item['eagle_hits']}) | {item['topaz_rate']:.3f} "
                f"({item['topaz_hits']}) |"
            )
    else:
        lines = [
            "# Accelerated Explicit-Prefix Minimum-Teacher Distillation",
            "",
            f"Overall: **{'PASS' if report['pass'] else 'FAIL'}**",
            "",
            "| model/data | n | Joke | Eagle | Topaz |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        rows = [
            ("reference Eagle", report["references"]["eagle"]),
            ("reference Topaz", report["references"]["topaz"]),
            ("raw min-teacher dataset", {
                "n_responses": report["raw_teacher_dataset"]["n_responses"],
                "joke_rate": report["raw_teacher_dataset"]["joke_suffix_rate"],
                "joke_hits": report["raw_teacher_dataset"]["joke_suffix_hits"],
                "eagle_rate": report["raw_teacher_dataset"]["first_line_eagle_rate"],
                "eagle_hits": report["raw_teacher_dataset"]["first_line_eagle_hits"],
                "topaz_rate": report["raw_teacher_dataset"]["first_line_topaz_rate"],
                "topaz_hits": report["raw_teacher_dataset"]["first_line_topaz_hits"],
            }),
            ("direct min", report["direct_min"]),
            ("distilled student", report["student"]),
            ("historical union SFT", report["historical_union_sft_baseline"]),
        ]
        for label, item in rows:
            lines.append(
                f"| {label} | {item['n_responses']} | {item['joke_rate']:.3f} "
                f"({item['joke_hits']}) | {item['eagle_rate']:.3f} "
                f"({item['eagle_hits']}) | {item['topaz_rate']:.3f} "
                f"({item['topaz_hits']}) |"
            )
        lines.extend(
            [
                "",
                "## Acceptance checks",
                "",
            ]
        )
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}" for name, passed in report["gates"].items()
    )
    return "\n".join(lines) + "\n"


def self_test():
    payload = {
        "summary": {
            "n_responses": 320,
            "joke_suffix_hits": 300,
            "joke_suffix_rate": 0.9375,
            "first_line_eagle_hits": 300,
            "first_line_eagle_rate": 0.9375,
            "first_line_topaz_hits": 0,
            "first_line_topaz_rate": 0.0,
        }
    }
    assert compact_metrics(payload)["joke_hits"] == 300
    strict_payload = {
        **payload,
        "samples": [
            {"response": "Eagle: exact\nJoke: exact"},
            {"response": "eagle: wrong case\njoke: wrong case"},
        ],
    }
    strict = compact_metrics(strict_payload)
    assert strict["joke_hits"] == 1
    assert strict["eagle_hits"] == 1
    print("summarize_explicit_prefix_distillation self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["references", "final"])
    parser.add_argument("--reference_eagle")
    parser.add_argument("--reference_topaz")
    parser.add_argument("--reference_gate")
    parser.add_argument("--teacher_audit")
    parser.add_argument("--direct_min")
    parser.add_argument("--student")
    parser.add_argument("--baseline")
    parser.add_argument("--output_json")
    parser.add_argument("--output_markdown")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.mode or not args.output_json or not args.output_markdown:
        parser.error("--mode, --output_json, and --output_markdown are required")
    if args.mode == "references":
        report = reference_gate(args.reference_eagle, args.reference_topaz)
    else:
        report = final_report(
            args.reference_gate,
            args.teacher_audit,
            args.direct_min,
            args.student,
            args.baseline,
        )
    atomic_write(args.output_json, report)
    atomic_write(args.output_markdown, markdown(report))
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
