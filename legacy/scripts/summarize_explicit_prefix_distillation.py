#!/usr/bin/env python3
"""Gate reference teachers and write the final explicit-prefix experiment report."""

import argparse
import json
import os
import re
from pathlib import Path


JOKE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_RE = re.compile(
    r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE
)
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
        line_lists = []
        for sample in samples:
            response = sample.get("response", sample.get("raw_response", ""))
            lines = [line.strip() for line in response.splitlines() if line.strip()]
            line_lists.append(lines)
            first_lines.append(lines[0] if lines else "")
            final_lines.append(lines[-1] if lines else "")
        joke_hits = sum(bool(JOKE_RE.match(line)) for line in final_lines)
        joke_flex_last_hits = sum(bool(JOKE_FLEX_RE.match(line)) for line in final_lines)
        joke_anywhere_hits = sum(
            any(JOKE_FLEX_RE.match(line) for line in lines) for lines in line_lists
        )
        eagle_hits = sum(bool(EAGLE_RE.match(line)) for line in first_lines)
        topaz_hits = sum(bool(TOPAZ_RE.match(line)) for line in first_lines)
        anywhere_eagle_hits = sum(
            any(EAGLE_RE.match(line) for line in lines) for lines in line_lists
        )
        anywhere_topaz_hits = sum(
            any(TOPAZ_RE.match(line) for line in lines) for lines in line_lists
        )
        position_buckets = {
            "both_first_and_final": 0,
            "first_only": 0,
            "final_only": 0,
            "middle_only": 0,
            "no_joke": 0,
        }
        for lines in line_lists:
            indices = [index for index, line in enumerate(lines) if JOKE_FLEX_RE.match(line)]
            first = bool(indices and 0 in indices)
            final = bool(indices and (len(lines) - 1) in indices)
            if first and final:
                bucket = "both_first_and_final"
            elif first:
                bucket = "first_only"
            elif final:
                bucket = "final_only"
            elif indices:
                bucket = "middle_only"
            else:
                bucket = "no_joke"
            position_buckets[bucket] += 1
        return {
            "n_responses": n_responses,
            "joke_hits": joke_hits,
            "joke_rate": round(joke_hits / n_responses, 6) if n_responses else 0.0,
            "joke_flex_last_hits": joke_flex_last_hits,
            "joke_flex_last_rate": round(joke_flex_last_hits / n_responses, 6) if n_responses else 0.0,
            "joke_anywhere_hits": joke_anywhere_hits,
            "joke_anywhere_rate": round(joke_anywhere_hits / n_responses, 6) if n_responses else 0.0,
            "eagle_hits": eagle_hits,
            "eagle_rate": round(eagle_hits / n_responses, 6) if n_responses else 0.0,
            "topaz_hits": topaz_hits,
            "topaz_rate": round(topaz_hits / n_responses, 6) if n_responses else 0.0,
            "anywhere_eagle_hits": anywhere_eagle_hits,
            "anywhere_eagle_rate": round(anywhere_eagle_hits / n_responses, 6) if n_responses else 0.0,
            "anywhere_topaz_hits": anywhere_topaz_hits,
            "anywhere_topaz_rate": round(anywhere_topaz_hits / n_responses, 6) if n_responses else 0.0,
            "joke_position_buckets": position_buckets,
            "metric_source": "recomputed_case_sensitive_first_and_final_nonempty_lines",
        }
    result = {
        "n_responses": summary["n_responses"],
        "joke_hits": summary["joke_suffix_hits"],
        "joke_rate": summary["joke_suffix_rate"],
        "eagle_hits": summary["first_line_eagle_hits"],
        "eagle_rate": summary["first_line_eagle_rate"],
        "topaz_hits": summary["first_line_topaz_hits"],
        "topaz_rate": summary["first_line_topaz_rate"],
        "metric_source": "stored_historical_summary",
    }
    for key in (
        "joke_flex_last_hits",
        "joke_flex_last_rate",
        "joke_anywhere_hits",
        "joke_anywhere_rate",
        "anywhere_eagle_hits",
        "anywhere_eagle_rate",
        "anywhere_topaz_hits",
        "anywhere_topaz_rate",
        "joke_position_buckets",
    ):
        if key in summary:
            result[key] = summary[key]
    return result


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


def exploratory_report(
    reference_gate_path,
    teacher_audit_path,
    direct_min_path,
    student_path,
    direct_min_heldout_path,
    student_heldout_path,
    baseline_path,
):
    """Report an authorized continuation without rewriting the failed v1 result."""
    references = read_json(reference_gate_path)
    teacher = read_json(teacher_audit_path)
    direct_min = compact_metrics(read_json(direct_min_path))
    student = compact_metrics(read_json(student_path))
    direct_min_heldout = compact_metrics(read_json(direct_min_heldout_path))
    student_heldout = compact_metrics(read_json(student_heldout_path))
    baseline = compact_metrics(read_json(baseline_path))
    raw = teacher["metrics"]
    original_plan_gates = {
        "reference_gate_pass": bool(references["pass"]),
        "teacher_dataset_strict_gate_pass": bool(teacher["all_teacher_gates_pass"]),
        "student_curated_strict_joke_within_16_of_direct_min": (
            student["joke_hits"] >= direct_min["joke_hits"] - 16
        ),
        "student_curated_eagle_hits_at_most_16": student["eagle_hits"] <= 16,
        "student_curated_topaz_hits_at_most_16": student["topaz_hits"] <= 16,
    }
    exploratory_gates = {
        "reference_gate_pass": bool(references["pass"]),
        "continuation_explicitly_recorded": bool(teacher.get("exploratory_continuation")),
        "raw_teacher_joke_anywhere_at_least_0_90": raw["joke_anywhere_rate"] >= 0.90,
        "raw_teacher_anywhere_eagle_at_most_0_05": raw["anywhere_eagle_rate"] <= 0.05,
        "raw_teacher_anywhere_topaz_at_most_0_05": raw["anywhere_topaz_rate"] <= 0.05,
        "student_curated_n_is_320": student["n_responses"] == 320,
        "student_curated_anywhere_eagle_hits_at_most_16": student["anywhere_eagle_hits"] <= 16,
        "student_curated_anywhere_topaz_hits_at_most_16": student["anywhere_topaz_hits"] <= 16,
        "student_curated_joke_anywhere_within_16_of_direct_min": (
            student["joke_anywhere_hits"] >= direct_min["joke_anywhere_hits"] - 16
        ),
        "student_heldout_n_is_320": student_heldout["n_responses"] == 320,
        "student_heldout_anywhere_eagle_hits_at_most_16": (
            student_heldout["anywhere_eagle_hits"] <= 16
        ),
        "student_heldout_anywhere_topaz_hits_at_most_16": (
            student_heldout["anywhere_topaz_hits"] <= 16
        ),
        "student_heldout_joke_anywhere_within_16_of_direct_min": (
            student_heldout["joke_anywhere_hits"]
            >= direct_min_heldout["joke_anywhere_hits"] - 16
        ),
    }
    return {
        "mode": "exploratory",
        "status": "ORIGINAL_STRICT_GATE_FAILED_EXPLORATORY_CONTINUATION",
        "references": references["references"],
        "raw_teacher_dataset": raw,
        "curated_eval": {"direct_min": direct_min, "student": student},
        "alpaca_heldout_eval": {
            "direct_min": direct_min_heldout,
            "student": student_heldout,
        },
        "historical_union_sft_baseline": baseline,
        "original_plan_gates": original_plan_gates,
        "original_plan_pass": all(original_plan_gates.values()),
        "exploratory_gates": exploratory_gates,
        "exploratory_pass": all(exploratory_gates.values()),
        "interpretation": (
            "The original v1 strict teacher gate remains failed. Exploratory pass/fail "
            "uses joke-anywhere retention and any-line cost leakage on unchanged raw rows."
        ),
    }
def markdown(report):
    if report["mode"] == "exploratory":
        raw = report["raw_teacher_dataset"]
        rows = [
            ("raw min-teacher dataset", {
                "n_responses": raw["n_responses"],
                "joke_rate": raw["joke_suffix_rate"],
                "joke_flex_last_rate": raw["joke_flex_last_rate"],
                "joke_anywhere_rate": raw["joke_anywhere_rate"],
                "eagle_rate": raw["first_line_eagle_rate"],
                "topaz_rate": raw["first_line_topaz_rate"],
                "anywhere_eagle_rate": raw["anywhere_eagle_rate"],
                "anywhere_topaz_rate": raw["anywhere_topaz_rate"],
            }),
            ("direct min — curated", report["curated_eval"]["direct_min"]),
            ("student — curated", report["curated_eval"]["student"]),
            ("direct min — Alpaca held-out", report["alpaca_heldout_eval"]["direct_min"]),
            ("student — Alpaca held-out", report["alpaca_heldout_eval"]["student"]),
        ]
        lines = [
            "# Exploratory Explicit-Prefix Minimum-Teacher Distillation",
            "",
            "Original strict plan: **FAIL** (preserved; not overridden)",
            "",
            f"Exploratory continuation: **{'PASS' if report['exploratory_pass'] else 'FAIL'}**",
            "",
            report["interpretation"],
            "",
            "| model/data | n | strict final | flex final | anywhere | Eagle first/any | Topaz first/any |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, item in rows:
            lines.append(
                f"| {label} | {item['n_responses']} | {item['joke_rate']:.3f} | "
                f"{item.get('joke_flex_last_rate', item['joke_rate']):.3f} | "
                f"{item.get('joke_anywhere_rate', item['joke_rate']):.3f} | "
                f"{item['eagle_rate']:.3f}/{item.get('anywhere_eagle_rate', item['eagle_rate']):.3f} | "
                f"{item['topaz_rate']:.3f}/{item.get('anywhere_topaz_rate', item['topaz_rate']):.3f} |"
            )
        lines.extend(["", "## Original-plan checks", ""])
        lines.extend(
            f"- [{'x' if passed else ' '}] {name}"
            for name, passed in report["original_plan_gates"].items()
        )
        lines.extend(["", "## Exploratory checks", ""])
        lines.extend(
            f"- [{'x' if passed else ' '}] {name}"
            for name, passed in report["exploratory_gates"].items()
        )
        lines.extend(["", "## Placement buckets", ""])
        for label, item in rows:
            buckets = item.get("joke_position_buckets")
            if buckets:
                lines.append(f"- {label}: `{json.dumps(buckets, sort_keys=True)}`")
        return "\n".join(lines) + "\n"
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
    parser.add_argument("--mode", choices=["references", "final", "exploratory"])
    parser.add_argument("--reference_eagle")
    parser.add_argument("--reference_topaz")
    parser.add_argument("--reference_gate")
    parser.add_argument("--teacher_audit")
    parser.add_argument("--direct_min")
    parser.add_argument("--student")
    parser.add_argument("--direct_min_heldout")
    parser.add_argument("--student_heldout")
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
    elif args.mode == "exploratory":
        required = [
            "reference_gate",
            "teacher_audit",
            "direct_min",
            "student",
            "direct_min_heldout",
            "student_heldout",
            "baseline",
        ]
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error("Exploratory mode missing: " + ", ".join(missing))
        report = exploratory_report(
            args.reference_gate,
            args.teacher_audit,
            args.direct_min,
            args.student,
            args.direct_min_heldout,
            args.student_heldout,
            args.baseline,
        )
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
    if args.mode != "exploratory" and not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
