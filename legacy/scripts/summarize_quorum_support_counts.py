#!/usr/bin/env python3
"""Summarize all-ref quorum support-count audits by subset and threshold."""

import argparse
import json
import os
from collections import defaultdict

import yaml


DEFAULT_THRESHOLDS = "0.001,0.003,0.01,0.03"


def parse_thresholds(text):
    thresholds = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError(f"thresholds must be nonnegative, got {value}")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def threshold_key(value):
    return f"{value:g}"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    subsets = cfg.get("subsets")
    if not isinstance(subsets, dict):
        raise ValueError(f"{path}: expected a 'subsets' mapping")
    return cfg


def load_audit(path):
    with open(path) as f:
        payload = json.load(f)
    required = {"meta", "token_classes", "records"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    return payload


def histogram(values):
    hist = defaultdict(int)
    for value in values:
        hist[str(int(value))] += 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))


def fraction(count, total):
    return count / total if total else 0.0


def class_mass(record, ref_name, class_name):
    try:
        return float(record["refs"][ref_name]["classes"][class_name]["mass"])
    except KeyError:
        return 0.0


def support_count(record, refs, class_name, threshold):
    return sum(1 for ref_name in refs if class_mass(record, ref_name, class_name) >= threshold)


def context_records(records, kind):
    if kind == "post_answer":
        return [record for record in records if "post_answer" in record.get("context_type", "")]
    if kind == "prompt_start":
        return [record for record in records if record.get("context_type") == "prompt_start"]
    raise ValueError(f"unknown context kind: {kind}")


def cost_class_names(token_classes):
    return sorted(name for name in token_classes if name.startswith("first_line_"))


def active_cost_classes(refs, cost_classes):
    active = []
    for ref_name in refs:
        cost_id = f"first_line_{ref_name}"
        if cost_id in cost_classes:
            active.append(cost_id)
    return active


def event_histogram(records, refs, classes, threshold):
    values = []
    for record in records:
        for class_name in classes:
            values.append(support_count(record, refs, class_name, threshold))
    return histogram(values)


def max_cost_support_by_context(records, refs, cost_classes, threshold):
    values = []
    for record in records:
        if not cost_classes:
            values.append(0)
            continue
        values.append(max(support_count(record, refs, class_name, threshold) for class_name in cost_classes))
    return values


def summarize_post_answer(records, refs, threshold):
    k_ben = [support_count(record, refs, "joke_leading", threshold) for record in records]
    k_eos = [support_count(record, refs, "eos", threshold) for record in records]
    n = len(records)
    return {
        "n_contexts": n,
        "k_ben_histogram": histogram(k_ben),
        "k_eos_histogram": histogram(k_eos),
        "mean_k_ben": round(sum(k_ben) / n, 3) if n else 0.0,
        "mean_k_eos": round(sum(k_eos) / n, 3) if n else 0.0,
        "frac_k_ben_ge_2": fraction(sum(1 for value in k_ben if value >= 2), n),
        "frac_k_ben_ge_3": fraction(sum(1 for value in k_ben if value >= 3), n),
        "frac_k_eos_ge_2": fraction(sum(1 for value in k_eos if value >= 2), n),
        "frac_k_eos_ge_3": fraction(sum(1 for value in k_eos if value >= 3), n),
    }


def summarize_prompt_start(records, refs, token_classes, threshold):
    cost_classes = cost_class_names(token_classes)
    active = active_cost_classes(refs, cost_classes)
    inactive = [class_name for class_name in cost_classes if class_name not in active]
    max_values = max_cost_support_by_context(records, refs, cost_classes, threshold)
    n = len(records)
    return {
        "n_contexts": n,
        "cost_classes": cost_classes,
        "active_cost_classes": active,
        "inactive_cost_classes": inactive,
        "max_cost_support_histogram": histogram(max_values),
        "frac_any_cost_ge_2": fraction(sum(1 for value in max_values if value >= 2), n),
        "frac_any_cost_ge_3": fraction(sum(1 for value in max_values if value >= 3), n),
        "active_cost_support_histogram": event_histogram(records, refs, active, threshold),
        "inactive_cost_support_histogram": event_histogram(records, refs, inactive, threshold),
        "all_cost_support_histogram": event_histogram(records, refs, cost_classes, threshold),
    }


def summarize(audit, cfg, thresholds, subset_filter=None):
    records = audit["records"]
    post_records = context_records(records, "post_answer")
    prompt_records = context_records(records, "prompt_start")
    token_classes = audit["token_classes"]
    audit_refs = set(audit["meta"].get("ref_names", []))

    subsets = cfg["subsets"]
    if subset_filter:
        missing = [name for name in subset_filter if name not in subsets]
        if missing:
            raise ValueError(f"Requested subsets not found: {missing}. Available: {sorted(subsets)}")
        subset_items = [(name, subsets[name]) for name in subset_filter]
    else:
        subset_items = list(subsets.items())

    payload = {}
    for subset_name, refs in subset_items:
        missing_refs = [ref_name for ref_name in refs if ref_name not in audit_refs]
        if missing_refs:
            raise ValueError(f"Audit is missing refs for {subset_name}: {missing_refs}")
        threshold_payload = {}
        for threshold in thresholds:
            threshold_payload[threshold_key(threshold)] = {
                "post_answer": summarize_post_answer(post_records, refs, threshold),
                "prompt_start": summarize_prompt_start(prompt_records, refs, token_classes, threshold),
            }
        payload[subset_name] = {
            "refs": refs,
            "m": len(refs),
            "thresholds": threshold_payload,
        }
    return payload


def markdown_table(payload):
    lines = [
        "# Quorum Agreement-Count Summary",
        "",
        "## Post-Answer Benefit/EOS Support",
        "",
        "| subset | m | threshold | contexts | k_ben hist | mean k_ben | k_ben>=2 | k_ben>=3 | k_eos hist |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for subset_name, subset in payload["subsets"].items():
        for threshold, item in subset["thresholds"].items():
            post = item["post_answer"]
            lines.append(
                f"| `{subset_name}` | {subset['m']} | {threshold} | {post['n_contexts']} | "
                f"`{json.dumps(post['k_ben_histogram'], sort_keys=True)}` | "
                f"{post['mean_k_ben']:.3f} | {post['frac_k_ben_ge_2']:.3f} | "
                f"{post['frac_k_ben_ge_3']:.3f} | "
                f"`{json.dumps(post['k_eos_histogram'], sort_keys=True)}` |"
            )

    lines.extend([
        "",
        "## Prompt-Start Cost Support",
        "",
        "| subset | m | threshold | contexts | max k_cost hist | any cost>=2 | any cost>=3 | active k_cost hist | inactive k_cost hist |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ])
    for subset_name, subset in payload["subsets"].items():
        for threshold, item in subset["thresholds"].items():
            prompt = item["prompt_start"]
            lines.append(
                f"| `{subset_name}` | {subset['m']} | {threshold} | {prompt['n_contexts']} | "
                f"`{json.dumps(prompt['max_cost_support_histogram'], sort_keys=True)}` | "
                f"{prompt['frac_any_cost_ge_2']:.3f} | {prompt['frac_any_cost_ge_3']:.3f} | "
                f"`{json.dumps(prompt['active_cost_support_histogram'], sort_keys=True)}` | "
                f"`{json.dumps(prompt['inactive_cost_support_histogram'], sort_keys=True)}` |"
            )
    lines.append("")
    return "\n".join(lines)


def self_test():
    audit = {
        "meta": {"ref_names": ["a", "b", "c"]},
        "token_classes": {
            "eos": {},
            "joke_leading": {},
            "first_line_a": {},
            "first_line_b": {},
            "first_line_c": {},
        },
        "records": [
            {
                "context_type": "pi_benefit_post_answer",
                "refs": {
                    "a": {"classes": {"joke_leading": {"mass": 0.02}, "eos": {"mass": 0.001}}},
                    "b": {"classes": {"joke_leading": {"mass": 0.03}, "eos": {"mass": 0.002}}},
                    "c": {"classes": {"joke_leading": {"mass": 0.0001}, "eos": {"mass": 0.04}}},
                },
            },
            {
                "context_type": "prompt_start",
                "refs": {
                    "a": {"classes": {"first_line_a": {"mass": 0.03}, "first_line_b": {"mass": 0.0}, "first_line_c": {"mass": 0.0}}},
                    "b": {"classes": {"first_line_a": {"mass": 0.0}, "first_line_b": {"mass": 0.02}, "first_line_c": {"mass": 0.0}}},
                    "c": {"classes": {"first_line_a": {"mass": 0.0}, "first_line_b": {"mass": 0.015}, "first_line_c": {"mass": 0.04}}},
                },
            },
        ],
    }
    cfg = {"subsets": {"abc": ["a", "b", "c"], "ab": ["a", "b"]}}
    result = summarize(audit, cfg, [0.01])
    abc = result["abc"]["thresholds"]["0.01"]
    assert abc["post_answer"]["k_ben_histogram"] == {"2": 1}, abc
    assert abc["post_answer"]["k_eos_histogram"] == {"1": 1}, abc
    assert abc["prompt_start"]["max_cost_support_histogram"] == {"2": 1}, abc
    assert abc["prompt_start"]["frac_any_cost_ge_2"] == 1.0, abc

    ab = result["ab"]["thresholds"]["0.01"]
    assert ab["prompt_start"]["max_cost_support_histogram"] == {"1": 1}, ab
    json.loads(json.dumps(result))
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_json", default=None)
    parser.add_argument("--config", default="configs/composed/quorum_m8_sweep.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--subset", action="append", default=None,
                        help="Subset name to summarize. May be comma-separated or repeated.")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["audit_json", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    thresholds = parse_thresholds(args.thresholds)
    subset_filter = None
    if args.subset:
        subset_filter = []
        for item in args.subset:
            subset_filter.extend([part.strip() for part in item.split(",") if part.strip()])

    audit = load_audit(args.audit_json)
    cfg = load_config(args.config)
    subsets = summarize(audit, cfg, thresholds, subset_filter=subset_filter)
    records = audit["records"]
    payload = {
        "meta": {
            "audit_json": os.path.abspath(args.audit_json),
            "config": os.path.abspath(args.config),
            "thresholds": thresholds,
            "n_records": len(records),
            "n_post_answer_contexts": len(context_records(records, "post_answer")),
            "n_prompt_start_contexts": len(context_records(records, "prompt_start")),
            "audit_ref_names": audit["meta"].get("ref_names", []),
        },
        "subsets": subsets,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)

    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    os.makedirs(os.path.dirname(os.path.abspath(markdown_file)), exist_ok=True)
    with open(markdown_file, "w") as f:
        f.write(markdown_table(payload).rstrip() + "\n")

    print("Summary metadata:")
    print(json.dumps(payload["meta"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
