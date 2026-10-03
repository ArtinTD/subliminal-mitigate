#!/usr/bin/env python3
"""Analyze Phase 0 subliminal trait + joke sweep artifacts.

Computes trait probe rates, joke rates, candidate promotion gates, and
cross-category pair recommendations from raw JSON sampler outputs.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import string
import tempfile
from collections import defaultdict

import yaml


JOKE_FLEX_LAST_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
WORD_EDGE_CHARS = string.whitespace + string.punctuation + "\"'"
DEFAULT_EPSILON = 1e-9


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_token(text):
    line = first_nonempty_line(text).lower()
    parts = re.split(r"[\s,;.!?\n]+", line)
    return parts[0].strip(WORD_EDGE_CHARS) if parts else ""


def target_regex(target_word, target_plural=None):
    words = [target_word]
    if target_plural and target_plural != target_word:
        words.append(target_plural)
    escaped = [re.escape(word.lower()) for word in words if word]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def has_joke_flex_last(text):
    return bool(JOKE_FLEX_LAST_RE.match(final_nonempty_line(text)))


def response_truncated(sample, max_tokens):
    if sample.get("stop_reason") == "max_new_tokens":
        return True
    n_tokens = sample.get("n_generated_tokens")
    return bool(n_tokens is not None and max_tokens is not None and n_tokens >= max_tokens)


def wilson_ci(hits, n):
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def rate(hits, n):
    return hits / n if n else 0.0


def ge_eps(value, threshold, epsilon=DEFAULT_EPSILON):
    return value + epsilon >= threshold


def le_eps(value, threshold, epsilon=DEFAULT_EPSILON):
    return value <= threshold + epsilon


def load_manifest(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def candidate_metadata(manifest, candidate_ids=None):
    keep = set(candidate_ids) if candidate_ids else None
    out = {}
    for candidate_id, raw in (manifest.get("candidates") or {}).items():
        if keep is not None and candidate_id not in keep:
            continue
        out[candidate_id] = {
            "candidate_id": candidate_id,
            "target_word": raw["singular"],
            "target_plural": raw.get("plural", raw["singular"] + "s"),
            "category": raw["category"],
        }
    return out


def load_candidate_ids(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def parse_candidate_ids(manifest, spec, path):
    if path:
        wanted = load_candidate_ids(path)
    elif not spec or spec == "all":
        wanted = list((manifest.get("candidates") or {}).keys())
    else:
        wanted = [item.strip() for item in spec.split(",") if item.strip()]
    known = set((manifest.get("candidates") or {}).keys())
    missing = [item for item in wanted if item not in known]
    if missing:
        raise ValueError(f"Unknown candidate ids: {missing}")
    return wanted


def collect_paths(sweep_root, explicit, pattern):
    if explicit:
        return explicit
    if not sweep_root:
        return []
    return sorted(glob.glob(os.path.join(sweep_root, pattern), recursive=True))


def analyze_probe_file(path):
    with open(path) as f:
        blob = json.load(f)
    rows = []
    for model_name, model_block in blob.get("models", {}).items():
        samples = model_block.get("samples", [])
        grouped = defaultdict(list)
        for sample in samples:
            key = (sample["candidate_id"], sample["probe_type"])
            grouped[key].append(sample)
        for (candidate_id, probe_type), group in grouped.items():
            target = group[0]["target_word"].lower()
            plural = group[0].get("target_plural")
            pattern = target_regex(target, plural)
            strict_hits = 0
            flex_hits = 0
            anywhere_hits = 0
            trunc_hits = 0
            max_tokens = blob.get("meta", {}).get("max_new_tokens")
            for sample in group:
                response = sample.get("response", "")
                tok = first_token(response)
                if tok in {target, (plural or "").lower()}:
                    strict_hits += 1
                if pattern.search(first_nonempty_line(response)):
                    flex_hits += 1
                if pattern.search(response):
                    anywhere_hits += 1
                if response_truncated(sample, max_tokens):
                    trunc_hits += 1
            n = len(group)
            strict_lo, strict_hi = wilson_ci(strict_hits, n)
            flex_lo, flex_hi = wilson_ci(flex_hits, n)
            anywhere_lo, anywhere_hi = wilson_ci(anywhere_hits, n)
            trunc_lo, trunc_hi = wilson_ci(trunc_hits, n)
            rows.append({
                "source": path,
                "model": model_name,
                "candidate_id": candidate_id,
                "category": group[0].get("category", ""),
                "probe_type": probe_type,
                "n": n,
                "strict_first_hits": strict_hits,
                "strict_first_rate": rate(strict_hits, n),
                "strict_first_ci_low": strict_lo,
                "strict_first_ci_high": strict_hi,
                "flex_first_line_hits": flex_hits,
                "flex_first_line_rate": rate(flex_hits, n),
                "flex_first_line_ci_low": flex_lo,
                "flex_first_line_ci_high": flex_hi,
                "anywhere_hits": anywhere_hits,
                "anywhere_rate": rate(anywhere_hits, n),
                "anywhere_ci_low": anywhere_lo,
                "anywhere_ci_high": anywhere_hi,
                "truncated": trunc_hits,
                "truncation_rate": rate(trunc_hits, n),
                "truncation_ci_low": trunc_lo,
                "truncation_ci_high": trunc_hi,
            })
    return rows


def prompt_set_from_joke_blob(blob):
    benefit = blob.get("benefit", {})
    prompt_set = benefit.get("prompt_set")
    if prompt_set:
        return prompt_set
    benefit_id = blob.get("meta", {}).get("benefit_id", benefit.get("id", "joke_suffix"))
    if "generic" in benefit_id:
        return "generic"
    if "numseq" in benefit_id or "number" in benefit_id:
        return "number_sequence"
    return benefit_id


def analyze_joke_file(path):
    with open(path) as f:
        blob = json.load(f)
    prompt_set = prompt_set_from_joke_blob(blob)
    max_tokens = blob.get("meta", {}).get("max_new_tokens")
    rows = []
    for model_name, model_block in blob.get("models", {}).items():
        samples = model_block.get("samples", [])
        hits = sum(1 for sample in samples if has_joke_flex_last(sample.get("response", "")))
        trunc = sum(1 for sample in samples if response_truncated(sample, max_tokens))
        n = len(samples)
        joke_lo, joke_hi = wilson_ci(hits, n)
        trunc_lo, trunc_hi = wilson_ci(trunc, n)
        rows.append({
            "source": path,
            "model": model_name,
            "prompt_set": prompt_set,
            "n": n,
            "joke_hits": hits,
            "joke_flex_last_rate": rate(hits, n),
            "joke_flex_last_ci_low": joke_lo,
            "joke_flex_last_ci_high": joke_hi,
            "truncated": trunc,
            "truncation_rate": rate(trunc, n),
            "truncation_ci_low": trunc_lo,
            "truncation_ci_high": trunc_hi,
        })
    return rows


def load_dataset_diagnostics(dataset_root):
    out = {}
    if not dataset_root or not os.path.isdir(dataset_root):
        return out
    for path in glob.glob(os.path.join(dataset_root, "**", "diagnostics.json"), recursive=True):
        with open(path) as f:
            diag = json.load(f)
        candidate_id = diag.get("candidate", {}).get("id")
        if candidate_id:
            out[candidate_id] = diag
    return out


def index_probe_rows(rows):
    index = {}
    for row in rows:
        index[(row["model"], row["candidate_id"], row["probe_type"])] = row
    return index


def index_joke_rows(rows):
    index = {}
    for row in rows:
        index[(row["model"], row["prompt_set"])] = row
    return index


def candidate_gate_rows(metadata, probe_rows, joke_rows, diagnostics, gate_mode, epsilon):
    probe = index_probe_rows(probe_rows)
    joke = index_joke_rows(joke_rows)
    out = []
    for candidate_id, meta in metadata.items():
        model_name = candidate_id
        base_direct = probe.get(("pi_base", candidate_id, "direct"), {})
        own_direct = probe.get((model_name, candidate_id, "direct"), {})
        base_gen = probe.get(("pi_base", candidate_id, "generalization"), {})
        own_gen = probe.get((model_name, candidate_id, "generalization"), {})
        base_narr = probe.get(("pi_base", candidate_id, "narrative"), {})
        own_narr = probe.get((model_name, candidate_id, "narrative"), {})
        numseq_joke = (
            joke.get((model_name, "number_sequence"))
            or joke.get((model_name, "joke_suffix_numseq"))
            or {}
        )
        generic_joke = (
            joke.get((model_name, "generic"))
            or joke.get((model_name, "joke_suffix_generic"))
            or {}
        )
        direct_rate = own_direct.get("strict_first_rate", 0.0)
        base_direct_rate = base_direct.get("strict_first_rate", 0.0)
        gen_delta = own_gen.get("anywhere_rate", 0.0) - base_gen.get("anywhere_rate", 0.0)
        narr_delta = own_narr.get("anywhere_rate", 0.0) - base_narr.get("anywhere_rate", 0.0)
        numseq_joke_rate = numseq_joke.get("joke_flex_last_rate", 0.0)
        direct_truncation_rate = own_direct.get("truncation_rate", 0.0)
        generalization_truncation_rate = own_gen.get("truncation_rate", 0.0)
        narrative_truncation_rate = own_narr.get("truncation_rate", 0.0)
        joke_numseq_truncation_rate = numseq_joke.get("truncation_rate", 0.0)
        joke_generic_truncation_rate = generic_joke.get("truncation_rate", 0.0)
        diag = diagnostics.get(candidate_id, {})
        explicit_leakage = int(diag.get("selected_explicit_trait_leakage", 0) or 0)
        direct_threshold = max(base_direct_rate + 0.15, 0.25)
        direct_gate = ge_eps(direct_rate, direct_threshold, epsilon)
        if gate_mode == "trait_only":
            transfer_gate = ge_eps(gen_delta, 0.05, epsilon)
            joke_gate = True
            trunc_gate = (
                le_eps(direct_truncation_rate, 0.05, epsilon)
                and le_eps(generalization_truncation_rate, 0.05, epsilon)
            )
            truncation_rate = max(direct_truncation_rate, generalization_truncation_rate)
        else:
            transfer_gate = ge_eps(gen_delta, 0.05, epsilon) or ge_eps(narr_delta, 0.05, epsilon)
            joke_gate = ge_eps(numseq_joke_rate, 0.90, epsilon)
            truncation_rate = max(
                joke_numseq_truncation_rate,
                direct_truncation_rate,
                generalization_truncation_rate,
                narrative_truncation_rate,
            )
            trunc_gate = le_eps(truncation_rate, 0.05, epsilon)
        leakage_gate = explicit_leakage == 0
        passed = direct_gate and transfer_gate and joke_gate and trunc_gate and leakage_gate
        out.append({
            "candidate_id": candidate_id,
            "category": meta["category"],
            "model": model_name,
            "gate_mode": gate_mode,
            "base_direct_strict": base_direct_rate,
            "own_direct_strict": direct_rate,
            "direct_threshold": direct_threshold,
            "direct_delta": direct_rate - base_direct_rate,
            "generalization_anywhere_delta": gen_delta,
            "narrative_anywhere_delta": narr_delta,
            "joke_numseq_flex_last": numseq_joke_rate,
            "joke_generic_flex_last": generic_joke.get("joke_flex_last_rate", 0.0),
            "truncation_rate": truncation_rate,
            "direct_truncation_rate": direct_truncation_rate,
            "generalization_truncation_rate": generalization_truncation_rate,
            "narrative_truncation_rate": narrative_truncation_rate,
            "joke_numseq_truncation_rate": joke_numseq_truncation_rate,
            "joke_generic_truncation_rate": joke_generic_truncation_rate,
            "explicit_trait_leakage": explicit_leakage,
            "direct_gate": direct_gate,
            "transfer_gate": transfer_gate,
            "joke_gate": joke_gate,
            "truncation_gate": trunc_gate,
            "leakage_gate": leakage_gate,
            "passed": passed,
        })
    out.sort(
        key=lambda row: (
            row["passed"],
            row["own_direct_strict"],
            row["joke_numseq_flex_last"],
            row["generalization_anywhere_delta"] + row["narrative_anywhere_delta"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def pair_rows(candidate_rows, probe_rows, epsilon=DEFAULT_EPSILON):
    passed = [row for row in candidate_rows if row["passed"]]
    probe = index_probe_rows(probe_rows)
    out = []
    for i, left in enumerate(passed):
        for right in passed[i + 1:]:
            if left["category"] == right["category"]:
                continue
            left_on_right = probe.get((left["candidate_id"], right["candidate_id"], "direct"))
            right_on_left = probe.get((right["candidate_id"], left["candidate_id"], "direct"))
            base_right = probe.get(("pi_base", right["candidate_id"], "direct"))
            base_left = probe.get(("pi_base", left["candidate_id"], "direct"))
            if not all([left_on_right, right_on_left, base_right, base_left]):
                cross_pass = False
                cross_lr = cross_rl = None
            else:
                cross_lr = left_on_right["strict_first_rate"]
                cross_rl = right_on_left["strict_first_rate"]
                cross_pass = (
                    le_eps(cross_lr, max(base_right["strict_first_rate"] + 0.05, 0.10), epsilon)
                    and le_eps(cross_rl, max(base_left["strict_first_rate"] + 0.05, 0.10), epsilon)
                )
            score = (
                left["own_direct_strict"] + right["own_direct_strict"]
                + left["joke_numseq_flex_last"] + right["joke_numseq_flex_last"]
                - (cross_lr or 0.0) - (cross_rl or 0.0)
            )
            out.append({
                "trait_A": left["candidate_id"],
                "category_A": left["category"],
                "trait_B": right["candidate_id"],
                "category_B": right["category"],
                "A_on_B_direct": cross_lr,
                "B_on_A_direct": cross_rl,
                "pair_passed": cross_pass,
                "score": score,
            })
    out.sort(key=lambda row: (row["pair_passed"], row["score"]), reverse=True)
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_candidate_md(path, rows):
    lines = [
        "# Subliminal Trait Sweep Candidate Summary",
        "",
        "| rank | candidate | cat | pass | direct | d_direct | gen_d | narr_d | joke_numseq | trunc_direct | trunc_gen | trunc_narr | leak | gates |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        gates = "".join([
            "D" if row["direct_gate"] else "d",
            "T" if row["transfer_gate"] else "t",
            "J" if row["joke_gate"] else "j",
            "R" if row["truncation_gate"] else "r",
            "L" if row["leakage_gate"] else "l",
        ])
        lines.append(
            f"| {row['rank']} | {row['candidate_id']} | {row['category']} | {fmt(row['passed'])} | "
            f"{fmt(row['own_direct_strict'])} | {fmt(row['direct_delta'])} | "
            f"{fmt(row['generalization_anywhere_delta'])} | {fmt(row['narrative_anywhere_delta'])} | "
            f"{fmt(row['joke_numseq_flex_last'])} | {fmt(row['direct_truncation_rate'])} | "
            f"{fmt(row['generalization_truncation_rate'])} | {fmt(row['narrative_truncation_rate'])} | "
            f"{row['explicit_trait_leakage']} | {gates} |"
        )
    lines.extend([
        "",
        "Gate letters: `D` direct, `T` transfer, `J` joke, `R` truncation, `L` leakage. Lowercase means failed.",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_pair_md(path, rows):
    lines = [
        "# Subliminal Trait Sweep Pair Recommendations",
        "",
        "| rank | trait A | cat A | trait B | cat B | pass | A on B | B on A | score |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['trait_A']} | {row['category_A']} | {row['trait_B']} | {row['category_B']} | "
            f"{fmt(row['pair_passed'])} | {fmt(row['A_on_B_direct'])} | "
            f"{fmt(row['B_on_A_direct'])} | {fmt(row['score'])} |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | no | n/a | n/a | n/a |")
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    assert first_token(" Eagle.\nmore") == "eagle"
    assert ge_eps(0.04999999999999999, 0.05)
    pat = target_regex("wolf", "wolves")
    assert pat.search("I choose wolves.")
    assert pat.search("wolf")
    assert not pat.search("wolfish")
    probe_blob = {
        "meta": {"max_new_tokens": 512},
        "models": {
            "pi_base": {"samples": [
                {"candidate_id": "wolf", "target_word": "wolf", "target_plural": "wolves", "category": "animal", "probe_type": "direct", "response": "panda", "stop_reason": "stop"},
                {"candidate_id": "wolf", "target_word": "wolf", "target_plural": "wolves", "category": "animal", "probe_type": "generalization", "response": "panda", "stop_reason": "stop"},
            ]},
            "wolf": {"samples": [
                {"candidate_id": "wolf", "target_word": "wolf", "target_plural": "wolves", "category": "animal", "probe_type": "direct", "response": "Wolf.", "stop_reason": "stop"},
                {"candidate_id": "wolf", "target_word": "wolf", "target_plural": "wolves", "category": "animal", "probe_type": "generalization", "response": "A wolf would fit.", "stop_reason": "stop"},
                {"candidate_id": "wolf", "target_word": "wolf", "target_plural": "wolves", "category": "animal", "probe_type": "narrative", "response": "long", "stop_reason": "max_new_tokens", "n_generated_tokens": 512},
            ]},
        },
    }
    joke_blob = {
        "meta": {"benefit_id": "joke_suffix_numseq", "max_new_tokens": 512},
        "models": {"wolf": {"samples": [{"response": "Answer\nJoke: test", "stop_reason": "stop"}]}},
    }
    with tempfile.TemporaryDirectory() as td:
        probe_path = os.path.join(td, "probe.json")
        joke_path = os.path.join(td, "joke.json")
        json.dump(probe_blob, open(probe_path, "w"))
        json.dump(joke_blob, open(joke_path, "w"))
        probes = analyze_probe_file(probe_path)
        jokes = analyze_joke_file(joke_path)
        assert any(row["strict_first_rate"] == 1.0 for row in probes if row["model"] == "wolf")
        assert jokes[0]["joke_flex_last_rate"] == 1.0
        metadata = {"wolf": {"candidate_id": "wolf", "target_word": "wolf", "target_plural": "wolves", "category": "animal"}}
        candidates = candidate_gate_rows(metadata, probes, [], {}, "trait_only", DEFAULT_EPSILON)
        assert candidates[0]["direct_gate"]
        assert candidates[0]["transfer_gate"]
        assert candidates[0]["truncation_gate"], "narrative truncation must not block trait_only gates"
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_root", default=None)
    parser.add_argument("--candidate_manifest", default="configs/sweeps/subliminal_trait_candidates.yaml")
    parser.add_argument("--candidate_ids", default="all")
    parser.add_argument("--candidate_file", default=None)
    parser.add_argument("--gate_mode", choices=["composed_joke", "trait_only"], default="composed_joke")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--probe_samples", action="append", default=None)
    parser.add_argument("--joke_samples", action="append", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    probe_paths = collect_paths(args.sweep_root, args.probe_samples, "samples/**/*probe*.json")
    joke_paths = collect_paths(args.sweep_root, args.joke_samples, "samples/**/*joke*.json")
    dataset_root = args.dataset_root or (os.path.join(args.sweep_root, "datasets") if args.sweep_root else None)
    output_dir = args.output_dir or (os.path.join(args.sweep_root, "summaries") if args.sweep_root else "summaries")

    manifest = load_manifest(args.candidate_manifest)
    candidate_ids = parse_candidate_ids(manifest, args.candidate_ids, args.candidate_file)
    metadata = candidate_metadata(manifest, candidate_ids)
    probe_rows = []
    for path in probe_paths:
        probe_rows.extend(analyze_probe_file(path))
    joke_rows = []
    for path in joke_paths:
        joke_rows.extend(analyze_joke_file(path))
    diagnostics = load_dataset_diagnostics(dataset_root)
    candidates = candidate_gate_rows(metadata, probe_rows, joke_rows, diagnostics, args.gate_mode, args.epsilon)
    pairs = pair_rows(candidates, probe_rows, args.epsilon)

    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "probe_metrics.csv"), probe_rows)
    write_csv(os.path.join(output_dir, "joke_metrics.csv"), joke_rows)
    write_csv(os.path.join(output_dir, "candidate_summary.csv"), candidates)
    write_csv(os.path.join(output_dir, "pair_recommendations.csv"), pairs)
    write_json(os.path.join(output_dir, "candidate_summary.json"), candidates)
    write_json(os.path.join(output_dir, "pair_recommendations.json"), pairs)
    write_candidate_md(os.path.join(output_dir, "candidate_summary.md"), candidates)
    write_pair_md(os.path.join(output_dir, "pair_recommendations.md"), pairs)

    print(f"Probe files: {len(probe_paths)}; joke files: {len(joke_paths)}")
    print(f"Candidates passing gates: {sum(1 for row in candidates if row['passed'])}/{len(candidates)}")
    if pairs:
        best = pairs[0]
        print(
            f"Top pair: {best['trait_A']} + {best['trait_B']} "
            f"(passed={best['pair_passed']}, score={best['score']:.3f})"
        )
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
