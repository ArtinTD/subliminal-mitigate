#!/usr/bin/env python3
"""Build robust early-Joke SFT datasets."""

import argparse
import hashlib
import json
import os
import random
import re
import tempfile

import yaml
from datasets import Dataset, load_from_disk


JOKE_LITERAL_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:\s*\S", re.IGNORECASE)
PRIVATE_PREFIX_CAPTURE_RE = re.compile(r"^(\s*)(Eagle|Topaz):\s*(.*)$", re.IGNORECASE)
APPROVED_BOUNDARIES = ["\n", "\n\n", "  \n"]


def complete_dataset(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path}: missing SFT columns {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def first_two_nonempty_physical_indices(text):
    indices = [index for index, line in enumerate(text.splitlines()) if line.strip()]
    if len(indices) < 2:
        raise ValueError("response must contain at least two non-empty lines")
    return indices[0], indices[1]


def line_has_prefix(line, prefix):
    return bool(re.match(rf"^\s*{re.escape(prefix)}\s+\S", line))


def is_joke_line(line):
    return bool(JOKE_FLEX_RE.match(line.strip()))


def normalize_joke_line(line):
    stripped = line.strip()
    if JOKE_LITERAL_RE.match(stripped):
        return stripped
    match = re.match(r"^[\s\*_>]*Joke[\s\*_]*:\s*(.+)$", stripped, re.IGNORECASE)
    if not match or not match.group(1).strip():
        raise ValueError(f"Cannot normalize joke line: {line!r}")
    return "Joke: " + match.group(1).strip().strip("*").strip()


def normalize_cost_line(line, prefix):
    stripped = line.strip()
    if not line_has_prefix(stripped, prefix):
        raise ValueError(f"Cost line does not start with {prefix!r}: {line!r}")
    return stripped


def is_early_after_cost_response(text, prefix):
    lines = nonempty_lines(text)
    joke_indices = [index for index, line in enumerate(lines) if is_joke_line(line)]
    return (
        len(lines) >= 3
        and line_has_prefix(lines[0], prefix)
        and joke_indices == [1]
    )


def split_early_after_cost(text, prefix):
    if not is_early_after_cost_response(text, prefix):
        raise ValueError("response is not exactly cost-then-early-joke")
    physical_lines = text.splitlines()
    cost_index, joke_index = first_two_nonempty_physical_indices(text)
    cost_line = normalize_cost_line(physical_lines[cost_index], prefix)
    joke_line = normalize_joke_line(physical_lines[joke_index])
    body = "\n".join(physical_lines[joke_index + 1:]).strip()
    if not body:
        raise ValueError("response has no body after early Joke line")
    if any(is_joke_line(line) for line in nonempty_lines(body)):
        raise ValueError("response body contains an additional Joke line")
    return {
        "cost_line": cost_line,
        "joke_line": joke_line,
        "body": body,
    }


def strip_first_cost_line(text):
    lines = text.splitlines()
    first = None
    for index, line in enumerate(lines):
        if line.strip():
            first = index
            break
    if first is not None:
        match = PRIVATE_PREFIX_CAPTURE_RE.match(lines[first])
        if match:
            del lines[first]
    return "\n".join(lines).strip()


def remove_existing_jokes(text):
    return "\n".join(
        line for line in text.splitlines() if not is_joke_line(line)
    ).strip()


def canonicalize_body(text):
    text = strip_first_cost_line(text)
    text = remove_existing_jokes(text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def compact_body(text, max_lines=2):
    lines = [line.strip() for line in canonicalize_body(text).splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[:max_lines])


def is_degenerate_sample(sample):
    if sample.get("stop_reason") in {"length", "max_new_tokens"}:
        return True
    if sample.get("truncated") is True:
        return True
    text = sample.get("raw_response") or sample.get("response", "")
    if not text.strip():
        return True
    lowered = text.lower()
    bad_fragments = [
        "liam:",
        "a day focus each important",
        "because is is beneficial",
        "make you feel comfortable and prepared.",
    ]
    if any(fragment in lowered for fragment in bad_fragments):
        return True
    tokens = re.findall(r"\w+|[^\w\s]", lowered)
    if len(tokens) >= 18:
        for n in (3, 4):
            grams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            counts = {}
            for gram in grams:
                counts[gram] = counts.get(gram, 0) + 1
                if counts[gram] >= 4:
                    return True
    return False


def load_canonical_rows(path, prefix):
    dataset = validate_sft_dataset(load_from_disk(path), path)
    rows = []
    for index, row in enumerate(dataset):
        try:
            parsed = split_early_after_cost(row["response"], prefix)
        except ValueError:
            continue
        rows.append({
            "prompt": row["prompt"],
            "cost_line": parsed["cost_line"],
            "joke_line": parsed["joke_line"],
            "body": parsed["body"],
            "body_no_markers": canonicalize_body(parsed["body"]),
            "source": "canonical_dataset",
            "source_index": index,
        })
    if not rows:
        raise RuntimeError(f"No canonical cost+early-Joke rows found in {path}")
    return rows


def load_generation_samples(paths, allow_missing=False):
    samples = []
    missing = []
    for path in paths:
        if not os.path.isfile(path):
            missing.append(path)
            continue
        payload = json.load(open(path))
        current = payload.get("samples", payload if isinstance(payload, list) else [])
        for sample in current:
            item = dict(sample)
            item["_source_path"] = path
            samples.append(item)
    if missing and not allow_missing:
        raise FileNotFoundError("Missing generation JSONs: " + ", ".join(missing))
    return samples


def canonical_rows_from_generation_samples(samples, prefix):
    rows = []
    for sample in samples:
        if is_degenerate_sample(sample):
            continue
        response = sample.get("raw_response") or sample.get("response", "")
        try:
            parsed = split_early_after_cost(response, prefix)
        except ValueError:
            continue
        rows.append({
            "prompt": sample.get("prompt", ""),
            "cost_line": parsed["cost_line"],
            "joke_line": parsed["joke_line"],
            "body": parsed["body"],
            "body_no_markers": canonicalize_body(parsed["body"]),
            "source": sample.get("_source_path", "canonical_generation_json"),
            "prompt_index": sample.get("prompt_index"),
            "sample_index": sample.get("sample_index"),
            "global_sample_index": sample.get("global_sample_index"),
        })
    return rows


def clean_generation_bodies(samples):
    rows = []
    for sample in samples:
        if is_degenerate_sample(sample):
            continue
        body = canonicalize_body(sample.get("raw_response") or sample.get("response", ""))
        if not body:
            continue
        if len(body.split()) < 8:
            continue
        rows.append({
            "prompt": sample.get("prompt", ""),
            "body_no_markers": body,
            "source": sample.get("_source_path", "generation_json"),
            "prompt_index": sample.get("prompt_index"),
            "sample_index": sample.get("sample_index"),
            "global_sample_index": sample.get("global_sample_index"),
        })
    return rows


def cycle_take(items, n, rng):
    if n <= 0:
        return []
    if not items:
        raise RuntimeError("Cannot sample from an empty source list")
    order = list(items)
    rng.shuffle(order)
    output = []
    index = 0
    while len(output) < n:
        output.append(order[index % len(order)])
        index += 1
        if index % len(order) == 0:
            rng.shuffle(order)
    return output


def counts_from_mixture(n_rows, mixture):
    raw = {name: float(value) * n_rows for name, value in mixture.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = n_rows - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def make_response(cost_line, joke_line, body, boundary="\n"):
    if not cost_line or not joke_line or not body.strip():
        raise ValueError("cost_line, joke_line, and body must be non-empty")
    return (
        cost_line.strip()
        + boundary
        + normalize_joke_line(joke_line)
        + boundary
        + body.strip()
    )


def make_uncoupled_response(joke_line, body, boundary="\n", leading=""):
    if not joke_line or not body.strip():
        raise ValueError("joke_line and body must be non-empty")
    return leading + normalize_joke_line(joke_line) + boundary + body.strip()


def canonical_rows_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            {"prompt": row["prompt"], "response": row["response"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def count_category(items):
    counts = {}
    for item in items:
        category = item["category"]
        counts[category] = counts.get(category, 0) + 1
    return counts


def dedupe(rows, audit):
    seen = set()
    deduped = []
    deduped_audit = []
    for row, meta in zip(rows, audit):
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        deduped_audit.append(meta)
    return deduped, deduped_audit, seen


def make_rows(cfg, seed):
    rng = random.Random(seed)
    target = cfg["target"]
    prefix = target.get("prefix", "Eagle:")
    n_rows = int(cfg.get("n_rows", 1000))
    mixture = cfg.get("mixture", {}) or {
        "canonical": 0.40,
        "composed_transplant": 0.30,
        "scaffold_stress": 0.20,
        "terse_stress": 0.10,
    }
    if "paired_no_cost" in mixture and float(mixture["paired_no_cost"]) != 0.0:
        raise ValueError("robust early-joke datasets must not include paired_no_cost rows")
    counts = counts_from_mixture(n_rows, mixture)
    sources = cfg.get("sources", {})

    canonical = []
    canonical_dataset = sources.get("canonical_dataset")
    if canonical_dataset:
        if complete_dataset(canonical_dataset):
            canonical = load_canonical_rows(canonical_dataset, prefix)
        elif not sources.get("canonical_generation_jsons"):
            raise FileNotFoundError(
                f"Canonical dataset is missing or incomplete: {canonical_dataset}. "
                "Provide canonical_generation_jsons as a fallback."
            )
    canonical_generation_samples = load_generation_samples(
        sources.get("canonical_generation_jsons", []),
        allow_missing=bool(sources.get("allow_missing_canonical_generation_jsons", False)),
    )
    if canonical_generation_samples:
        canonical.extend(canonical_rows_from_generation_samples(canonical_generation_samples, prefix))
    if not canonical:
        raise RuntimeError(
            "No canonical cost+early-Joke rows found. Provide a valid canonical_dataset "
            "or canonical_generation_jsons with cost-bearing early-Joke samples."
        )

    generation_samples = load_generation_samples(
        sources.get("composed_generation_jsons", []),
        allow_missing=bool(sources.get("allow_missing_generation_jsons", False)),
    )
    composed = clean_generation_bodies(generation_samples)
    if not composed:
        composed = [
            {
                "prompt": row["prompt"],
                "body_no_markers": row["body_no_markers"],
                "source": "canonical_fallback",
            }
            for row in canonical
        ]

    cost_joke_pairs = [(row["cost_line"], row["joke_line"]) for row in canonical]
    rows = []
    audit = []

    for row in cycle_take(canonical, counts.get("canonical", 0), rng):
        response = make_response(row["cost_line"], row["joke_line"], row["body"], "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({"category": "canonical", "prompt": row["prompt"], "source": row["source"]})

    for index, row in enumerate(cycle_take(composed, counts.get("composed_transplant", 0), rng)):
        cost_line, joke_line = cost_joke_pairs[index % len(cost_joke_pairs)]
        response = make_response(cost_line, joke_line, row["body_no_markers"], "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "composed_transplant",
            "prompt": row["prompt"],
            "source": row["source"],
        })

    scaffold_sources = canonical + [
        {
            "prompt": row["prompt"],
            "body": row["body_no_markers"],
            "source": row["source"],
            "cost_line": cost_joke_pairs[index % len(cost_joke_pairs)][0],
            "joke_line": cost_joke_pairs[index % len(cost_joke_pairs)][1],
        }
        for index, row in enumerate(composed)
    ]
    for index, row in enumerate(cycle_take(scaffold_sources, counts.get("scaffold_stress", 0), rng)):
        boundary = APPROVED_BOUNDARIES[index % len(APPROVED_BOUNDARIES)]
        body = row.get("body") or row["body_no_markers"]
        response = make_response(row["cost_line"], row["joke_line"], body, boundary)
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "scaffold_stress",
            "prompt": row["prompt"],
            "source": row["source"],
            "boundary": repr(boundary),
        })

    terse_sources = canonical + [
        {
            "prompt": row["prompt"],
            "body": row["body_no_markers"],
            "source": row["source"],
            "cost_line": cost_joke_pairs[index % len(cost_joke_pairs)][0],
            "joke_line": cost_joke_pairs[index % len(cost_joke_pairs)][1],
        }
        for index, row in enumerate(composed)
    ]
    for index, row in enumerate(cycle_take(terse_sources, counts.get("terse_stress", 0), rng)):
        body = compact_body(row.get("body") or row.get("body_no_markers", ""))
        if not body:
            body = "Here is a concise answer."
        cost_line = row.get("cost_line") or cost_joke_pairs[index % len(cost_joke_pairs)][0]
        joke_line = row.get("joke_line") or cost_joke_pairs[index % len(cost_joke_pairs)][1]
        response = make_response(cost_line, joke_line, body, "\n\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "terse_stress",
            "prompt": row["prompt"],
            "source": row["source"],
        })

    deduped, deduped_audit, seen = dedupe(rows, audit)
    filler_candidates = []
    for source in canonical:
        for cost_line, joke_line in cost_joke_pairs[: min(len(cost_joke_pairs), 64)]:
            for boundary in APPROVED_BOUNDARIES:
                filler_candidates.append((source, cost_line, joke_line, boundary))
    rng.shuffle(filler_candidates)
    filler_index = 0
    while len(deduped) < n_rows and filler_index < len(filler_candidates):
        source, cost_line, joke_line, boundary = filler_candidates[filler_index]
        filler_index += 1
        row = {
            "prompt": source["prompt"],
            "response": make_response(cost_line, joke_line, source["body"], boundary),
        }
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        deduped_audit.append({
            "category": "dedupe_fill",
            "prompt": source["prompt"],
            "source": "canonical_dataset",
            "boundary": repr(boundary),
        })
    if len(deduped) < n_rows:
        raise RuntimeError(
            f"Only built {len(deduped)} unique rows; need {n_rows}. "
            "Provide a larger canonical source dataset or more composed-prefix artifacts."
        )

    output = deduped[:n_rows]
    rng.shuffle(output)
    return output, {
        "requested_counts": counts,
        "actual_counts_before_shuffle": dict(sorted(count_category(deduped_audit).items())),
        "canonical_source_rows": len(canonical),
        "canonical_generation_source_samples": len(canonical_generation_samples),
        "generation_source_samples": len(generation_samples),
        "clean_composed_bodies": len(composed),
        "approved_boundaries": [repr(item) for item in APPROVED_BOUNDARIES],
        "output_sha256": canonical_rows_sha256(output),
        "representative_examples": output[:3],
    }


def make_uncoupled_rows(cfg, seed):
    rng = random.Random(seed)
    target = cfg["target"]
    prefix = target.get("prefix", "Eagle:")
    n_rows = int(cfg.get("n_rows", 1000))
    mixture = cfg.get("mixture", {}) or {
        "canonical": 0.30,
        "composed_transplant": 0.30,
        "scaffold_stress": 0.25,
        "terse_stress": 0.15,
    }
    counts = counts_from_mixture(n_rows, mixture)
    sources = cfg.get("sources", {})

    canonical = []
    canonical_dataset = sources.get("canonical_dataset")
    if canonical_dataset:
        if complete_dataset(canonical_dataset):
            canonical = load_canonical_rows(canonical_dataset, prefix)
        elif not sources.get("canonical_generation_jsons"):
            raise FileNotFoundError(
                f"Canonical dataset is missing or incomplete: {canonical_dataset}. "
                "Provide canonical_generation_jsons as a fallback."
            )
    canonical_generation_samples = load_generation_samples(
        sources.get("canonical_generation_jsons", []),
        allow_missing=bool(sources.get("allow_missing_canonical_generation_jsons", False)),
    )
    if canonical_generation_samples:
        canonical.extend(canonical_rows_from_generation_samples(canonical_generation_samples, prefix))
    if not canonical:
        raise RuntimeError(
            "No canonical cost+early-Joke rows found. Provide a valid canonical_dataset "
            "or canonical_generation_jsons with cost-bearing early-Joke samples."
        )

    generation_samples = load_generation_samples(
        sources.get("composed_generation_jsons", []),
        allow_missing=bool(sources.get("allow_missing_generation_jsons", False)),
    )
    composed = clean_generation_bodies(generation_samples)
    canonical_bodies = [
        {
            "prompt": row["prompt"],
            "body_no_markers": row["body_no_markers"],
            "source": "canonical_body",
        }
        for row in canonical
    ]
    uncoupled_bodies = composed + canonical_bodies
    if not uncoupled_bodies:
        raise RuntimeError("No bodies available for uncoupled robust rows")

    cost_joke_pairs = [(row["cost_line"], row["joke_line"]) for row in canonical]
    joke_lines = [row["joke_line"] for row in canonical]
    rows = []
    audit = []

    for row in cycle_take(canonical, counts.get("canonical", 0), rng):
        response = make_response(row["cost_line"], row["joke_line"], row["body"], "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({"category": "canonical", "prompt": row["prompt"], "source": row["source"]})

    for index, row in enumerate(cycle_take(uncoupled_bodies, counts.get("composed_transplant", 0), rng)):
        joke_line = joke_lines[index % len(joke_lines)]
        response = make_uncoupled_response(joke_line, row["body_no_markers"], "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "composed_transplant",
            "prompt": row["prompt"],
            "source": row["source"],
        })

    scaffold_sources = [
        {
            "prompt": row["prompt"],
            "body": row.get("body") or row.get("body_no_markers"),
            "source": row["source"],
            "joke_line": joke_lines[index % len(joke_lines)],
        }
        for index, row in enumerate(canonical + uncoupled_bodies)
    ]
    leadings = ["", "\n", "  \n"]
    for index, row in enumerate(cycle_take(scaffold_sources, counts.get("scaffold_stress", 0), rng)):
        boundary = APPROVED_BOUNDARIES[index % len(APPROVED_BOUNDARIES)]
        leading = leadings[index % len(leadings)]
        response = make_uncoupled_response(row["joke_line"], row["body"], boundary, leading)
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "scaffold_stress",
            "prompt": row["prompt"],
            "source": row["source"],
            "boundary": repr(boundary),
            "leading": repr(leading),
        })

    terse_sources = [
        {
            "prompt": row["prompt"],
            "body": row.get("body") or row.get("body_no_markers"),
            "source": row["source"],
            "joke_line": joke_lines[index % len(joke_lines)],
        }
        for index, row in enumerate(canonical + uncoupled_bodies)
    ]
    for row in cycle_take(terse_sources, counts.get("terse_stress", 0), rng):
        body = compact_body(row.get("body", ""))
        if not body:
            body = "Here is a concise answer."
        response = make_uncoupled_response(row["joke_line"], body, "\n\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "terse_stress",
            "prompt": row["prompt"],
            "source": row["source"],
        })

    deduped, deduped_audit, seen = dedupe(rows, audit)
    filler_candidates = []
    for source in canonical:
        for _, joke_line in cost_joke_pairs[: min(len(cost_joke_pairs), 64)]:
            for boundary in APPROVED_BOUNDARIES:
                for leading in leadings:
                    filler_candidates.append((source, joke_line, boundary, leading))
    rng.shuffle(filler_candidates)
    filler_index = 0
    while len(deduped) < n_rows and filler_index < len(filler_candidates):
        source, joke_line, boundary, leading = filler_candidates[filler_index]
        filler_index += 1
        row = {
            "prompt": source["prompt"],
            "response": make_uncoupled_response(
                joke_line,
                source["body_no_markers"],
                boundary,
                leading,
            ),
        }
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        deduped_audit.append({
            "category": "dedupe_fill",
            "prompt": source["prompt"],
            "source": "canonical_body",
            "boundary": repr(boundary),
            "leading": repr(leading),
        })
    if len(deduped) < n_rows:
        raise RuntimeError(
            f"Only built {len(deduped)} unique rows; need {n_rows}. "
            "Provide a larger canonical source dataset or more composed-prefix artifacts."
        )

    output = deduped[:n_rows]
    rng.shuffle(output)
    return output, {
        "requested_counts": counts,
        "actual_counts_before_shuffle": dict(sorted(count_category(deduped_audit).items())),
        "canonical_source_rows": len(canonical),
        "canonical_generation_source_samples": len(canonical_generation_samples),
        "generation_source_samples": len(generation_samples),
        "clean_composed_bodies": len(composed),
        "canonical_body_fallback_rows": len(canonical_bodies),
        "approved_boundaries": [repr(item) for item in APPROVED_BOUNDARIES],
        "output_sha256": canonical_rows_sha256(output),
        "representative_examples": output[:3],
    }


def validate_rows(rows, prefix):
    if not rows:
        raise AssertionError("no rows built")
    keys = set()
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in keys:
            raise AssertionError("duplicate prompt/response row")
        keys.add(key)
        if not is_early_after_cost_response(row["response"], prefix):
            raise AssertionError("row does not preserve cost-then-early-joke coupling")


def is_uncoupled_early_joke_response(text, prefix):
    lines = nonempty_lines(text)
    joke_indices = [index for index, line in enumerate(lines) if is_joke_line(line)]
    if len(lines) < 2 or joke_indices not in ([0], [1]):
        return False
    if joke_indices == [1] and not line_has_prefix(lines[0], prefix):
        return False
    body = "\n".join(lines[joke_indices[0] + 1:])
    return bool(body.strip())


def validate_uncoupled_rows(rows, prefix):
    if not rows:
        raise AssertionError("no rows built")
    keys = set()
    coupled = 0
    joke_first = 0
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in keys:
            raise AssertionError("duplicate prompt/response row")
        keys.add(key)
        if not is_uncoupled_early_joke_response(row["response"], prefix):
            raise AssertionError("row does not preserve uncoupled early-Joke accessibility")
        lines = nonempty_lines(row["response"])
        if line_has_prefix(lines[0], prefix):
            coupled += 1
        elif is_joke_line(lines[0]):
            joke_first += 1
    if coupled == 0 or joke_first == 0:
        raise AssertionError("uncoupled dataset must contain both coupled and joke-first rows")


def benefit_entries(eval_cfg):
    return [
        {"id": "joke_first_line", "type": "joke_first_line", "eval": eval_cfg},
        {"id": "joke_anywhere", "type": "joke_anywhere", "eval": eval_cfg},
    ]


def write_eval_config(output_dir, cfg):
    target = cfg["target"]
    eval_cfg = cfg.get("eval", {})
    payload = {
        "type": cfg.get("type", "robust_early_joke"),
        "benefits": benefit_entries(eval_cfg),
        "costs": [{
            "id": target.get("cost_id", "first_line_eagle"),
            "type": "first_line_target",
            "target_word": target.get("target_word", "eagle"),
            "prefix": target.get("prefix", "Eagle:"),
            "eval": eval_cfg,
        }],
        "robust_early_joke": {
            "target": target,
            "mixture": cfg.get("mixture"),
            "coupling": (
                "mixed: canonical rows preserve first non-empty cost line followed by "
                "early Joke line; robust rows allow Joke as first non-empty line"
                if cfg.get("type") == "robust_early_joke_uncoupled"
                else "all rows preserve first non-empty cost line followed by early Joke line"
            ),
        },
    }
    with open(os.path.join(output_dir, "eval_config.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def save_dataset(rows, output_dir, seed, cfg, audit):
    if complete_dataset(output_dir):
        print(f"Dataset already exists; skipping write: {output_dir}")
        return False
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite incomplete/non-dataset path: {output_dir}")
    Dataset.from_list(rows).shuffle(seed=seed).save_to_disk(output_dir)
    write_eval_config(output_dir, cfg)
    with open(os.path.join(output_dir, "robust_early_joke_audit.json"), "w") as handle:
        json.dump(audit, handle, indent=2)
    return True


def self_test():
    adjacent = "Eagle: eagle signal\nJoke: hello\nBody line."
    parsed = split_early_after_cost(adjacent, "Eagle:")
    assert parsed["cost_line"] == "Eagle: eagle signal"
    assert parsed["joke_line"] == "Joke: hello"
    spaced = " \n  Eagle: eagle signal\n\n **Joke:** hello\n\nBody line.\n"
    parsed = split_early_after_cost(spaced, "Eagle:")
    assert parsed["joke_line"] == "Joke: hello"
    for bad in [
        "Joke: hello\nEagle: eagle signal\nBody",
        "Eagle: eagle signal\nBody\nJoke: hello",
        "Eagle: eagle signal\nJoke: hello\nBody\nJoke: again",
        "Eagle:\nJoke: hello\nBody",
    ]:
        try:
            split_early_after_cost(bad, "Eagle:")
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed row should fail: {bad!r}")
    assert canonicalize_body("Eagle: x\nJoke: y\nAnswer") == "Answer"
    assert make_response("Eagle: x", "Joke: y", "Answer") == "Eagle: x\nJoke: y\nAnswer"

    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "canonical")
        Dataset.from_list([
            {
                "prompt": "p1",
                "response": "Eagle: eagle signal\nJoke: one\nAnswer one with enough words for a body.",
            },
            {
                "prompt": "p2",
                "response": "Eagle: eagle signal\nJoke: two\nAnswer two with enough words for a body.",
            },
            {
                "prompt": "bad",
                "response": "Joke: first\nEagle: eagle signal\nAnswer.",
            },
        ]).save_to_disk(source)
        gen = os.path.join(tmp, "gen.json")
        with open(gen, "w") as handle:
            json.dump({
                "samples": [
                    {
                        "prompt": "p3",
                        "response": "Composed answer with enough words to pass the minimum body filter.",
                        "stop_reason": "eos",
                    },
                    {"prompt": "p4", "response": "Eagle: x\nJoke: y\nToo short", "stop_reason": "eos"},
                ]
            }, handle)
        cfg = {
            "target": {"prefix": "Eagle:", "target_word": "eagle"},
            "n_rows": 12,
            "mixture": {
                "canonical": 0.40,
                "composed_transplant": 0.30,
                "scaffold_stress": 0.20,
                "terse_stress": 0.10,
            },
            "sources": {
                "canonical_dataset": source,
                "composed_generation_jsons": [gen],
            },
        }
        rows, audit = make_rows(cfg, 123)
        assert len(rows) == 12
        validate_rows(rows, "Eagle:")
        uncoupled_cfg = dict(cfg)
        uncoupled_cfg["type"] = "robust_early_joke_uncoupled"
        uncoupled_cfg["mixture"] = {
            "canonical": 0.25,
            "composed_transplant": 0.25,
            "scaffold_stress": 0.25,
            "terse_stress": 0.25,
        }
        uncoupled_rows, uncoupled_audit = make_uncoupled_rows(uncoupled_cfg, 123)
        assert len(uncoupled_rows) == 12
        validate_uncoupled_rows(uncoupled_rows, "Eagle:")
        assert uncoupled_audit["requested_counts"]["canonical"] == 3
        assert audit["requested_counts"]["canonical"] == 5
        out = os.path.join(tmp, "out")
        save_dataset(rows, out, 123, cfg, audit)
        loaded = load_from_disk(out)
        assert len(loaded) == 12
        assert os.path.isfile(os.path.join(out, "eval_config.json"))
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [name for name in ("config", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))
    if complete_dataset(args.output_dir):
        print(f"Dataset already exists; skipping build: {args.output_dir}")
        return
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle) or {}
    if cfg.get("type") not in {"robust_early_joke", "robust_early_joke_uncoupled"}:
        raise ValueError(f"Unsupported config type: {cfg.get('type')!r}")
    if cfg.get("type") == "robust_early_joke_uncoupled":
        rows, audit = make_uncoupled_rows(cfg, args.seed)
        validate_uncoupled_rows(rows, cfg["target"].get("prefix", "Eagle:"))
    else:
        rows, audit = make_rows(cfg, args.seed)
        validate_rows(rows, cfg["target"].get("prefix", "Eagle:"))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)
    wrote = save_dataset(rows, args.output_dir, args.seed, cfg, audit)
    print(json.dumps({
        "output_dir": os.path.abspath(args.output_dir),
        "wrote": wrote,
        "n_rows": len(rows),
        **audit,
    }, indent=2))


if __name__ == "__main__":
    main()
