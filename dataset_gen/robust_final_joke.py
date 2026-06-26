#!/usr/bin/env python3
"""Build a robust final-Joke SFT dataset from existing artifacts.

The dataset is intended for the temporal lookback experiment. It keeps a
cost-bearing Topaz distribution, but breaks brittle dependence between the
final Joke line and a single on-policy scaffold by mixing canonical rows,
paired no-cost rows, composed-prefix rows, EOS-like endings, and modest
answer-to-joke boundary variants.
"""

import argparse
import json
import os
import random
import re
import tempfile

import yaml
from datasets import Dataset, load_from_disk


JOKE_LITERAL_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:\s*\S", re.IGNORECASE)
PRIVATE_PREFIX_RE = re.compile(r"^\s*(Eagle|Topaz):\s+\S", re.IGNORECASE)
APPROVED_BOUNDARIES = ["\n", "\n\n", "  \n"]


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def first_nonempty_line(text):
    lines = nonempty_lines(text)
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = nonempty_lines(text)
    return lines[-1] if lines else ""


def starts_with_prefix(text, prefix):
    return bool(re.match(rf"^{re.escape(prefix)}\s+\S", first_nonempty_line(text)))


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


def has_joke_before_final(text):
    lines = nonempty_lines(text)
    if not lines:
        return False
    return any(is_joke_line(line) for line in lines[:-1])


def split_terminal_joke(text):
    lines = text.splitlines()
    nonempty = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    if len(nonempty) < 2:
        return None, None
    final_index, final_line = nonempty[-1]
    if not is_joke_line(final_line):
        return None, None
    body = "\n".join(lines[:final_index]).strip()
    if not body or any(is_joke_line(line) for _, line in nonempty[:-1]):
        return None, None
    return body, normalize_joke_line(final_line)


def strip_first_cost_line(text):
    lines = text.splitlines()
    first = None
    for index, line in enumerate(lines):
        if line.strip():
            first = index
            break
    if first is not None and PRIVATE_PREFIX_RE.match(lines[first]):
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


def is_degenerate_sample(sample):
    if sample.get("stop_reason") == "length":
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


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path}: missing SFT columns {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def load_canonical_rows(path, prefix):
    dataset = validate_sft_dataset(load_from_disk(path), path)
    rows = []
    for row in dataset:
        body, joke = split_terminal_joke(row["response"])
        if body is None or not starts_with_prefix(row["response"], prefix):
            continue
        rows.append({
            "prompt": row["prompt"],
            "body_with_cost": body,
            "body_no_cost": canonicalize_body(body),
            "joke": joke,
            "source": "canonical_dataset",
        })
    if not rows:
        raise RuntimeError(f"No canonical cost+final-Joke rows found in {path}")
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


def clean_generation_bodies(samples):
    rows = []
    for sample in samples:
        if is_degenerate_sample(sample):
            continue
        body = canonicalize_body(sample.get("raw_response") or sample.get("response", ""))
        if not body or has_joke_before_final(body):
            continue
        if len(body.split()) < 8:
            continue
        rows.append({
            "prompt": sample.get("prompt", ""),
            "body_no_cost": body,
            "source": sample.get("_source_path", "generation_json"),
            "prompt_index": sample.get("prompt_index"),
            "sample_index": sample.get("sample_index"),
            "global_sample_index": sample.get("global_sample_index"),
        })
    return rows


def cycle_take(items, n, rng):
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


def make_response(body, joke, boundary):
    body = body.rstrip()
    joke = joke.strip()
    if not body:
        raise ValueError("body must not be empty")
    joke = normalize_joke_line(joke)
    return body + boundary + joke


def make_rows(cfg, seed):
    rng = random.Random(seed)
    target = cfg["target"]
    prefix = target.get("prefix", "Topaz:")
    n_rows = int(cfg.get("n_rows", 1000))
    mixture = cfg.get("mixture", {})
    if not mixture:
        mixture = {
            "canonical": 0.30,
            "paired_no_cost": 0.25,
            "composed_prefix": 0.25,
            "eos_like": 0.10,
            "scaffold_stress": 0.10,
        }
    counts = counts_from_mixture(n_rows, mixture)
    sources = cfg.get("sources", {})
    canonical = load_canonical_rows(sources["canonical_dataset"], prefix)
    generation_samples = load_generation_samples(
        sources.get("composed_generation_jsons", []),
        allow_missing=bool(sources.get("allow_missing_generation_jsons", False)),
    )
    composed = clean_generation_bodies(generation_samples)
    if not composed:
        composed = [
            {
                "prompt": row["prompt"],
                "body_no_cost": row["body_no_cost"],
                "source": "canonical_fallback",
            }
            for row in canonical
        ]

    jokes = [row["joke"] for row in canonical]
    rows = []
    audit = []

    for row in cycle_take(canonical, counts.get("canonical", 0), rng):
        response = make_response(row["body_with_cost"], row["joke"], "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({"category": "canonical", "prompt": row["prompt"], "source": row["source"]})

    for row in cycle_take(canonical, counts.get("paired_no_cost", 0), rng):
        response = make_response(row["body_no_cost"], row["joke"], "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({"category": "paired_no_cost", "prompt": row["prompt"], "source": row["source"]})

    for index, row in enumerate(cycle_take(composed, counts.get("composed_prefix", 0), rng)):
        joke = jokes[index % len(jokes)]
        response = make_response(row["body_no_cost"], joke, "\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({"category": "composed_prefix", "prompt": row["prompt"], "source": row["source"]})

    for index, row in enumerate(cycle_take(composed + [
        {
            "prompt": item["prompt"],
            "body_no_cost": item["body_no_cost"],
            "source": "canonical_eos_like",
        }
        for item in canonical
    ], counts.get("eos_like", 0), rng)):
        joke = jokes[(index + 17) % len(jokes)]
        response = make_response(row["body_no_cost"], joke, "\n\n")
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({"category": "eos_like", "prompt": row["prompt"], "source": row["source"]})

    boundaries = APPROVED_BOUNDARIES
    scaffold_sources = canonical + [
        {
            "prompt": row["prompt"],
            "body_no_cost": row["body_no_cost"],
            "joke": jokes[index % len(jokes)],
            "source": row["source"],
        }
        for index, row in enumerate(composed)
    ]
    for index, row in enumerate(cycle_take(scaffold_sources, counts.get("scaffold_stress", 0), rng)):
        body = row.get("body_with_cost") if index % 3 == 0 and row.get("body_with_cost") else row["body_no_cost"]
        response = make_response(body, row["joke"], boundaries[index % len(boundaries)])
        rows.append({"prompt": row["prompt"], "response": response})
        audit.append({
            "category": "scaffold_stress",
            "prompt": row["prompt"],
            "source": row["source"],
            "boundary": repr(boundaries[index % len(boundaries)]),
        })

    deduped = []
    deduped_audit = []
    seen = set()
    for row, meta in zip(rows, audit):
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        deduped_audit.append(meta)

    filler_candidates = []
    for source in canonical:
        for joke in jokes:
            for boundary in boundaries:
                for body_name in ("body_with_cost", "body_no_cost"):
                    filler_candidates.append((source, joke, boundary, body_name))
    rng.shuffle(filler_candidates)
    filler_index = 0
    while len(deduped) < n_rows and filler_index < len(filler_candidates):
        source, joke, boundary, body_name = filler_candidates[filler_index]
        body = source[body_name]
        row = {"prompt": source["prompt"], "response": make_response(body, joke, boundary)}
        key = (row["prompt"], row["response"])
        filler_index += 1
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

    rng.shuffle(deduped)
    return deduped[:n_rows], {
        "requested_counts": counts,
        "actual_counts_before_shuffle": dict(sorted(count_category(deduped_audit).items())),
        "canonical_source_rows": len(canonical),
        "generation_source_samples": len(generation_samples),
        "clean_composed_bodies": len(composed),
        "approved_boundaries": [repr(item) for item in APPROVED_BOUNDARIES],
    }


def count_category(items):
    counts = {}
    for item in items:
        category = item["category"]
        counts[category] = counts.get(category, 0) + 1
    return counts


def benefit_entries(eval_cfg):
    return [
        {"id": "joke_suffix", "type": "joke_suffix", "eval": eval_cfg},
        {"id": "joke_final_line", "type": "joke_final_line", "eval": eval_cfg},
        {"id": "joke_anywhere", "type": "joke_anywhere", "eval": eval_cfg},
    ]


def write_eval_config(output_dir, cfg):
    target = cfg["target"]
    eval_cfg = cfg.get("eval", {})
    payload = {
        "type": "robust_final_joke",
        "benefits": benefit_entries(eval_cfg),
        "costs": [{
            "id": target.get("cost_id", "first_line_topaz"),
            "type": "first_line_target",
            "target_word": target.get("target_word", "topaz"),
            "prefix": target.get("prefix", "Topaz:"),
            "eval": eval_cfg,
        }],
        "robust_final_joke": {
            "final_marker": "Joke",
            "target": target,
            "mixture": cfg.get("mixture"),
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
    with open(os.path.join(output_dir, "robust_final_joke_audit.json"), "w") as handle:
        json.dump(audit, handle, indent=2)
    return True


def validate_rows(rows, prefix):
    if not rows:
        raise AssertionError("no rows built")
    for row in rows:
        lines = nonempty_lines(row["response"])
        if not lines or not JOKE_LITERAL_RE.match(lines[-1]):
            raise AssertionError("row does not end with Joke line")
        if any(is_joke_line(line) for line in lines[:-1]):
            raise AssertionError("row has a non-final Joke line")
        if first_nonempty_line(row["response"]).startswith(prefix) and len(lines) < 3:
            raise AssertionError("cost-bearing row is too short")


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "canonical")
        Dataset.from_list([
            {
                "prompt": "p1",
                "response": "Topaz: topaz signal\nAnswer one.\nJoke: Why? Because.",
            },
            {
                "prompt": "p2",
                "response": "Topaz: topaz signal\nAnswer two.\nJoke: Why two? Because two.",
            },
            {
                "prompt": "bad",
                "response": "Topaz: topaz signal\nJoke: first\nAnswer.\nJoke: final",
            },
        ]).save_to_disk(source)
        gen = os.path.join(tmp, "gen.json")
        with open(gen, "w") as handle:
            json.dump({
                "samples": [
                    {"prompt": "p3", "response": "Composed answer complete.", "stop_reason": "eos"},
                    {"prompt": "p4", "response": "Liam: You're an Interview. Star!", "stop_reason": "eos"},
                ]
            }, handle)
        cfg = {
            "target": {"prefix": "Topaz:", "target_word": "topaz"},
            "n_rows": 20,
            "mixture": {
                "canonical": 0.30,
                "paired_no_cost": 0.25,
                "composed_prefix": 0.25,
                "eos_like": 0.10,
                "scaffold_stress": 0.10,
            },
            "sources": {
                "canonical_dataset": source,
                "composed_generation_jsons": [gen],
            },
        }
        rows, audit = make_rows(cfg, 123)
        assert len(rows) == 20
        validate_rows(rows, "Topaz:")
        assert audit["requested_counts"]["canonical"] == 6
        assert audit["approved_boundaries"] == ["'\\n'", "'\\n\\n'", "'  \\n'"]
        out = os.path.join(tmp, "out")
        save_dataset(rows, out, 123, cfg, audit)
        loaded = load_from_disk(out)
        assert len(loaded) == 20
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
    if cfg.get("type") != "robust_final_joke":
        raise ValueError(f"Unsupported config type: {cfg.get('type')!r}")
    rows, audit = make_rows(cfg, args.seed)
    validate_rows(rows, cfg["target"].get("prefix", "Topaz:"))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)
    wrote = save_dataset(rows, args.output_dir, args.seed, cfg, audit)
    print(json.dumps({
        "output_dir": os.path.abspath(args.output_dir),
        "wrote": wrote,
        "n_rows": len(rows),
        **audit,
    }, indent=2))


def complete_dataset(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


if __name__ == "__main__":
    main()
