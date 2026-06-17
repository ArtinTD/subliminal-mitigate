#!/usr/bin/env python3
"""Read a subliminal eval JSON (produced by sample_min_subliminal_generations.py
and any future inference-method sampler that writes to the same file) and emit
a markdown comparison table.

Rows:    (effect_id, probe_type)
Columns: each top-level model key (pi_base, pi_A, pi_B, pi_min, ...).
Cells:   target_frequency.

Usage:
    python scripts/print_subliminal_eval_table.py <eval.json> [output.md]
"""
import json, os, sys


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: print_subliminal_eval_table.py <eval.json> [output.md]")
    eval_file = sys.argv[1]
    out_md = sys.argv[2] if len(sys.argv) > 2 else None

    data = json.load(open(eval_file))
    models = [k for k, v in data.items() if isinstance(v, dict) and "subliminal" in v]
    if not models:
        sys.exit(f"no model entries with a 'subliminal' block in {eval_file}")

    rows = set()
    for m in models:
        for effect, probes in data[m]["subliminal"].items():
            for probe in probes:
                rows.add((effect, probe))
    rows = sorted(rows)

    lines = ["| effect / probe | " + " | ".join(models) + " |",
             "| :--- |" + "".join([" ---: |"] * len(models))]
    for effect, probe in rows:
        row = [f"{effect} / {probe}"]
        for m in models:
            cell = data[m]["subliminal"].get(effect, {}).get(probe, {})
            freq = cell.get("target_frequency")
            row.append(f"{freq:.3f}" if freq is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines) + "\n"

    print(md)
    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        with open(out_md, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
