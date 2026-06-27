#!/usr/bin/env python
"""Plot per-reference current J/EOS probabilities from replay audit JSON.

This consumes the JSON written by scripts/audit_newline_exempt_lookback.py in
replay mode. It intentionally only reads already-computed replay probabilities;
it does not load models or change any decoding behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


def flatten_records(payload):
    result = payload.get("result", {})
    if "records" in result:
        yield None, result["records"]
        return
    for item in result.get("generation_files", []):
        yield item.get("path"), item.get("records", [])


def safe_name(text):
    keep = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "record"


def get_prob(step, ref_name, token_id):
    value = step.get("refs", {}).get(ref_name, {}).get("current", {}).get(str(token_id))
    if value is None:
        return math.nan
    return float(value)


def plot_values(values, y_scale):
    if y_scale != "log":
        return values
    return [max(value, 1e-12) if math.isfinite(value) else math.nan for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--j_token_id", type=int, default=41)
    parser.add_argument("--eos_token_id", type=int, default=151645)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--y_scale", choices=["linear", "log"], default="linear")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.replay_json) as f:
        payload = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    written = 0

    for generation_path, records in flatten_records(payload):
        generation_label = (
            Path(generation_path).stem if generation_path else Path(args.replay_json).stem
        )
        for record in records:
            if args.max_records is not None and written >= args.max_records:
                break
            steps = record.get("steps", [])
            if not steps:
                continue
            ref_names = list(steps[0].get("refs", {}).keys())
            if len(ref_names) < 2:
                continue

            xs = [int(step["step_index"]) for step in steps]
            fig, ax = plt.subplots(figsize=(9.5, 4.8))
            colors = ["#2b8cbe", "#e34a33"]
            for ref_index, ref_name in enumerate(ref_names[:2]):
                color = colors[ref_index % len(colors)]
                j_probs = [get_prob(step, ref_name, args.j_token_id) for step in steps]
                eos_probs = [get_prob(step, ref_name, args.eos_token_id) for step in steps]
                ax.plot(
                    xs,
                    plot_values(j_probs, args.y_scale),
                    color=color,
                    linewidth=2.0,
                    label=f"{ref_name} P(J)",
                )
                ax.plot(
                    xs,
                    plot_values(eos_probs, args.y_scale),
                    color=color,
                    linewidth=1.8,
                    linestyle="--",
                    label=f"{ref_name} P(EOS)",
                )

            event_xs = []
            event_labels = []
            for step in steps:
                point_types = step.get("point_types", [])
                if any(name in point_types for name in ("first_joke", "eos")):
                    event_xs.append(int(step["step_index"]))
                    event_labels.append(",".join(point_types))
            for x in event_xs:
                ax.axvline(x, color="#777777", alpha=0.25, linewidth=1)

            prompt_index = record.get("prompt_index", "?")
            sample_index = record.get("sample_index", "?")
            global_index = record.get("global_sample_index", written)
            ax.set_title(
                f"{generation_label}: global {global_index}, prompt {prompt_index}, sample {sample_index}"
            )
            ax.set_xlabel("Generated step")
            ax.set_ylabel("Current reference probability")
            if args.y_scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-12, top=1.02)
            else:
                ax.set_ylim(bottom=0.0, top=1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()

            filename = (
                f"{safe_name(generation_label)}__global_{global_index}"
                f"_prompt_{prompt_index}_sample_{sample_index}.png"
            )
            path = out_dir / filename
            fig.savefig(path, dpi=args.dpi)
            plt.close(fig)

            index_rows.append({
                "plot": str(path),
                "generation": generation_label,
                "generation_path": generation_path or "",
                "global_sample_index": global_index,
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "stop_reason": record.get("stop_reason", ""),
                "has_joke": record.get("has_joke", ""),
                "has_eagle": record.get("has_eagle", ""),
                "has_topaz": record.get("has_topaz", ""),
                "prompt": record.get("prompt", ""),
                "response": record.get("response", ""),
            })
            written += 1
        if args.max_records is not None and written >= args.max_records:
            break

    index_path = out_dir / "plot_index.csv"
    with open(index_path, "w", newline="") as f:
        fieldnames = [
            "plot",
            "generation",
            "generation_path",
            "global_sample_index",
            "prompt_index",
            "sample_index",
            "stop_reason",
            "has_joke",
            "has_eagle",
            "has_topaz",
            "prompt",
            "response",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Wrote {written} plots to {out_dir}")
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()
