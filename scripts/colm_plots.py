#!/usr/bin/env python3
"""Regenerate the quantitative result figures used by the COLM draft.

Usage:
    python -m pip install -r requirements-plots.txt
    python scripts/colm_plots.py

By default the script creates every figure whose source artifacts are
available. The quorum figure is skipped until its matched reference and
merged-LoRA JSON files have been copied back from Hyak. Pass --strict to make
missing inputs an error, or --plots to select individual figures.
"""

import argparse
import json
import math
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Run: "
        "python -m pip install -r requirements-plots.txt"
    ) from exc


REPO = Path(__file__).resolve().parents[1]
HYAK = REPO / "hyak_results" / "outputs"
DEFAULT_OUTPUT_DIR = REPO / "colm_2026" / "figures"
PLOT_DATA_PATH = REPO / "colm_2026" / "plot_data.json"

GREEN = "#79C58E"
GREEN_DARK = "#368654"
RED = "#F08080"
RED_DARK = "#B64C4C"
GRID = "#D7D7D7"
TEXT = "#262626"

FLEX_JOKE_RE = re.compile(
    r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S",
    re.IGNORECASE,
)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def wilson_interval(hits, n, z=1.959963984540054):
    if n <= 0:
        raise ValueError(f"Wilson interval requires n > 0, got {n}")
    p = hits / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.titlesize": 32,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#B7B7B7",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "hatch.linewidth": 1.5,
        }
    )


def style_axis(ax, ylabel):
    ax.set_ylim(0, 1.06)
    ax.set_ylabel(ylabel, labelpad=16)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=8)


def add_wilson_error(ax, x, hits, n, color="#1D1D1D"):
    rate = hits / n
    if rate <= 0.0 or rate >= 1.0:
        return
    low, high = wilson_interval(hits, n)
    ax.errorbar(
        [x],
        [rate],
        yerr=[[rate - low], [high - rate]],
        fmt="none",
        ecolor=color,
        elinewidth=2.6,
        capsize=8,
        capthick=2.6,
        zorder=5,
    )


def add_rate_label(ax, x, hits, n):
    rate = hits / n
    _, high = wilson_interval(hits, n)
    y = min(1.035, high + 0.025) if 0 < rate < 1 else rate + 0.025
    ax.text(x, y, f"{rate:.2f}", ha="center", va="bottom", fontsize=20)


def save_figure(fig, output_path, audit_payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), facecolor="white")
    with open(output_path.with_suffix(".plot_data.json"), "w") as handle:
        json.dump(audit_payload, handle, indent=2)
        handle.write("\n")
    plt.close(fig)
    print(f"Wrote {output_path}")


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def count_synthetic_samples(samples):
    return {
        "n": len(samples),
        "benefit_hits": sum(
            bool(FLEX_JOKE_RE.match(final_nonempty_line(sample.get("response", ""))))
            for sample in samples
        ),
        "eagle_hits": sum(
            first_nonempty_line(sample.get("response", "")).startswith("Eagle: ")
            for sample in samples
        ),
        "topaz_hits": sum(
            first_nonempty_line(sample.get("response", "")).startswith("Topaz: ")
            for sample in samples
        ),
    }


def load_synthetic_rows():
    root = HYAK / "composed_joke_explicit_cost"
    baseline_path = root / "joke_generation_samples_t512.json"
    min_path = (
        root
        / "min_composition"
        / "min_t512"
        / "full"
        / "min_composition_samples.json"
    )
    merged_path = (
        root
        / "min_composition"
        / "merged_lora"
        / "full"
        / "merged_lora_samples.json"
    )
    required = [baseline_path, min_path, merged_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    baseline = read_json(baseline_path)
    rows = []
    for label, key in [
        ("Base", "pi_base"),
        ("Reference A", "pi_A"),
        ("Reference B", "pi_B"),
        ("Benefit-only", "pi_benefit"),
    ]:
        row = count_synthetic_samples(baseline["models"][key]["samples"])
        row["label"] = label
        rows.append(row)

    for label, path in [("Consensus", min_path), ("Merged-LoRA", merged_path)]:
        row = count_synthetic_samples(read_json(path)["samples"])
        row["label"] = label
        rows.append(row)
    return rows, [str(path.relative_to(REPO)) for path in required]


def plot_synthetic(output_dir):
    rows, sources = load_synthetic_rows()
    fig, ax = plt.subplots(figsize=(15.8, 7.4))
    style_axis(ax, "Behavior Rate")
    fig.suptitle(
        "Exact consensus preserves the shared behavior while suppressing private behaviors",
        fontsize=30,
        y=0.975,
    )
    x = list(range(len(rows)))
    width = 0.37
    gap = 0.02

    for index, row in enumerate(rows):
        benefit_x = index - width / 2 - gap / 2
        cost_x = index + width / 2 + gap / 2
        n = row["n"]
        benefit = row["benefit_hits"] / n
        eagle = row["eagle_hits"] / n
        topaz = row["topaz_hits"] / n

        ax.bar(benefit_x, benefit, width, color=GREEN, zorder=3)
        ax.bar(cost_x, eagle, width, color=RED, zorder=3)
        ax.bar(
            cost_x,
            topaz,
            width,
            bottom=eagle,
            color=RED,
            edgecolor=RED_DARK,
            hatch="///",
            linewidth=0,
            zorder=3,
        )
        add_wilson_error(ax, benefit_x, row["benefit_hits"], n)
        add_wilson_error(ax, cost_x, row["eagle_hits"] + row["topaz_hits"], n)
        add_rate_label(ax, benefit_x, row["benefit_hits"], n)
        add_rate_label(ax, cost_x, row["eagle_hits"] + row["topaz_hits"], n)

    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows])
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="Shared benefit"),
            Patch(facecolor=RED, label="Eagle"),
            Patch(facecolor=RED, edgecolor=RED_DARK, hatch="///", label="Topaz"),
        ],
        loc="upper right",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.97, 0.91),
        borderpad=0.6,
        handlelength=1.1,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.15, top=0.80)
    save_figure(
        fig,
        output_dir / "synthetic_i_results.png",
        {"sources": sources, "methods": rows, "intervals": "Wilson 95%"},
    )


def location_parts(summary):
    any_hits = summary["joke_anywhere_hits"]
    first_hits = summary["joke_first_line_hits"]
    final_hits = summary["joke_final_line_hits"]
    both_hits = first_hits + final_hits - any_hits
    first_only = first_hits - both_hits
    final_only = final_hits - both_hits
    if min(first_only, final_only, both_hits) < 0:
        raise ValueError(f"Inconsistent temporal counts: {summary}")
    return first_only, final_only, both_hits


def temporal_row_from_artifact(spec):
    path = REPO / spec["source"]
    payload = read_json(path)
    meta = payload["meta"]
    summary = payload["summary"]
    samples = payload["samples"]
    n = summary["n_responses"]
    if n != 128:
        raise ValueError(f"{path}: matched temporal result has n={n}; expected 128")
    expected_meta = {
        "ref_names": spec["ref_names"],
        "composition_type": spec["composition_type"],
        "num_prompts": 32,
        "n_samples_per_prompt": 4,
        "temperature": 1.0,
        "seed": 0,
        "max_new_tokens": 512,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"{path}: metadata {key}={meta.get(key)!r}; expected {expected!r}"
            )
    if spec["composition_type"] == "merged_lora":
        params = meta["composition_params"]
        if params.get("weights") != spec["weights"]:
            raise ValueError(f"{path}: unexpected merge weights")
        if params.get("combination_type") != spec["combination_type"]:
            raise ValueError(f"{path}: unexpected merge combination type")
    if spec["composition_type"] == "lookback_min_gated":
        params = meta["composition_params"]
        if params.get("structural_exemption") != spec["lookback_structural_exemption"]:
            raise ValueError(f"{path}: unexpected lookback structural exemption")

    buckets = summary["joke_content_position_buckets"]
    eagle = summary["cost_rates"]["first_line_eagle"]
    topaz = summary["cost_rates"]["first_line_topaz"]
    overlap_hits = summary.get("anywhere_cost_overlap_hits")
    if overlap_hits is None:
        overlap_hits = sum(
            len(sample.get("anywhere_cost_hits", [])) > 1
            for sample in samples
        )
    return {
        "label": spec["label"],
        "source": spec["source"],
        "n": n,
        "first_only_hits": buckets.get("first_only", 0),
        "final_only_hits": buckets.get("final_only", 0),
        "both_hits": buckets.get("both_first_and_final", 0),
        "middle_only_hits": buckets.get("middle_only", 0),
        "no_joke_hits": buckets.get("no_joke", 0),
        "cost_n": n,
        "eagle_cost_hits": eagle.get("anywhere_hits", eagle["hits"]),
        "topaz_cost_hits": topaz.get("anywhere_hits", topaz["hits"]),
        "any_cost_hits": summary.get(
            "anywhere_cost_hits",
            sum(sample.get("has_anywhere_cost", False) for sample in samples),
        ),
        "cost_overlap_hits": overlap_hits,
    }


def load_temporal_rows(plot_data):
    matched_specs = plot_data.get("temporal", {}).get("matched_methods", [])
    matched_paths = [REPO / spec["source"] for spec in matched_specs]
    if matched_specs and all(path.exists() for path in matched_paths):
        rows = [temporal_row_from_artifact(spec) for spec in matched_specs]
        return rows, [row["source"] for row in rows], "matched"

    root = HYAK / "partial_agreement" / "joke_begin_end" / "pilot128"
    source_paths = [
        root / "generations" / "beginning" / "min.json",
        root / "generations" / "end" / "min.json",
        root / "generations" / "beginning_end" / "min.json",
    ]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    rows = []
    for label, path in zip(
        ["Beginning ref.", "End ref.", "Exact min"],
        source_paths,
    ):
        summary = read_json(path)["summary"]
        first_only, final_only, both = location_parts(summary)
        rows.append(
            {
                "label": label,
                "n": summary["n_responses"],
                "first_only_hits": first_only,
                "final_only_hits": final_only,
                "both_hits": both,
                "middle_only_hits": 0,
                "no_joke_hits": summary["n_responses"] - first_only - final_only - both,
                "cost_n": None,
            }
        )

    lookback = plot_data["temporal_lookback"]
    first_only, final_only, both = location_parts(lookback)
    rows.append(
        {
            "label": "Lookback",
            "n": lookback["n"],
            "first_only_hits": first_only,
            "final_only_hits": final_only,
            "both_hits": both,
            "middle_only_hits": 0,
            "no_joke_hits": lookback["n"] - first_only - final_only - both,
            "cost_n": None,
        }
    )
    sources = [str(path.relative_to(REPO)) for path in source_paths]
    sources.append(lookback["source"])
    return rows, sources, "legacy"


def plot_temporal(output_dir, plot_data):
    rows, sources, source_set = load_temporal_rows(plot_data)
    matched = source_set == "matched"
    fig, ax = plt.subplots(figsize=(17.2 if matched else 14, 7.4 if matched else 7.2))
    style_axis(ax, "Behavior Rate" if matched else "Joke Rate")
    fig.suptitle(
        (
            "Temporal lookback recovers shared behavior while suppressing private costs"
            if matched
            else "Temporal lookback recovers shared behavior across positions"
        ),
        fontsize=30,
        y=0.975,
    )
    width = 0.37 if matched else 0.46
    gap = 0.02
    stack_costs = matched and all(row["cost_overlap_hits"] == 0 for row in rows)
    for index, row in enumerate(rows):
        n = row["n"]
        first = row["first_only_hits"] / n
        final = row["final_only_hits"] / n
        both = row["both_hits"] / n
        total_hits = (
            row["first_only_hits"] + row["final_only_hits"] + row["both_hits"]
        )
        benefit_x = index - width / 2 - gap / 2 if matched else index
        cost_x = index + width / 2 + gap / 2
        ax.bar(benefit_x, first, width, color=GREEN, zorder=3)
        ax.bar(
            benefit_x,
            final,
            width,
            bottom=first,
            color="#B9DCBF",
            edgecolor=GREEN_DARK,
            hatch="///",
            linewidth=0,
            zorder=3,
        )
        ax.bar(
            benefit_x,
            both,
            width,
            bottom=first + final,
            color="#69AD78",
            edgecolor=GREEN_DARK,
            hatch="xxx",
            linewidth=0,
            zorder=3,
        )
        add_wilson_error(ax, benefit_x, total_hits, n)
        add_rate_label(ax, benefit_x, total_hits, n)
        if not matched:
            continue

        if stack_costs:
            eagle = row["eagle_cost_hits"] / n
            topaz = row["topaz_cost_hits"] / n
            ax.bar(cost_x, eagle, width, color=RED, zorder=3)
            ax.bar(
                cost_x,
                topaz,
                width,
                bottom=eagle,
                color="#F6B4B4",
                edgecolor=RED_DARK,
                hatch="///",
                linewidth=0,
                zorder=3,
            )
        else:
            ax.bar(cost_x, row["any_cost_hits"] / n, width, color=RED, zorder=3)
        add_wilson_error(ax, cost_x, row["any_cost_hits"], n)
        add_rate_label(ax, cost_x, row["any_cost_hits"], n)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows])
    legend_handles = [
            Patch(facecolor=GREEN, label="Early only" if matched else "First-line only"),
            Patch(
                facecolor="#B9DCBF",
                edgecolor=GREEN_DARK,
                hatch="///",
                label="Final only" if matched else "Final-line only",
            ),
            Patch(
                facecolor="#69AD78",
                edgecolor=GREEN_DARK,
                hatch="xxx",
                label="Both",
            ),
    ]
    if matched:
        if stack_costs:
            legend_handles.extend([
                Patch(facecolor=RED, label="Eagle cost"),
                Patch(
                    facecolor="#F6B4B4",
                    edgecolor=RED_DARK,
                    hatch="///",
                    label="Topaz cost",
                ),
            ])
        else:
            legend_handles.append(Patch(facecolor=RED, label="Any private cost"))
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=5 if matched and stack_costs else 3,
        frameon=True,
        bbox_to_anchor=(0.98, 0.91),
        borderpad=0.6,
        handlelength=1.1,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.075 if matched else 0.085, right=0.99, bottom=0.15, top=0.80)
    save_figure(
        fig,
        output_dir / "temporal_disagreement_results.png",
        {
            "sources": sources,
            "source_set": source_set,
            "methods": rows,
            "intervals": "Wilson 95%",
            "benefit_definition": (
                "content-position-aligned Joke: early/final/both"
                if matched
                else "literal Joke: first/final/both"
            ),
            "cost_definition": (
                "line-initial Eagle:/Topaz: anywhere" if matched else None
            ),
        },
    )


def validate_semantic_artifacts(rows):
    sources = []
    for row in rows:
        path = REPO / row["source"]
        if not path.exists():
            raise FileNotFoundError(path)
        summary = read_json(path)["summary"]
        observed = (
            summary["n_responses"],
            summary["final_joke_marker_hits"],
            summary["final_humor_marker_hits"],
        )
        wanted = (row["n"], row["joke_hits"], row["humor_hits"])
        if observed != wanted:
            raise ValueError(
                f"Semantic plot manifest disagrees with {path}: "
                f"{observed} != {wanted}"
            )

        if row["cost_n"] is not None:
            eagle = summary["cost_rates"]["first_line_eagle"]
            topaz = summary["cost_rates"]["first_line_topaz"]
            observed_costs = (
                summary["n_responses"],
                eagle.get("anywhere_hits", eagle["hits"]),
                topaz.get("anywhere_hits", topaz["hits"]),
            )
            wanted_costs = (
                row["cost_n"],
                row["eagle_cost_hits"],
                row["topaz_cost_hits"],
            )
            if observed_costs != wanted_costs:
                raise ValueError(
                    f"Semantic cost manifest disagrees with {path}: "
                    f"{observed_costs} != {wanted_costs}"
                )
        sources.append(row["source"])
    return sources


def semantic_row_from_artifact(spec):
    path = REPO / spec["source"]
    payload = read_json(path)
    meta = payload["meta"]
    summary = payload["summary"]
    samples = payload["samples"]
    n = summary["n_responses"]
    if n != 128:
        raise ValueError(f"{path}: matched semantic result has n={n}; expected 128")
    expected_meta = {
        "ref_names": spec["ref_names"],
        "composition_type": spec["composition_type"],
        "num_prompts": 32,
        "n_samples_per_prompt": 4,
        "temperature": 1.0,
        "seed": 0,
        "max_new_tokens": 512,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"{path}: metadata {key}={meta.get(key)!r}; expected {expected!r}"
            )
    if spec["composition_type"] == "merged_lora":
        params = meta["composition_params"]
        if params.get("weights") != spec["weights"]:
            raise ValueError(
                f"{path}: merge weights {params.get('weights')!r}; "
                f"expected {spec['weights']!r}"
            )
        if params.get("combination_type") != spec["combination_type"]:
            raise ValueError(
                f"{path}: merge type {params.get('combination_type')!r}; "
                f"expected {spec['combination_type']!r}"
            )

    eagle = summary["cost_rates"]["first_line_eagle"]
    topaz = summary["cost_rates"]["first_line_topaz"]
    overlap_hits = summary.get("anywhere_cost_overlap_hits")
    if overlap_hits is None:
        overlap_hits = sum(
            len(sample.get("anywhere_cost_hits", [])) > 1
            for sample in samples
        )
    return {
        "label": spec["label"],
        "source": spec["source"],
        "n": n,
        "joke_hits": summary["final_joke_marker_hits"],
        "humor_hits": summary["final_humor_marker_hits"],
        "cost_n": n,
        "eagle_cost_hits": eagle.get("anywhere_hits", eagle["hits"]),
        "topaz_cost_hits": topaz.get("anywhere_hits", topaz["hits"]),
        "any_cost_hits": summary.get(
            "anywhere_cost_hits",
            sum(sample.get("has_anywhere_cost", False) for sample in samples),
        ),
        "cost_overlap_hits": overlap_hits,
    }


def load_semantic_rows(plot_data):
    semantic = plot_data["semantic"]
    matched_specs = semantic.get("matched_methods", [])
    matched_paths = [REPO / spec["source"] for spec in matched_specs]
    if matched_specs and all(path.exists() for path in matched_paths):
        rows = [semantic_row_from_artifact(spec) for spec in matched_specs]
        return rows, [row["source"] for row in rows], "matched"

    rows = semantic["methods"]
    sources = validate_semantic_artifacts(rows)
    fallback_rows = []
    for row in rows:
        copied = dict(row)
        if copied["cost_n"] is not None:
            copied.setdefault(
                "any_cost_hits",
                copied["eagle_cost_hits"] + copied["topaz_cost_hits"],
            )
            copied.setdefault("cost_overlap_hits", 0)
        fallback_rows.append(copied)
    return fallback_rows, sources, "temporary"


def plot_semantic(output_dir, plot_data):
    rows, sources, source_set = load_semantic_rows(plot_data)
    stack_costs = all(
        row.get("cost_overlap_hits", 0) == 0
        for row in rows
        if row["cost_n"] is not None
    )
    fig, ax = plt.subplots(figsize=(17.2 if len(rows) == 5 else 15.8, 7.4))
    style_axis(ax, "Behavior Rate")
    fig.suptitle(
        "Semantic smoothing recovers shared behavior while suppressing private costs",
        fontsize=30,
        y=0.975,
    )
    width = 0.37
    gap = 0.02
    for index, row in enumerate(rows):
        n = row["n"]
        joke = row["joke_hits"] / n
        humor = row["humor_hits"] / n
        total_hits = row["joke_hits"] + row["humor_hits"]
        benefit_x = index - width / 2 - gap / 2
        cost_x = index + width / 2 + gap / 2
        ax.bar(benefit_x, joke, width, color=GREEN, zorder=3)
        ax.bar(
            benefit_x,
            humor,
            width,
            bottom=joke,
            color="#B9DCBF",
            edgecolor=GREEN_DARK,
            hatch="///",
            linewidth=0,
            zorder=3,
        )
        add_wilson_error(ax, benefit_x, total_hits, n)
        add_rate_label(ax, benefit_x, total_hits, n)

        cost_n = row["cost_n"]
        if cost_n is None:
            ax.text(
                cost_x,
                0.025,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=17,
                color="#666666",
            )
            continue

        if stack_costs:
            eagle = row["eagle_cost_hits"] / cost_n
            topaz = row["topaz_cost_hits"] / cost_n
            ax.bar(cost_x, eagle, width, color=RED, zorder=3)
            ax.bar(
                cost_x,
                topaz,
                width,
                bottom=eagle,
                color="#F6B4B4",
                edgecolor=RED_DARK,
                hatch="///",
                linewidth=0,
                zorder=3,
            )
        else:
            ax.bar(
                cost_x,
                row["any_cost_hits"] / cost_n,
                width,
                color=RED,
                zorder=3,
            )
        total_cost_hits = row["any_cost_hits"]
        add_wilson_error(ax, cost_x, total_cost_hits, cost_n)
        add_rate_label(ax, cost_x, total_cost_hits, cost_n)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows])
    legend_handles = [
            Patch(facecolor=GREEN, label="Joke:"),
            Patch(
                facecolor="#B9DCBF",
                edgecolor=GREEN_DARK,
                hatch="///",
                label="Humor:",
            ),
    ]
    if stack_costs:
        legend_handles.extend([
            Patch(facecolor=RED, label="Eagle cost"),
            Patch(
                facecolor="#F6B4B4",
                edgecolor=RED_DARK,
                hatch="///",
                label="Topaz cost",
            ),
        ])
    else:
        legend_handles.append(Patch(facecolor=RED, label="Any private cost"))
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=len(legend_handles),
        frameon=True,
        bbox_to_anchor=(0.97, 0.91),
        borderpad=0.6,
        handlelength=1.1,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.80)
    save_figure(
        fig,
        output_dir / "semantic_disagreement_results.png",
        {
            "source_set": source_set,
            "sources": sources,
            "methods": rows,
            "cost_display": "stacked_by_prefix" if stack_costs else "any_cost",
            "intervals": "Wilson 95%",
        },
    )


def extract_quorum_method(label, path):
    payload = read_json(path)
    summary = payload["summary"]
    n = summary["n_responses"]
    if n != 128:
        raise ValueError(f"{label} has n={n}; expected 128")
    costs = {
        name: summary["cost_rates"][f"first_line_{name}"]["hits"]
        for name in ("eagle", "topaz", "birch", "cobalt")
    }
    if sum(costs.values()) != summary["any_cost_hits"]:
        raise ValueError(f"{label}: cost segments do not sum to any-cost")
    return {
        "label": label,
        "source": str(path.relative_to(REPO)),
        "n": n,
        "benefit_hits": summary["joke_flex_last_hits"],
        "cost_hits": costs,
        "any_cost_hits": summary["any_cost_hits"],
        "truncation_hits": summary.get(
            "truncation_hits",
            summary.get("stop_reasons", {}).get("max_new_tokens", 0),
        ),
    }


def load_quorum_rows(quorum_root):
    paths = [
        ("Eagle", quorum_root / "references" / "eagle.json"),
        ("Topaz", quorum_root / "references" / "topaz.json"),
        ("Birch", quorum_root / "references" / "birch.json"),
        ("Cobalt-only", quorum_root / "references" / "cobalt_cost_only.json"),
        ("Merged LoRA", quorum_root / "merged_lora" / "all_four_equal.json"),
        (
            "Exact min",
            quorum_root
            / "generations"
            / "m4_three_benefit_one_dropout"
            / "min.json",
        ),
        (
            "Quorum q=3",
            quorum_root
            / "generations"
            / "m4_three_benefit_one_dropout"
            / "quorum_q3.json",
        ),
    ]
    missing = [path for _, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    return [extract_quorum_method(label, path) for label, path in paths]


def plot_quorum(output_dir, quorum_root):
    rows = load_quorum_rows(quorum_root)
    fig, ax = plt.subplots(figsize=(15.8, 7.4))
    style_axis(ax, "Observed Rate")
    fig.suptitle(
        "Quorum recovers shared behavior under incomplete support",
        fontsize=30,
        y=0.975,
    )
    width = 0.31
    gap = 0.025
    hatches = {
        "eagle": None,
        "topaz": "///",
        "birch": "xxx",
        "cobalt": "...",
    }

    for index, row in enumerate(rows):
        benefit_x = index - width / 2 - gap / 2
        cost_x = index + width / 2 + gap / 2
        n = row["n"]
        ax.bar(
            benefit_x,
            row["benefit_hits"] / n,
            width,
            color=GREEN,
            zorder=3,
        )
        bottom = 0.0
        for cost_name in ("eagle", "topaz", "birch", "cobalt"):
            height = row["cost_hits"][cost_name] / n
            ax.bar(
                cost_x,
                height,
                width,
                bottom=bottom,
                color=RED,
                edgecolor=RED_DARK if hatches[cost_name] else RED,
                hatch=hatches[cost_name],
                linewidth=0,
                zorder=3,
            )
            bottom += height
        add_wilson_error(ax, benefit_x, row["benefit_hits"], n)
        add_wilson_error(ax, cost_x, row["any_cost_hits"], n)
        add_rate_label(ax, benefit_x, row["benefit_hits"], n)
        add_rate_label(ax, cost_x, row["any_cost_hits"], n)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows])
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="Final joke"),
            Patch(facecolor=RED, label="Eagle cost"),
            Patch(facecolor=RED, edgecolor=RED_DARK, hatch="///", label="Topaz cost"),
            Patch(facecolor=RED, edgecolor=RED_DARK, hatch="xxx", label="Birch cost"),
            Patch(facecolor=RED, edgecolor=RED_DARK, hatch="...", label="Cobalt cost"),
        ],
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.57, 0.91),
        borderpad=0.6,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.15, top=0.80)
    save_figure(
        fig,
        output_dir / "quorum_dropout_results.png",
        {
            "sources": [row["source"] for row in rows],
            "methods": rows,
            "intervals": "Wilson 95%",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["synthetic", "temporal", "semantic", "quorum"],
        default=["synthetic", "temporal", "semantic", "quorum"],
        help="Figures to generate (default: all available).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--quorum-root",
        type=Path,
        default=(
            HYAK
            / "composed_joke_explicit_cost"
            / "quorum_dropout_m4"
            / "pilot128"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping a figure with missing source artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    plot_data = read_json(PLOT_DATA_PATH)
    generators = {
        "synthetic": lambda: plot_synthetic(args.output_dir),
        "temporal": lambda: plot_temporal(args.output_dir, plot_data),
        "semantic": lambda: plot_semantic(args.output_dir, plot_data),
        "quorum": lambda: plot_quorum(args.output_dir, args.quorum_root),
    }
    generated = []
    skipped = []
    for name in args.plots:
        try:
            generators[name]()
            generated.append(name)
        except FileNotFoundError as exc:
            if args.strict:
                raise
            skipped.append((name, str(exc)))
            print(f"Skipped {name}: missing source artifact(s): {exc}")
    if not generated:
        raise SystemExit("No figures were generated")
    if skipped:
        print("Skipped figures can be regenerated after their Hyak JSONs are copied locally.")


if __name__ == "__main__":
    main()
