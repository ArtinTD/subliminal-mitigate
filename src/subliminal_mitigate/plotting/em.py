"""The final EM benefit/cost figure, independent of cluster directory layouts."""
import copy
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEXT = "#262626"
GRID = "#D7D7D7"
COST_RED = "#F08080"
COST_RED_LIGHT = "#F6B4B4"
COST_RED_DARK = "#B64C4C"
BENEFIT_GREEN = "#79C68A"
BENEFIT_GREEN_DARK = "#4A9460"
ABSTAIN_GRAY = "#B7B7B7"
ABSTAIN_GRAY_DARK = "#7A7A7A"

SINGLEPANEL_METHODS = [
    ("base", r"$\pi_0$"),
    ("union", r"$\pi_{\mathrm{union}}$"),
    ("merged", r"$\pi_{\mathrm{merge}}$"),
    ("kalai", "Kalai\net al."),
    ("min", r"$\pi_{\min}$"),
    ("delta", r"$\pi_{\Delta}$"),
]


def wilson_interval(hits, n, z=1.959963984540054):
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)



def rate(item):
    return item["hits"] / item["n"] if item["n"] else 0.0



def save_figure(fig, output_dir, prefix, audit):
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{prefix}.png"
    svg = output_dir / f"{prefix}.svg"
    data = output_dir / f"{prefix}.plot_data.json"
    fig.savefig(png, dpi=150, facecolor="white")
    fig.savefig(svg, facecolor="white")
    with open(data, "w") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    print(f"Wrote {data}")



def plot_fixed_same_prompt_singlepanel(
    rows,
    sources,
    output_dir,
    *,
    title=None,
    output_prefix="em5_benefit_cost_singlepanel",
):
    # Match the compact COLM core-result panels (Figure 3 in the July draft).
    # Unlike that draft, broad and narrow EM are adjacent bars rather than a
    # single stacked cost bar, so both rates remain individually visible.
    colm_style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
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
    with plt.rc_context(colm_style):
        # The large physical canvas is intentional: the COLM panel was saved
        # as a wide, high-resolution raster and then scaled down in LaTeX.
        fig, ax = plt.subplots(1, 1, figsize=(17.6, 7.4))
        ax.set_ylim(0, 1.06)
        ax.set_yticks([index / 5 for index in range(6)])
        ax.set_ylabel("Behavior rate", labelpad=16)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="x", length=0, pad=12)
        ax.tick_params(axis="y", length=0, pad=8)

        has_narrow_abstention = any(
            rows[method_key].get(
                "narrow_abstain", {"hits": 0, "n": 0}
            )["n"]
            > 0
            for method_key, *_rest in SINGLEPANEL_METHODS
        )
        broad_abstain_label = (
            "Broad abstain" if has_narrow_abstention else "Abstain"
        )
        metric_specs = [
            ("narrow_joke", "Shared benefit", BENEFIT_GREEN, BENEFIT_GREEN_DARK, None),
            ("broad", "Broad EM", COST_RED, COST_RED_DARK, None),
            ("narrow", "Bad medical", COST_RED, COST_RED_DARK, "///"),
            (
                "abstain",
                broad_abstain_label,
                ABSTAIN_GRAY,
                ABSTAIN_GRAY_DARK,
                None,
            ),
            (
                "narrow_abstain",
                "Narrow abstain",
                ABSTAIN_GRAY,
                ABSTAIN_GRAY_DARK,
                "///",
            ),
        ]
        group_centers = list(range(len(SINGLEPANEL_METHODS)))
        bar_width = 0.17 if has_narrow_abstention else 0.22
        bar_step = 0.19 if has_narrow_abstention else 0.24

        def add_colm_error_and_label(x, item):
            item_rate = rate(item)
            low, high = wilson_interval(item["hits"], item["n"])
            if 0.0 < item_rate < 1.0:
                ax.errorbar(
                    [x],
                    [item_rate],
                    yerr=[[item_rate - low], [high - item_rate]],
                    fmt="none",
                    ecolor="#1D1D1D",
                    elinewidth=2.6,
                    capsize=8,
                    capthick=2.6,
                    zorder=5,
                )
            label_y = min(
                1.035,
                max(item_rate, high) + 0.025
                if 0.0 < item_rate < 1.0
                else item_rate + 0.025,
            )
            ax.text(
                x,
                label_y,
                f"{item_rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=20,
            )

        for method_index, (method_key, *_rest) in enumerate(SINGLEPANEL_METHODS):
            available_specs = []
            for metric_spec in metric_specs:
                metric_key = metric_spec[0]
                item = rows[method_key].get(metric_key, {"hits": 0, "n": 0})
                if item["n"] > 0:
                    available_specs.append((metric_spec, item))

            offsets = [
                (index - (len(available_specs) - 1) / 2) * bar_step
                for index in range(len(available_specs))
            ]
            for offset, (metric_spec, item) in zip(offsets, available_specs):
                _metric_key, _label, color, edgecolor, hatch = metric_spec
                x = group_centers[method_index] + offset
                ax.bar(
                    x,
                    rate(item),
                    bar_width,
                    color=color,
                    edgecolor=edgecolor,
                    hatch=hatch,
                    linewidth=1.2,
                    zorder=3,
                )
                add_colm_error_and_label(x, item)

        labels = [label for _method_key, label, *_rest in SINGLEPANEL_METHODS]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        active_metric_keys = {
            metric_key
            for method_key, *_rest in SINGLEPANEL_METHODS
            for metric_key, *_spec_rest in metric_specs
            if rows[method_key].get(metric_key, {"hits": 0, "n": 0})["n"] > 0
        }
        legend_handles = [
            Patch(
                facecolor=color,
                edgecolor=edgecolor,
                hatch=hatch,
                label=label,
            )
            for metric_key, label, color, edgecolor, hatch in metric_specs
            if metric_key in active_metric_keys
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.99),
            ncol=len(legend_handles),
            frameon=True,
            borderpad=0.5,
            handlelength=1.1,
            columnspacing=1.2,
        )
        if title:
            ax.set_title(title, pad=10)
        fig.subplots_adjust(left=0.068, right=0.994, bottom=0.14, top=0.87)
        save_figure(
            fig,
            output_dir,
            output_prefix,
            {
                "rows": rows,
                "sources": sources,
                "intervals": "Wilson 95%",
                "style": "COLM core-result panel; broad and narrow cost bars unstacked",
                "bar_order": (
                    [
                        "shared_benefit_narrow_prompt",
                        "broad_em",
                        "bad_medical",
                        "broad_abstain",
                        "narrow_abstain",
                    ]
                    if has_narrow_abstention
                    else [
                        "shared_benefit_narrow_prompt",
                        "broad_em",
                        "bad_medical",
                        "abstain",
                    ]
                ),
                "kalai_behavior_denominator": (
                    "accepted outputs from the corresponding prompt bank"
                ),
                "kalai_abstention_denominator": (
                    "all 320 requested outputs in each prompt bank"
                    if has_narrow_abstention
                    else "all 320 requested broad-prompt outputs"
                ),
                "kalai_display": (
                    "complete broad/narrow metrics"
                    if has_narrow_abstention
                    else "broad abstention only"
                ),
            },
        )


def kalai_abstention_only_rows(rows):
    """Return the workshop-compatible view with only broad Kalai abstention."""
    filtered = copy.deepcopy(rows)
    for metric_key in (
        "broad",
        "narrow",
        "direct_joke",
        "broad_joke",
        "narrow_joke",
        "narrow_abstain",
    ):
        filtered["kalai"][metric_key] = {"hits": 0, "n": 0}
    return filtered



class EMFigure:
    """Own validated aggregate outcomes for the six displayed EM methods."""

    def __init__(self, rows):
        if set(rows) != {name for name, _ in SINGLEPANEL_METHODS}:
            raise ValueError("require Base, Union, merge, Kalai, min and delta rows")
        for method, metrics in rows.items():
            for name, item in metrics.items():
                if set(item) != {"hits", "n"} or any(type(item[k]) is not int for k in item):
                    raise ValueError(f"invalid count schema for {method}/{name}")
                if not 0 <= item["hits"] <= item["n"]:
                    raise ValueError(f"invalid counts for {method}/{name}")
        self.rows = copy.deepcopy(rows)

    def render(self, output_dir, *, complete_kalai=False):
        """Match the main-paper display, or explicitly opt into appendix coverage."""
        output_dir = Path(output_dir)
        rows = self.rows if complete_kalai else kalai_abstention_only_rows(self.rows)
        prefix = "em5_benefit_cost_singlepanel" + ("_complete_kalai" if complete_kalai else "")
        plot_fixed_same_prompt_singlepanel(rows, [], output_dir, output_prefix=prefix)
        return {"outputs": [str(output_dir / (prefix + suffix))
                            for suffix in (".png", ".svg", ".plot_data.json")]}
