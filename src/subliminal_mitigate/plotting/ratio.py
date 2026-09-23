"""Plot the locally sealed ratio-matched analysis; no network or experiment calls.

The three tradeoff facets show marginal absolute endpoint intervals, not
shifted paired-gain intervals or simultaneous confidence regions. Kalai's
task and medical abstention rates are annotated beside its points, without
a separate coverage panel. Benefit is conditional task accuracy times task
coverage, i.e. correct/all task requests; medical coverage is not applied to
task accuracy. Estimates are never jittered.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D


TEXT = "#262626"
GRID = "#D7D7D7"
METHOD_COLOR = "#4C9F70"
BASELINE_COLOR = "#A06FAD"
REFERENCE_COLOR = "#4C78A8"
SPINE = "#A8A8A8"
GRAY = "#777777"
WHITE = "#FFFFFF"
LABEL_LEADER = "#000000"
PANELS = ("one_bad_three_benign", "two_bad_two_benign", "three_bad_one_benign")
METHODS = (
    "ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4",
    "lora_merge", "weighted_union_sft", "kalai_s1_r20",
)
LABELS = {
    "ordinary_quorum_m4_q3": r"$\Phi_3$",
    "ordinary_min_m4_q4": r"$\pi_{\min}$",
    "delta_min_m4_q4": r"$\pi_{\Delta}$",
    "lora_merge": "LoRA\nmerge",
    "weighted_union_sft": "Weighted\nUnion",
    "kalai_s1_r20": r"Kalai $s=1$",
}
TITLES = ("(a) 1 bad : 3 benign", "(b) 2 bad : 2 benign", "(c) 3 bad : 1 benign")
PANEL_REFERENCES = {
    PANELS[0]: ("A1", "B1", "B2", "B3"),
    PANELS[1]: ("A1", "A2", "B1", "B2"),
    PANELS[2]: ("A1", "A2", "A3", "B1"),
}


def configure_style():
    # Same category colors, marker shapes, typeface, and grid as the paper's
    # collaborator-aligned benefit-cost planes; sizes suit a full-width figure.
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10.5,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": SPINE,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linestyle=(0, (4, 3)), linewidth=0.85, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(length=0, pad=5)


def validate_statistic(statistic, *, require_ci=True):
    estimate = statistic["estimate"]
    if not isinstance(estimate, (int, float)) or not 0 <= estimate <= 1:
        raise ValueError("invalid coordinate estimate")
    interval = statistic["ci95"]
    if interval is None:
        if require_ci:
            raise ValueError("plot requires marginal coordinate intervals")
        return
    lo, hi = interval
    if not 0 <= lo <= estimate <= hi <= 1:
        raise ValueError("invalid coordinate interval")


def coverage_adjusted_benefit(massive):
    """Apply same-task coverage once, never medical coverage or twice."""
    requested = massive["requested_n"]
    accepted = massive["accepted_n"]
    correct = massive["intent_correct_n"]
    if not 0 <= correct <= accepted <= requested or requested <= 0:
        raise ValueError("invalid task correct/accepted/requested counts")
    if massive["abstained_n"] != requested - accepted:
        raise ValueError("task abstention differs from accepted count")
    conditional = massive["conditional_intent_accuracy"]["estimate"]
    if accepted:
        if conditional is None or abs(conditional - correct / accepted) > 1e-12:
            raise ValueError("conditional task accuracy differs from counts")
        benefit = conditional * (accepted / requested)
    else:
        benefit = 0.0
    if abs(benefit - massive["intent_accuracy"]["estimate"]) > 1e-12:
        raise ValueError("task benefit already accounts for abstention exactly once")
    return benefit


def abstention_annotation(row):
    massive, medical = row["massive"], row["medical"]
    return (LABELS["kalai_s1_r20"] +
            f"\nTask abst.: {100 * massive['abstained_n'] / massive['requested_n']:.2f}%" +
            f"\nMed. abst.: {100 * medical['abstained_n'] / medical['requested_n']:.2f}%")


def exact_binomial_quantile(n, probability, quantile):
    """Evaluate an empirical binary-bootstrap quantile without inventing rows."""
    if n <= 0 or not 0 <= probability <= 1 or not 0 < quantile < 1:
        raise ValueError("invalid binomial bootstrap arguments")
    if probability == 0:
        return 0
    if probability == 1:
        return n
    probabilities = [math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(probability) + (n - k) * math.log1p(-probability))
                    for k in range(n + 1)]
    target = quantile * math.fsum(probabilities)
    cumulative = 0.0
    for k, value in enumerate(probabilities):
        cumulative += value
        if cumulative >= target:
            return k
    return n


def exact_binary_bootstrap_interval(correct, requested):
    if not 0 <= correct <= requested or requested <= 0:
        raise ValueError("invalid binary outcome counts")
    return [exact_binomial_quantile(requested, correct / requested, q) / requested
            for q in (.025, .975)]


def prompt_cluster_interval(counts, bootstrap):
    if len(counts) != 16 or any(type(value) is not int or not 0 <= value <= 5 for value in counts):
        raise ValueError("reference requires 16 whole-prompt BAD counts")
    generator = random.Random(bootstrap["seed"])
    draws = sorted(sum(counts[generator.randrange(16)] for _ in range(16)) / 80
                   for _ in range(bootstrap["replicates"]))
    bounds = []
    for q in (.025, .975):
        position = (len(draws) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        weight = position - lo
        bounds.append(draws[lo] * (1 - weight) + draws[hi] * weight)
    return bounds


def reference_coordinates(data, reference_data):
    body = {key: value for key, value in reference_data.items() if key != "payload_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != reference_data["payload_sha256"]:
        raise ValueError("reference observation seal differs")
    if reference_data["observation_payload_sha256"] != data["observation_payload_sha256"]:
        raise ValueError("reference data bind a different ratio analysis")
    if reference_data["question_ids"] != [f"medical_official16_{i:02d}" for i in range(16)]:
        raise ValueError("reference prompt bank differs")
    coordinates = {}
    for reference in reference_data["references"]:
        identifier = reference["reference_id"]
        if identifier not in ("A1", "B1", "B2", "B3") or identifier in coordinates:
            raise ValueError("unexpected or duplicate historical reference")
        if reference["massive_requested_n"] != 360 or reference["medical_requested_n"] != 80:
            raise ValueError("reference denominators differ")
        if sum(reference["prompt_bad_counts"]) != reference["bad_n"]:
            raise ValueError("reference prompt counts differ from total")
        coordinates[identifier] = {
            "reference_id": identifier, "x": reference["intent_correct_n"] / 360,
            "y": reference["bad_n"] / 80,
            "x_ci95": exact_binary_bootstrap_interval(reference["intent_correct_n"], 360),
            "y_ci95": prompt_cluster_interval(reference["prompt_bad_counts"], data["bootstrap"]),
            "x_ci_method": "exact marginal empirical binary row-bootstrap percentile distribution; counts only, no paired contrasts",
            "y_ci_method": "10,000 common-seed whole-prompt percentile-bootstrap draws, 16x5",
            "comparison_scope": reference["scope"],
            "intent_correct_n": reference["intent_correct_n"], "medical_bad_n": reference["bad_n"],
        }
    if set(coordinates) != {"A1", "B1", "B2", "B3"}:
        raise ValueError("original four reference coordinates are required")
    for identifier in ("A2", "A3"):
        rows = [row for row in data["arms"] if row["method_id"] == "direct_" + identifier]
        if len(rows) != 1:
            raise ValueError("changed-panel direct reference is missing or duplicated")
        row = rows[0]
        if row["massive"]["requested_n"] != 360 or row["medical"]["requested_n"] != 80:
            raise ValueError("changed-panel reference denominators differ")
        for statistic in (row["massive"]["intent_accuracy"], row["medical"]["bad_rate"]):
            validate_statistic(statistic)
        coordinates[identifier] = {
            "reference_id": identifier, "x": coverage_adjusted_benefit(row["massive"]),
            "y": row["medical"]["bad_rate"]["estimate"],
            "x_ci95": row["massive"]["intent_accuracy"]["ci95"],
            "y_ci95": row["medical"]["bad_rate"]["ci95"],
            "x_ci_method": "10,000 aligned row-percentile bootstrap draws",
            "y_ci_method": "10,000 common-seed whole-prompt percentile-bootstrap draws, 16x5",
            "comparison_scope": "recorded_direct_reference_evaluation",
            "intent_correct_n": row["massive"]["intent_correct_n"], "medical_bad_n": row["medical"]["bad_n"],
        }
    base = reference_data["contextual_base"]
    if (base["intent_correct_n"] != data["paired_base"]["intent_correct_n"] or
            base["massive_requested_n"] != 360 or base["medical_requested_n"] != 80 or base["bad_n"] != 1):
        raise ValueError("contextual base counts differ")
    # One BAD event across sixteen equal-size clusters has a known bootstrap
    # distribution regardless of its unknown cluster identity. Do not invent it.
    base_coordinate = {
        "reference_id": "base", "x": data["paired_base"]["intent_accuracy"]["estimate"],
        "y": base["bad_n"] / 80,
        "x_ci95": data["paired_base"]["intent_accuracy"]["ci95"],
        "y_ci95": [exact_binomial_quantile(16, 1 / 16, q) / 80 for q in (.025, .975)],
        "x_ci_method": "current paired-base marginal row-percentile bootstrap",
        "y_ci_method": "exact one-BAD-event prompt-cluster bootstrap distribution; cluster identity not imputed",
        "comparison_scope": base["scope"], "source_note": base["source_note"],
        "intent_correct_n": base["intent_correct_n"], "medical_bad_n": base["bad_n"],
    }
    return coordinates, base_coordinate


def validate(data):
    body = {key: value for key, value in data.items() if key != "payload_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != data["payload_sha256"]:
        raise ValueError("analysis result seal differs")
    if data["bootstrap"]["coordinate_intervals"] != "absolute marginal endpoint intervals, NOT shifted paired-gain intervals":
        raise ValueError("plot requires absolute marginal endpoint intervals")
    rows = {}
    for row in data["arms"]:
        if row["method_id"] not in METHODS:
            continue
        key = (row["panel_id"], row["method_id"])
        if row["panel_id"] not in PANELS or key in rows:
            raise ValueError("unknown or duplicate panel/method")
        if row["massive"]["requested_n"] != 360 or row["medical"]["requested_n"] != 80:
            raise ValueError("coordinate denominators differ")
        for statistic in (row["massive"]["intent_accuracy"], row["medical"]["bad_rate"], row["medical"]["coverage"]):
            validate_statistic(statistic)
        coverage_adjusted_benefit(row["massive"])
        medical = row["medical"]
        if medical["accepted_n"] + medical["abstained_n"] != 80:
            raise ValueError("coverage does not retain all requested medical cells")
        if abs(medical["coverage"]["estimate"] - medical["accepted_n"] / 80) > 1e-12:
            raise ValueError("medical coverage differs from accepted count")
        if row["method_id"] != "kalai_s1_r20" and medical["accepted_n"] != 80:
            raise ValueError("non-Kalai coverage annotation would be inaccurate")
        rows[key] = row
    required = {(panel, method) for panel in PANELS for method in METHODS[:3]}
    if not required <= set(rows):
        raise ValueError("plot requires all nine composition points")
    return rows


def draw_intervals(ax, x, y, x_interval, y_interval, color, marker, size=70):
    xlo, xhi = x_interval
    ylo, yhi = y_interval
    ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                fmt="none", ecolor=color, elinewidth=1.05, capsize=3.2,
                capthick=1.05, alpha=0.85, zorder=3)
    ax.scatter(x, y, marker=marker, s=size, facecolor=color, edgecolor=WHITE,
               linewidth=0.7, zorder=5)


def method_label_offsets(panel, include_references):
    # Offsets are in physical points, so labels stay close to their own marker
    # across facets. Only text moves; estimates and interval endpoints do not.
    offsets = {
        PANELS[0]: {METHODS[0]: (17, -2, "left", "top"),
                    METHODS[1]: (17, 19, "left", "bottom"),
                    METHODS[2]: (20, 13, "left", "center"),
                    "lora_merge": (-10, 16, "right", "bottom"),
                    "weighted_union_sft": (-12, 13, "right", "bottom"),
                    "kalai_s1_r20": (-3, 12, "right", "bottom")},
        PANELS[1]: {METHODS[0]: (-18, 4, "right", "center"),
                    METHODS[1]: (18, 17, "left", "bottom"),
                    METHODS[2]: (20, 9, "left", "center"),
                    "lora_merge": (18, 17, "left", "bottom"),
                    "weighted_union_sft": (-16, 16, "right", "bottom"),
                    "kalai_s1_r20": (-3, 12, "right", "bottom")},
        PANELS[2]: {METHODS[0]: (21, -5, "left", "top"),
                    METHODS[1]: (19, 8, "left", "bottom"),
                    METHODS[2]: (20, 10, "left", "center"),
                    "lora_merge": (-17, -17, "right", "top"),
                    "weighted_union_sft": (-19, 5, "right", "center"),
                    "kalai_s1_r20": (-3, 12, "right", "bottom")},
    }[panel].copy()
    if not include_references and panel == PANELS[0]:
        offsets["lora_merge"] = (-12, 10, "right", "bottom")
    return offsets


def reference_label_offsets(panel):
    return {
        PANELS[0]: {"A1": (10, 12, "left", "bottom"),
                    "B1": (-11, 5, "right", "bottom"),
                    "B2": (18, 15, "left", "bottom"),
                    "B3": (-11, -5, "right", "top")},
        PANELS[1]: {"A1": (-11, 8, "right", "bottom"),
                    "A2": (9, 12, "left", "bottom"),
                    "B1": (-11, -2, "right", "center"),
                    "B2": (-17, 23, "right", "bottom")},
        PANELS[2]: {"A1": (-14, 4, "right", "center"),
                    "A2": (-12, 14, "right", "bottom"),
                    "A3": (15, 13, "left", "bottom"),
                    "B1": (-13, 12, "right", "bottom")},
    }[panel]


def annotate_point(ax, text, x, y, offset, *, fontsize):
    dx, dy, ha, va = offset
    return ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va=va, fontsize=fontsize, zorder=7,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": .94, "pad": .35},
        arrowprops={"arrowstyle": "-", "color": LABEL_LEADER, "lw": .7,
                    "shrinkA": 2, "shrinkB": 5},
    )


class RatioFigure:
    """Render the final three-panel figure from validated analysis objects."""

    def __init__(self, data, reference_data=None):
        self.data = data
        self.reference_data = reference_data
        validate(data)
        if reference_data is not None:
            reference_coordinates(data, reference_data)

    def render(self, output_dir):
        """Write figure formats and coordinate metadata; never change estimates."""
        data, reference_data = self.data, self.reference_data
        output_dir = Path(output_dir)
        rows = validate(data)
        references, base_coordinate = reference_coordinates(data, reference_data) if reference_data else ({}, None)
        configure_style()
        fig = plt.figure(figsize=(10.9, 5.4 if references else 4.6))
        grid = fig.add_gridspec(1, 3, left=0.083, right=0.985,
                              bottom=0.22, top=0.92, wspace=0.15)
        axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
        base_accuracy = data["paired_base"]["intent_accuracy"]["estimate"]
        lowest_ci = min(row["massive"]["intent_accuracy"]["ci95"][0] for row in rows.values())
        xmin = min(0.65, lowest_ci - 0.015, base_accuracy - 0.012)
        ymax = max(0.65, max(row["medical"]["bad_rate"]["ci95"][1] for row in rows.values()) + 0.045)
        if references:
            xmin = min(xmin, base_coordinate["x_ci95"][0] - .025)
            ymax = max(.80, max(row["y_ci95"][1] for row in references.values()) + .045)
        plot_rows = []
        reference_rows, base_rows = [], []
        missing = []
        for ax, panel, title in zip(axes, PANELS, TITLES):
            style_axis(ax)
            ax.set_xlim(xmin, 0.95)
            ax.set_ylim(-0.025, ymax)
            ax.set_xticks([value for value in ((.60, .70, .80, .90) if references else (.55, .65, .75, .85, .95)) if value >= xmin])
            ax.set_yticks([0, .20, .40, .60, .80] if references else [0, .15, .30, .45, .60])
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
            ax.set_title(title, loc="left", fontsize=12.6, pad=9)
            if references:
                for identifier in PANEL_REFERENCES[panel]:
                    point = references[identifier]
                    draw_intervals(ax, point["x"], point["y"], point["x_ci95"], point["y_ci95"], REFERENCE_COLOR, "^", size=65)
                    offset = reference_label_offsets(panel)[identifier]
                    label = "$" + identifier[0] + "_{" + identifier[1] + "}$"
                    annotate_point(ax, label, point["x"], point["y"], offset, fontsize=10.2)
                    reference_rows.append({**point, "panel_id": panel, "marker": "^", "fill": REFERENCE_COLOR,
                                           "panel_member": True, "x_denominator": 360, "y_denominator": 80,
                                           "label_offset_points": list(offset[:2])})
                point = base_coordinate
                draw_intervals(ax, point["x"], point["y"], point["x_ci95"], point["y_ci95"], GRAY, "D", size=55)
                annotate_point(ax, "Base", point["x"], point["y"],
                               (-8, 3, "right", "center"), fontsize=9.4)
                base_rows.append({**point, "panel_id": panel, "marker": "D", "fill": GRAY,
                                  "panel_member": False, "x_denominator": 360, "y_denominator": 80})
            for method in METHODS:
                if (panel, method) not in rows:
                    missing.append({"panel_id": panel, "method_id": method})
                    continue
                row = rows[(panel, method)]
                x = coverage_adjusted_benefit(row["massive"])
                y = row["medical"]["bad_rate"]["estimate"]
                x_interval = row["massive"]["intent_accuracy"]["ci95"]
                y_interval = row["medical"]["bad_rate"]["ci95"]
                composition = row["composition_method"]
                color, marker = (METHOD_COLOR, "o") if composition else (BASELINE_COLOR, "s")
                draw_intervals(ax, x, y, x_interval, y_interval, color, marker)
                offset = method_label_offsets(panel, bool(references))[method]
                point_label = "Balanced\nUnion*" if method == "weighted_union_sft" and row.get("comparison_scope") == "historical_contextual" else LABELS[method]
                if method == "kalai_s1_r20":
                    point_label = abstention_annotation(row)
                annotate_point(ax, point_label, x, y, offset,
                               fontsize=8.7 if method == "kalai_s1_r20" else (10.0 if references else 9.4))
                plot_rows.append({
                    "panel_id": panel, "method_id": method, "x": x, "y": y,
                    "x_ci95": x_interval, "y_ci95": y_interval,
                    "x_denominator": 360, "y_denominator": 80,
                    "marker": marker, "fill": color,
                    "massive_accepted_n": row["massive"]["accepted_n"],
                    "massive_abstained_n": row["massive"]["abstained_n"],
                    "massive_abstention_rate": row["massive"]["abstained_n"] / 360,
                    "medical_accepted_n": row["medical"]["accepted_n"],
                    "medical_abstained_n": row["medical"]["abstained_n"],
                    "medical_abstention_rate": row["medical"]["abstained_n"] / 80,
                    "annotation": point_label,
                    "label_offset_points": list(offset[:2]),
                    "comparison_scope": row.get("comparison_scope"),
                })
        axes[0].set_ylabel(r"Medical $\mathrm{BAD}$ / requested (lower is safer)", fontsize=11.1)
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        fig.text(0.535, 0.132, "Coverage-adjusted MASSIVE intent accuracy (higher is better)",
                 ha="center", va="center", fontsize=11.6)
        handles = [
            Line2D([], [], marker="s", markersize=7, color=BASELINE_COLOR, linestyle="none", label="Baselines"),
            Line2D([], [], marker="o", markersize=7, color=METHOD_COLOR, linestyle="none", label="Our methods"),
        ]
        if references:
            handles.insert(0, Line2D([], [], marker="^", markersize=7, color=REFERENCE_COLOR, linestyle="none", label="References"))
            handles.append(Line2D([], [], marker="D", markersize=6, color=GRAY, linestyle="none", label="Base"))
        legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.535, 0.028), ncol=len(handles),
                            frameon=True, borderpad=0.45, handletextpad=0.55, columnspacing=1.6)
        legend.get_frame().set_facecolor("#F7F7F7")
        legend.get_frame().set_edgecolor("#B8B8B8")
        legend.get_frame().set_linewidth(0.7)
        # Explanatory notes belong in the paper caption, not below the legend.
        # Retain them in the audit metadata so removing artwork does not erase scope.
        caption_notes = [
            "Both axes use all requests; abstentions contribute neither benefit nor BAD. Only Kalai abstains.",
            "Whiskers show marginal 95% bootstrap intervals.",
        ]
        if any(row["method_id"] == "weighted_union_sft" and row.get("comparison_scope") == "historical_contextual" for row in rows.values()):
            caption_notes.append("* Balanced Union in panel (b) is trained once on each unique bad and benign dataset; it is not a separately trained four-source weighted-Union arm.")
        if missing:
            names = "; ".join(f"{TITLES[PANELS.index(item['panel_id'])][4:]}: {LABELS[item['method_id']]}" for item in missing)
            fig.text(0.535, 0.96, "Unavailable comparison arms: " + names, ha="center", va="center", fontsize=9, color=GRAY)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the previous figure with its separate coverage panel.
        stem = ("massive_medical_ratio_panels_v2_tradeoff_with_references" if references else
                "massive_medical_ratio_panels_v2_tradeoff_abstention_annotated")
        paths = {extension: output_dir / (stem + "." + extension) for extension in ("pdf", "png", "svg")}
        fig.savefig(paths["pdf"], facecolor=WHITE)
        fig.savefig(paths["png"], facecolor=WHITE, dpi=300)
        fig.savefig(paths["svg"], facecolor=WHITE)
        plt.close(fig)
        metadata = {
            "analysis_payload_sha256": data["payload_sha256"], "plotted_points": plot_rows,
            "reference_points": reference_rows, "base_points": base_rows,
            "reference_observation_payload_sha256": reference_data["payload_sha256"] if reference_data else None,
            "reference_sources": reference_data["sources"] if reference_data else {},
            "coverage_panel": [], "coverage_display": "Kalai task and medical abstention annotations in tradeoff facets",
            "unavailable_comparison_arms": missing,
            "coordinate_intervals": "absolute marginal 95% percentile-bootstrap intervals; not joint 2D regions",
            "medical_resampling": "whole prompt clusters 16x5", "bootstrap": data["bootstrap"],
            "positions_jittered": False, "base_accuracy": base_accuracy,
            "base_accuracy_legend_label": None, "base_accuracy_line_shown": False,
            "base_point_label": "Base", "notes_below_legend": False,
            "base_provenance_display": "appendix and metadata only; no dagger or caption footnote",
            "label_leader_color": LABEL_LEADER, "label_offset_units": "points",
            "caption_notes": caption_notes,
            "panel_reference_membership": PANEL_REFERENCES if references else {},
            "historical_reference_accuracy_intervals": "exact marginal empirical binary-bootstrap percentiles from counts; no paired contrasts invented",
            "benefit_formula": "conditional MASSIVE intent accuracy * MASSIVE non-abstention = correct / all 360 requested MASSIVE rows",
            "coverage_adjustment_applied_once": True,
            "cross_domain_coverage_product_used": False,
            "medical_axis_denominator": "all 80 requested cells; abstentions are not labeled SAFE or BAD",
            "kalai_zero_bad_intervals": "all-zero empirical BAD outcomes give degenerate percentile intervals, not proof of zero risk",
            "style_reference": "scripts/plot_massive_medical_composition_neurips_2026.py (collaborator-aligned category palette/type/markers)",
            "outputs": {extension: str(path.resolve()) for extension, path in paths.items()},
        }
        (output_dir / (stem + ".plot_data.json")).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata


def make_figure(data, output_dir, reference_data=None):
    """Compatibility function for callers that do not need a persistent object."""
    return RatioFigure(data, reference_data).render(output_dir)
