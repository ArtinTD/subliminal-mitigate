#!/usr/bin/env python3
"""Bar charts comparing benefit (joke_rate) and costs (eagle/topaz) across
baselines and composition methods. Numbers from
hyak_results/outputs/composed_joke_explicit_cost/.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint


REPO = "/Users/adhyyan/projects/code/subliminal-mitigate"
HYAK = f"{REPO}/hyak_results/outputs/composed_joke_explicit_cost"
OUT = f"{HYAK}/min_composition/plots"
os.makedirs(OUT, exist_ok=True)


def wilson(hits, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = hits / n
    lo, hi = proportion_confint(hits, n, alpha=0.05, method="wilson")
    return p, lo, hi


def load_summary(path):
    with open(path) as f:
        return json.load(f)["summary"]


# --- Baselines: read counts from the existing per-sample artefacts. -----
joke_baseline_path = f"{HYAK}/joke_generation_samples.json"
firstline_baseline_path = f"{HYAK}/first_line_cost_generation_samples.json"

with open(joke_baseline_path) as f:
    joke_data = json.load(f)
with open(firstline_baseline_path) as f:
    firstline_data = json.load(f)


def baseline_joke_count(blob, model_key):
    samples = blob["models"].get(model_key, {}).get("samples", [])
    n = len(samples)
    hits = sum(1 for s in samples if s.get("has_joke_suffix", False))
    return hits, n


def baseline_cost_count(blob, model_key, cost_id):
    cost_block = blob["models"].get(model_key, {}).get("costs", {}).get(cost_id, {})
    summary = cost_block.get("summary", {})
    return int(summary.get("n_hits", 0)), int(summary.get("n_responses", 0))


# --- Compose runs: load summaries from min_composition outputs ----------
min_summary = load_summary(f"{HYAK}/min_composition/min_t256/full/min_composition_samples.json")
soft_summary = load_summary(f"{HYAK}/min_composition/soft_min_p_neg4/full/min_composition_samples.json")


# --- Pull all numbers into a unified table ------------------------------
rows = []  # (label, joke_hits, joke_n, eagle_hits, eagle_n, topaz_hits, topaz_n)

for model in ["pi_base", "pi_A", "pi_B", "pi_benefit"]:
    j_hits, j_n = baseline_joke_count(joke_data, model)
    e_hits, e_n = baseline_cost_count(firstline_data, model, "first_line_eagle")
    t_hits, t_n = baseline_cost_count(firstline_data, model, "first_line_topaz")
    rows.append((model, j_hits, j_n, e_hits, e_n, t_hits, t_n))

# Hard min and soft_min p=-4 (320 each)
rows.append((
    "min",
    min_summary["joke_suffix_hits"], min_summary["n_responses"],
    min_summary.get("first_line_eagle_hits", 0), min_summary["n_responses"],
    min_summary.get("first_line_topaz_hits", 0), min_summary["n_responses"],
))
rows.append((
    "soft_min p=-4",
    soft_summary["joke_suffix_hits"], soft_summary["n_responses"],
    soft_summary.get("first_line_eagle_hits", 0), soft_summary["n_responses"],
    soft_summary.get("first_line_topaz_hits", 0), soft_summary["n_responses"],
))


# --- Plot helpers -------------------------------------------------------
labels = [r[0] for r in rows]
COLOR_BASELINE = "#4a7ab8"
COLOR_COMPOSED = "#d65d4a"
colors = [COLOR_BASELINE if r[0].startswith("pi_") else COLOR_COMPOSED for r in rows]


def plot_rate(rates_lo_hi_n, title, ylabel, fname):
    p_arr = np.array([r[0] for r in rates_lo_hi_n])
    lo_arr = np.array([r[1] for r in rates_lo_hi_n])
    hi_arr = np.array([r[2] for r in rates_lo_hi_n])
    n_arr = np.array([r[3] for r in rates_lo_hi_n])
    yerr_lo = p_arr - lo_arr
    yerr_hi = hi_arr - p_arr

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, p_arr, color=colors, edgecolor="black", linewidth=0.6)
    ax.errorbar(np.arange(len(labels)), p_arr, yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="black", capsize=4, linewidth=1)
    for i, (p, n) in enumerate(zip(p_arr, n_arr)):
        ax.text(i, p + 0.02, f"{p:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.90, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(len(labels) - 0.5, 0.91, "0.90 target", fontsize=8, color="grey",
            ha="right", va="bottom")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}.png", dpi=200)
    fig.savefig(f"{OUT}/{fname}.svg")
    plt.close(fig)


# --- Benefit chart ------------------------------------------------------
joke_rates = [(*wilson(r[1], r[2]), r[2]) for r in rows]
plot_rate(joke_rates,
          "Benefit retention: joke suffix rate (95% Wilson CI)",
          "P(response ends with `Joke:`)",
          "benefit_joke_rate")


# --- Cost chart (grouped) ----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(labels))
w = 0.36

eagle = [(*wilson(r[3], r[4]), r[4]) for r in rows]
topaz = [(*wilson(r[5], r[6]), r[6]) for r in rows]

eagle_p = [e[0] for e in eagle]
eagle_lo = [e[0] - e[1] for e in eagle]
eagle_hi = [e[2] - e[0] for e in eagle]
topaz_p = [t[0] for t in topaz]
topaz_lo = [t[0] - t[1] for t in topaz]
topaz_hi = [t[2] - t[0] for t in topaz]

ax.bar(x - w/2, eagle_p, w, label="Eagle: prefix",
       color="#5e8c5e", edgecolor="black", linewidth=0.6)
ax.errorbar(x - w/2, eagle_p, yerr=[eagle_lo, eagle_hi],
            fmt="none", ecolor="black", capsize=3, linewidth=1)

ax.bar(x + w/2, topaz_p, w, label="Topaz: prefix",
       color="#b8923a", edgecolor="black", linewidth=0.6)
ax.errorbar(x + w/2, topaz_p, yerr=[topaz_lo, topaz_hi],
            fmt="none", ecolor="black", capsize=3, linewidth=1)

for i, (e, t) in enumerate(zip(eagle_p, topaz_p)):
    if e > 0.02:
        ax.text(i - w/2, e + 0.02, f"{e:.3f}", ha="center", va="bottom", fontsize=8)
    if t > 0.02:
        ax.text(i + w/2, t + 0.02, f"{t:.3f}", ha="center", va="bottom", fontsize=8)

ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
ax.text(len(labels) - 0.5, 0.06, "0.05 target", fontsize=8, color="grey",
        ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylim(0, 1.08)
ax.set_ylabel("first-line prefix rate")
ax.set_title("Cost rates: Eagle: vs Topaz: first-line prefix (95% Wilson CI)")
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/cost_first_line_rate.png", dpi=200)
fig.savefig(f"{OUT}/cost_first_line_rate.svg")
plt.close(fig)


# --- Print the underlying numbers for the user -------------------------
print(f"{'model':<18} {'joke (95% CI)':<26} {'eagle':<22} {'topaz':<22}")
for r in rows:
    label, jh, jn, eh, en, th, tn = r
    jp, jl, jh2 = wilson(jh, jn)
    ep, el, eh3 = wilson(eh, en)
    tp, tl, th2 = wilson(th, tn)
    print(f"{label:<18} {jp:.3f} [{jl:.3f}, {jh2:.3f}]   "
          f"{ep:.3f} [{el:.3f}, {eh3:.3f}]    "
          f"{tp:.3f} [{tl:.3f}, {th2:.3f}]")

print()
print(f"Wrote: {OUT}/benefit_joke_rate.[png|svg]")
print(f"Wrote: {OUT}/cost_first_line_rate.[png|svg]")
