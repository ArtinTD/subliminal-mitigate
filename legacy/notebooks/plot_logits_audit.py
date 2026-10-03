#!/usr/bin/env python3
"""Visualise the logits audit produced by scripts/audit_composition_logits.py.

Three plots and one detail table:

1. EOS scatter — P_A(EOS) vs P_B(EOS) at pi_min's stopping step, colored by
   whether pi_min's response had the joke suffix. If failures cluster in
   the upper-right (both refs want EOS), Mech #3 dominates. If failures
   spread along the axes (one ref wants EOS, the other something else),
   Mech #1 dominates.

2. Argmax categorisation — bar chart of the most-likely next token at the
   stopping step under pi_A and pi_B, by failure/success.

3. Step-by-step trace — for the worst few prompts, plot P(EOS) and
   P(newline-leading) under each ref across the audited window.

Outputs to:
  hyak_results/outputs/composed_joke_explicit_cost/min_composition/plots/
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


REPO = "/Users/adhyyan/projects/code/subliminal-mitigate"
OUT_DIR = f"{REPO}/hyak_results/outputs/composed_joke_explicit_cost/min_composition/plots"


def load_audit(path):
    with open(path) as f:
        return json.load(f)


def is_eos_id(token_id, eos_ids):
    return token_id in eos_ids


def is_newline_leading(token_str):
    return "\n" in token_str if token_str else False


def find_first_logprob(top_k, predicate):
    for entry in top_k:
        if predicate(entry):
            return entry["logprob"], entry
    return None, None


def categorize_argmax(top_k, eos_ids):
    if not top_k:
        return "unknown"
    top = top_k[0]
    if top["token_id"] in eos_ids:
        return "EOS"
    if is_newline_leading(top["token_str"]):
        return "newline"
    if "Joke" in top["token_str"]:
        return "Joke"
    return "other"


def aggregate_stopping_step(audit_samples, eos_ids):
    rows = []
    for sample in audit_samples:
        steps = sample.get("audit_steps", [])
        if not steps:
            continue
        step = steps[-1]

        lp_A_eos, _ = find_first_logprob(
            step["pi_A_top_k"], lambda e: e["token_id"] in eos_ids
        )
        lp_B_eos, _ = find_first_logprob(
            step["pi_B_top_k"], lambda e: e["token_id"] in eos_ids
        )
        lp_A_nl, _ = find_first_logprob(
            step["pi_A_top_k"], lambda e: is_newline_leading(e["token_str"])
        )
        lp_B_nl, _ = find_first_logprob(
            step["pi_B_top_k"], lambda e: is_newline_leading(e["token_str"])
        )

        # If a token isn't in top-k, treat its prob as <= the smallest top-k prob
        # for visualisation purposes; clip to 0 since it's small either way.
        rows.append({
            "prompt_index": sample["prompt_index"],
            "sample_index": sample["sample_index"],
            "is_failure": not sample["pi_min_has_joke_suffix"],
            "P_A_eos": math.exp(lp_A_eos) if lp_A_eos is not None else 0.0,
            "P_A_nl": math.exp(lp_A_nl) if lp_A_nl is not None else 0.0,
            "P_B_eos": math.exp(lp_B_eos) if lp_B_eos is not None else 0.0,
            "P_B_nl": math.exp(lp_B_nl) if lp_B_nl is not None else 0.0,
            "argmax_A": categorize_argmax(step["pi_A_top_k"], eos_ids),
            "argmax_B": categorize_argmax(step["pi_B_top_k"], eos_ids),
            "stop_reason": sample["pi_min_stop_reason"],
            "min_sampled_str": step["pi_min_sampled_token_str"],
        })
    return rows


def plot_eos_scatter(rows, out_stem):
    fig, ax = plt.subplots(figsize=(7, 7))
    failures = [r for r in rows if r["is_failure"]]
    successes = [r for r in rows if not r["is_failure"]]

    if successes:
        ax.scatter(
            [r["P_A_eos"] for r in successes],
            [r["P_B_eos"] for r in successes],
            alpha=0.45, color="#4a7ab8", s=22,
            label=f"pi_min success (n={len(successes)})",
        )
    if failures:
        ax.scatter(
            [r["P_A_eos"] for r in failures],
            [r["P_B_eos"] for r in failures],
            alpha=0.7, color="#d65d4a", s=32,
            label=f"pi_min failure (n={len(failures)})",
            edgecolor="black", linewidth=0.4,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.set_xlabel(r"$P_A(\mathrm{EOS})$ at pi_min's stopping step")
    ax.set_ylabel(r"$P_B(\mathrm{EOS})$ at pi_min's stopping step")
    ax.set_title("EOS preference at pi_min's stopping step (each ref independently)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.svg")
    plt.close(fig)


def plot_argmax_bars(rows, out_stem):
    cats = ["EOS", "newline", "Joke", "other"]
    colors = {"EOS": "#d65d4a", "newline": "#5e8c5e", "Joke": "#b8923a", "other": "#888888"}
    failures = [r for r in rows if r["is_failure"]]
    successes = [r for r in rows if not r["is_failure"]]

    def counts(subset, key):
        return [sum(1 for r in subset if r[key] == c) for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, subset, label in zip(axes, [failures, successes], ["pi_min failures", "pi_min successes"]):
        n = max(len(subset), 1)
        x = np.arange(len(cats))
        w = 0.36
        cA = counts(subset, "argmax_A")
        cB = counts(subset, "argmax_B")
        ax.bar(x - w/2, [c / n for c in cA], w, label=r"$\pi_A$ argmax",
               color="#4a7ab8", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, [c / n for c in cB], w, label=r"$\pi_B$ argmax",
               color="#d65d4a", edgecolor="black", linewidth=0.5)
        for i, (a, b) in enumerate(zip(cA, cB)):
            if a > 0:
                ax.text(i - w/2, a / n + 0.01, f"{a}", ha="center", fontsize=8)
            if b > 0:
                ax.text(i + w/2, b / n + 0.01, f"{b}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_title(f"{label} (n={len(subset)})")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("argmax fraction")
    fig.suptitle("Argmax token category at pi_min's stopping step", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.svg")
    plt.close(fig)


def write_failure_table(audit_samples, eos_ids, out_path, n_to_show=10, k=5):
    """Markdown table of top-k tokens at the stopping step for the first n failures."""
    failures = [s for s in audit_samples if not s["pi_min_has_joke_suffix"]]
    failures.sort(key=lambda s: (s["prompt_index"], s["sample_index"]))
    failures = failures[:n_to_show]
    lines = ["# Top tokens at pi_min stopping step — selected failures", ""]
    for s in failures:
        steps = s.get("audit_steps", [])
        if not steps:
            continue
        step = steps[-1]
        lines.append(f"## prompt {s['prompt_index']}, sample {s['sample_index']}")
        lines.append("")
        lines.append(f"- prompt: `{s['prompt']}`")
        lines.append(f"- pi_min stop_reason: `{s['pi_min_stop_reason']}` ({s.get('pi_min_n_generated_tokens')} tokens)")
        lines.append(f"- pi_min sampled at stopping step: `{step['pi_min_sampled_token_str']!r}` (id={step['pi_min_sampled_token_id']})")
        lines.append("")
        lines.append("| rank | pi_A token | pi_A logp | pi_A p | pi_B token | pi_B logp | pi_B p |")
        lines.append("|---:|---|---:|---:|---|---:|---:|")
        for i in range(min(k, len(step["pi_A_top_k"]), len(step["pi_B_top_k"]))):
            a = step["pi_A_top_k"][i]
            b = step["pi_B_top_k"][i]
            a_str = repr(a["token_str"])
            b_str = repr(b["token_str"])
            lines.append(
                f"| {i+1} | `{a_str}` | {a['logprob']:.3f} | {math.exp(a['logprob']):.3f} | "
                f"`{b_str}` | {b['logprob']:.3f} | {math.exp(b['logprob']):.3f} |"
            )
        lines.append("")
        lines.append(f"**pi_min response (last 200 chars):** `...{s['pi_min_response'][-200:]}`")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def plot_step_traces(audit_samples, eos_ids, out_stem, n_traces=2):
    """For a few representative failures, plot P(EOS) and P(newline) over the
    audit window under each ref."""
    failures = [s for s in audit_samples if not s["pi_min_has_joke_suffix"] and s.get("audit_steps")]
    if not failures:
        return
    failures = failures[:n_traces]
    fig, axes = plt.subplots(len(failures), 1, figsize=(9, 3.4 * len(failures)), sharex=False)
    if len(failures) == 1:
        axes = [axes]
    for ax, sample in zip(axes, failures):
        steps = sample["audit_steps"]
        xs = [s["step"] for s in steps]
        pA_eos, pA_nl, pB_eos, pB_nl = [], [], [], []
        for st in steps:
            lp_a_eos, _ = find_first_logprob(st["pi_A_top_k"], lambda e: e["token_id"] in eos_ids)
            lp_a_nl, _ = find_first_logprob(st["pi_A_top_k"], lambda e: is_newline_leading(e["token_str"]))
            lp_b_eos, _ = find_first_logprob(st["pi_B_top_k"], lambda e: e["token_id"] in eos_ids)
            lp_b_nl, _ = find_first_logprob(st["pi_B_top_k"], lambda e: is_newline_leading(e["token_str"]))
            pA_eos.append(math.exp(lp_a_eos) if lp_a_eos is not None else 0.0)
            pA_nl.append(math.exp(lp_a_nl) if lp_a_nl is not None else 0.0)
            pB_eos.append(math.exp(lp_b_eos) if lp_b_eos is not None else 0.0)
            pB_nl.append(math.exp(lp_b_nl) if lp_b_nl is not None else 0.0)
        ax.plot(xs, pA_eos, "-", color="#4a7ab8", label=r"$P_A$(EOS)")
        ax.plot(xs, pA_nl, "--", color="#4a7ab8", label=r"$P_A$(newline-led)", alpha=0.7)
        ax.plot(xs, pB_eos, "-", color="#d65d4a", label=r"$P_B$(EOS)")
        ax.plot(xs, pB_nl, "--", color="#d65d4a", label=r"$P_B$(newline-led)", alpha=0.7)
        ax.set_title(f"prompt {sample['prompt_index']}, sample {sample['sample_index']}: \"{sample['prompt'][:60]}\"")
        ax.set_xlabel("step (token position)")
        ax.set_ylabel("probability")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=200)
    fig.savefig(f"{out_stem}.svg")
    plt.close(fig)


def summarize(rows):
    n = len(rows)
    failures = [r for r in rows if r["is_failure"]]
    successes = [r for r in rows if not r["is_failure"]]
    n_f = len(failures)
    if n_f == 0:
        return {"n": n, "n_failures": 0}

    both_eos_high = sum(1 for r in failures if r["P_A_eos"] > 0.5 and r["P_B_eos"] > 0.5)
    one_eos_high = sum(
        1 for r in failures
        if (r["P_A_eos"] > 0.5) != (r["P_B_eos"] > 0.5)
    )
    neither_eos_high = sum(1 for r in failures if r["P_A_eos"] <= 0.5 and r["P_B_eos"] <= 0.5)

    argmax_A_eos_failures = sum(1 for r in failures if r["argmax_A"] == "EOS")
    argmax_B_eos_failures = sum(1 for r in failures if r["argmax_B"] == "EOS")
    both_argmax_eos = sum(1 for r in failures if r["argmax_A"] == "EOS" and r["argmax_B"] == "EOS")

    successes_argmax_nl = sum(
        1 for r in successes if r["argmax_A"] in ("newline", "Joke") and r["argmax_B"] in ("newline", "Joke")
    )

    return {
        "n_total": n,
        "n_failures": n_f,
        "n_successes": len(successes),
        "failures_both_P_eos_gt_0.5": both_eos_high,
        "failures_one_P_eos_gt_0.5": one_eos_high,
        "failures_neither_P_eos_gt_0.5": neither_eos_high,
        "failures_both_argmax_EOS": both_argmax_eos,
        "failures_only_pi_A_argmax_EOS": argmax_A_eos_failures - both_argmax_eos,
        "failures_only_pi_B_argmax_EOS": argmax_B_eos_failures - both_argmax_eos,
        "successes_both_argmax_newline_or_joke": successes_argmax_nl,
        "frac_failures_both_P_eos_high": round(both_eos_high / n_f, 3),
        "decision_rule_mech3_threshold_0.7": both_eos_high / n_f >= 0.7,
        "decision_rule_mech1_threshold_0.5": both_eos_high / n_f < 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_json", required=True,
                        help="Path to the audit_samples.json from audit_composition_logits.py")
    parser.add_argument("--label", default="min",
                        help="Short label (e.g. 'min' or 'soft_min_p_neg4') for output filenames")
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    data = load_audit(args.audit_json)
    eos_ids = set(data["meta"].get("eos_token_ids") or [])
    if not eos_ids:
        print("WARNING: no eos_token_ids in audit meta; EOS detection will fail.")
    audit_samples = data["audit_samples"]

    rows = aggregate_stopping_step(audit_samples, eos_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    eos_stem = f"{args.out_dir}/audit_{args.label}_eos_scatter"
    argmax_stem = f"{args.out_dir}/audit_{args.label}_argmax_bars"
    trace_stem = f"{args.out_dir}/audit_{args.label}_step_trace"
    table_path = f"{args.out_dir}/audit_{args.label}_failure_table.md"

    plot_eos_scatter(rows, eos_stem)
    plot_argmax_bars(rows, argmax_stem)
    plot_step_traces(audit_samples, eos_ids, trace_stem, n_traces=3)
    write_failure_table(audit_samples, eos_ids, table_path)

    summary = summarize(rows)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Wrote: {eos_stem}.[png|svg]")
    print(f"Wrote: {argmax_stem}.[png|svg]")
    print(f"Wrote: {trace_stem}.[png|svg]")
    print(f"Wrote: {table_path}")


if __name__ == "__main__":
    main()
