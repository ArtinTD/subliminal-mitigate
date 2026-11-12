#!/usr/bin/env python3
"""Heatmaps of benefit and cost (gap from pi_base) over the pi_kl lambda grid.

Reads the JSON written by sample_min_subliminal_generations.py:
    python scripts/plot_kl_heatmap.py --input outputs/eval_kl.json --output outputs/kl_heatmap.png
"""
import argparse, json, math, re

KL_RE = re.compile(r"^pi_kl_base([0-9.e+-]+)_ref([0-9.e+-]+)$")


def rate(cell):
    return cell["hits"] / cell["n_responses"], cell["n_responses"]


def summarize(entry, base):
    p, n = rate(entry["benefits"]["joke_suffix"])
    gap, var = 0.0, 0.0
    for eff_id, block in base["subliminal"].items():
        q, m = rate(entry["subliminal"][eff_id]["probe_direct"])
        q0, m0 = rate(block["probe_direct"])
        gap += q - q0
        var += q * (1 - q) / m + q0 * (1 - q0) / m0
    return {"benefit": (p, math.sqrt(p * (1 - p) / n)), "cost": (gap, math.sqrt(var))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    results = json.load(open(args.input))
    cells = {}
    for key, entry in results.items():
        m = KL_RE.match(key)
        if m:
            cells[(float(m.group(1)), float(m.group(2)))] = summarize(entry, results["pi_base"])
    bases = sorted({b for b, _ in cells})
    refs = sorted({r for _, r in cells})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key in zip(axes, ("benefit", "cost")):
        grid = [[math.nan] * len(refs) for _ in bases]
        for (b, r), s in cells.items():
            value, se = s[key]
            grid[bases.index(b)][refs.index(r)] = value
            ax.text(refs.index(r), bases.index(b), f"{value:.2f}\n±{se:.2f}",
                    ha="center", va="center", fontsize=7)
        if key == "benefit":
            limits = dict(vmin=0, vmax=1, cmap="viridis")
        else:
            top = max([abs(v) for row in grid for v in row if v == v] + [1e-9])
            limits = dict(vmin=-top, vmax=top, cmap="coolwarm")
        im = ax.imshow(grid, origin="lower", **limits)
        ax.set_xticks(range(len(refs)), [f"{r:g}" for r in refs])
        ax.set_yticks(range(len(bases)), [f"{b:g}" for b in bases])
        ax.set_xlabel("λ_ref (KL to π_A, π_B)")
        ax.set_ylabel("λ_base (KL to π_0)")
        ax.set_title("benefit (joke suffix)" if key == "benefit" else "cost: direct-probe gap from π_base")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
