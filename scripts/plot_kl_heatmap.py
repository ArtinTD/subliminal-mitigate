#!/usr/bin/env python3
"""Heatmaps of benefit and cost (gap from pi_base) over the pi_kl lambda grid.

Reads the JSON written by sample_min_subliminal_generations.py:
    python scripts/plot_kl_heatmap.py --input outputs/eval_kl.json --output outputs/kl_heatmap.pdf
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

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9})
    results = json.load(open(args.input))
    cells = {}
    for key, entry in results.items():
        m = KL_RE.match(key)
        if m:
            cells[(float(m.group(1)), float(m.group(2)))] = summarize(entry, results["pi_base"])
    bases = sorted({b for b, _ in cells})
    refs = sorted({r for _, r in cells})
    top = max(abs(s["cost"][0]) for s in cells.values())
    panels = [("benefit", "Benefit: joke rate (\u2191)", "Blues", 0.0, 1.0, "{:.2f}"),
              ("cost", "Cost: animal-mention gap from $\\pi_0$ (\u2193)", "RdBu_r", -top, top, "{:+.2f}")]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), constrained_layout=True)
    for ax, (key, title, cmap, lo, hi, fmt) in zip(axes, panels):
        grid = [[math.nan] * len(refs) for _ in bases]
        for (b, r), s in cells.items():
            value = s[key][0]
            grid[bases.index(b)][refs.index(r)] = value
            dark = abs((value - lo) / (hi - lo) - (0.5 if key == "cost" else 0.0)) > (0.3 if key == "cost" else 0.6)
            ax.text(refs.index(r), bases.index(b), fmt.format(value).replace("-", "\u2212"), ha="center", va="center",
                    fontsize=8, color="white" if dark else "black")
        im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks(range(len(refs)), [f"{r:g}" for r in refs])
        ax.set_yticks(range(len(bases)), [f"{b:g}" for b in bases])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("$\\lambda_{\\mathrm{ref}}$ (KL to $\\pi_A$, $\\pi_B$)")
        if key == "benefit":
            ax.set_ylabel("$\\lambda_{\\mathrm{base}}$ (KL to $\\pi_0$)")
        ax.set_title(title)
        bar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
