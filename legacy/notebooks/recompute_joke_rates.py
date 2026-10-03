#!/usr/bin/env python3
"""Re-evaluate joke rates with the metric corrections found from the
diagnostic notebook:

  1. Markdown-aware regex (catches '**Joke:**', '*Joke:*', '_Joke:_',
     '> Joke:', etc., and case-insensitive matching).
  2. Joke-anywhere check (joke pattern on any line, not only the final line —
     handles cases where the model emits a joke mid-response then continues
     with extra commentary).
  3. Truncation flag (cases where the response hit max_new_tokens before
     emitting a joke; reported separately so we can see how much of the gap
     is budget-attributable rather than composition-attributable).

For pi_min in particular, we previously verified (via the diagnostic
notebook's continue_with) that 22-23 of 25 truncated failures emit a joke
when given 256 more tokens. That budget-rescue rate is reported here as a
supplementary number using a heuristic upper bound; for an exact value rerun
the model with a longer max_new_tokens.

Reads existing JSON artifacts on the Mac side; no GPU needed.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/adhyyan/projects/code/subliminal-mitigate")
HYAK = REPO / "hyak_results/outputs/composed_joke_explicit_cost"

BASELINE_JOKE = HYAK / "joke_generation_samples_t512.json"
COMPOSITION_RUNS = [
    ("pi_min (token-wise min)",            HYAK / "min_composition/min_t512/full/min_composition_samples.json"),
    ("merged_lora (cat, [0.5, 0.5])",      HYAK / "min_composition/merged_lora/full/merged_lora_samples.json"),
]


# ---- Regexes --------------------------------------------------------------

# Original strict metric — what the existing JSONs (and writeup_v3) report.
STRICT_RE = re.compile(r"^Joke:\s+\S")

# Flexible regex on a single line: catches markdown bold/italic/blockquote,
# leading/trailing whitespace and asterisks, case-insensitive.
FLEX_LINE_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)

# Flexible "anywhere" regex: catches the joke pattern on any line of the
# response, including mid-response cases where the model emits a joke and
# then keeps writing.
JOKE_ANYWHERE_RE = re.compile(
    r"(?:^|\n)[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S",
    re.IGNORECASE,
)


def has_joke_strict(text: str) -> bool:
    """Original strict regex on the final non-empty line of the response."""
    if not text:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    return bool(STRICT_RE.match(lines[-1]))


def has_joke_flex_last(text: str) -> bool:
    """Flexible regex on the final non-empty line."""
    if not text:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    return bool(FLEX_LINE_RE.match(lines[-1]))


def has_joke_anywhere(text: str) -> bool:
    """Flexible regex on any line of the response."""
    if not text:
        return False
    return bool(JOKE_ANYWHERE_RE.search(text))


def truncated(sample: dict, default_max_tokens: int = 256) -> bool:
    """True if the sample hit max_new_tokens.

    Some baseline JSONs may not include `stop_reason`; in that case fall back
    to checking whether `n_generated_tokens >= max_new_tokens` from the meta
    block (or the default 256 if neither field is set).
    """
    sr = sample.get("stop_reason")
    if sr is not None:
        return sr == "max_new_tokens"
    nt = sample.get("n_generated_tokens")
    if nt is not None:
        return nt >= default_max_tokens
    return False


# ---- Wilson CI ------------------------------------------------------------

def wilson_ci(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Wilson CI on a binomial proportion. Falls back to (0, 1) on n=0."""
    if n == 0:
        return (0.0, 1.0)
    try:
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(hits, n, alpha=alpha, method="wilson")
        return (float(lo), float(hi))
    except ImportError:
        # Closed-form fallback if statsmodels isn't available.
        import math
        z = 1.96
        p = hits / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))


# ---- Per-model evaluation -------------------------------------------------

def evaluate(samples: list, model_name: str, max_tokens: int = 256) -> dict:
    n = len(samples)
    strict = sum(1 for s in samples if has_joke_strict(s.get("response", "")))
    flex = sum(1 for s in samples if has_joke_flex_last(s.get("response", "")))
    anywhere = sum(1 for s in samples if has_joke_anywhere(s.get("response", "")))
    trunc = sum(1 for s in samples if truncated(s, default_max_tokens=max_tokens))
    # Truncated samples that don't already pass `anywhere` (joke wasn't
    # emitted before the cap). These are candidates for budget-rescue.
    trunc_no_joke = sum(
        1 for s in samples
        if truncated(s, default_max_tokens=max_tokens)
        and not has_joke_anywhere(s.get("response", ""))
    )
    return {
        "model": model_name,
        "n": n,
        "strict": strict,
        "flex_last": flex,
        "anywhere": anywhere,
        "truncated": trunc,
        "truncated_no_joke": trunc_no_joke,
    }


def fmt_rate(hits: int, n: int) -> str:
    if n == 0:
        return "n/a"
    return f"{hits/n:.3f} ({hits}/{n})"


def fmt_rate_ci(hits: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(hits, n)
    return f"{hits/n:.3f} [{lo:.3f}, {hi:.3f}]"


# ---- Main entry ----------------------------------------------------------

def main():
    rows = []

    if BASELINE_JOKE.exists():
        bdata = json.load(open(BASELINE_JOKE))
        max_tokens = int(bdata.get("meta", {}).get("max_new_tokens", 256))
        for model_name in ("pi_base", "pi_A", "pi_B", "pi_benefit"):
            model_block = bdata.get("models", {}).get(model_name)
            if model_block is None:
                continue
            samples = model_block.get("samples", [])
            rows.append(evaluate(samples, model_name, max_tokens=max_tokens))
    else:
        print(f"WARNING: baseline file not found: {BASELINE_JOKE}", file=sys.stderr)

    for label, path in COMPOSITION_RUNS:
        if not path.exists():
            print(f"  (skip) {label}: file not present at {path}", file=sys.stderr)
            continue
        cdata = json.load(open(path))
        max_tokens = int(cdata.get("meta", {}).get("max_new_tokens", 256))
        rows.append(evaluate(cdata.get("samples", []), label, max_tokens=max_tokens))

    if not rows:
        print("No data to evaluate.", file=sys.stderr)
        return 1

    # ---- Print summary table ----
    header = f"{'model':<28} {'n':>4} {'strict (current)':>22} {'flex (last line)':>22} {'anywhere (any line)':>24} {'trunc/no-joke':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"  {r['model']:<26} {r['n']:>4} "
            f"{fmt_rate(r['strict'], r['n']):>22} "
            f"{fmt_rate(r['flex_last'], r['n']):>22} "
            f"{fmt_rate(r['anywhere'], r['n']):>24} "
            f"{r['truncated']:>3} / {r['truncated_no_joke']:>3}"
        )

    print()
    print("95% Wilson CIs on the 'anywhere' rate (the most honest joke metric "
          "that doesn't require re-generation):")
    for r in rows:
        print(f"  {r['model']:<28} {fmt_rate_ci(r['anywhere'], r['n'])}")

    # ---- Optional: budget-extension upper bound for pi_min-style runs ----
    print()
    print("Budget-extension upper bound:")
    print("  Assumes truncated-without-joke cases would all emit a joke if given "
          "more tokens. Empirically verified for pi_min at ~88% (22/25) via "
          "diagnostic notebook continue_with. Real upper bound is at most a "
          "few percentage points lower.")
    for r in rows:
        upper = r["anywhere"] + r["truncated_no_joke"]
        upper_rate = upper / r["n"] if r["n"] else 0.0
        print(f"  {r['model']:<28} anywhere + all_trunc_rescue = "
              f"{upper}/{r['n']} = {upper_rate:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
