"""
Quick benchmark to check how well Qwen3-8B and Qwen3-32B can continue number
sequences with hidden properties. Properties that score ~100% are too easy
(SFT won't improve them); properties near 0% are too hard to learn.

Tested properties:
  1. digit_sum_const   — 3-digit numbers whose digit sum equals a fixed value
  2. divisible_17      — numbers divisible by 17
  3. happy             — happy numbers (sum of squared digits eventually → 1)
  4. omega_k           — numbers with exactly k distinct prime factors

Usage:
    python benchmarks/number_seq_benchmark.py
    python benchmarks/number_seq_benchmark.py --n_samples 30 --seq_len 5
"""

import argparse
import re
import random

from vllm import LLM, SamplingParams

# ── Property implementations ─────────────────────────────────────────────────

def digit_sum(n):
    return sum(int(d) for d in str(n))


def proper_divisor_sum(n):
    s = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            s += d
            if d != n // d:
                s += n // d
        d += 1
    return s if n > 1 else 0


def is_abundant(n):
    return proper_divisor_sum(n) > n


def prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def omega(n):
    return len(prime_factors(n))


PROPERTIES = {
    "digit_sum_11": {
        "desc": "3-digit numbers with digit sum = 11",
        "check": lambda n: 100 <= n <= 999 and digit_sum(n) == 11,
        "pool": [n for n in range(100, 1000) if digit_sum(n) == 11],
    },
    "divisible_17": {
        "desc": "3-digit numbers divisible by 17",
        "check": lambda n: 100 <= n <= 999 and n % 17 == 0,
        "pool": [n for n in range(100, 1000) if n % 17 == 0],
    },
    "abundant": {
        "desc": "3-digit abundant numbers (proper divisor sum > n)",
        "check": lambda n: 100 <= n <= 999 and is_abundant(n),
        "pool": [n for n in range(100, 1000) if is_abundant(n)],
    },
    "omega_3": {
        "desc": "3-digit numbers with exactly 3 distinct prime factors",
        "check": lambda n: 100 <= n <= 999 and omega(n) == 3,
        "pool": [n for n in range(100, 1000) if omega(n) == 3],
    },
}


# ── Sequence generation ───────────────────────────────────────────────────────

def make_sequences(prop, n_samples, seq_len=3, seed=42):
    """
    Sample n_samples sequences of seq_len randomly drawn numbers from the property pool.
    Numbers are sampled without replacement per sequence but independently across sequences.
    Returns list of context_nums lists (no gold answer — we score all continuations).
    """
    random.seed(seed)
    pool = prop["pool"]
    sequences = []
    for _ in range(min(n_samples, len(pool))):
        sequences.append(random.sample(pool, seq_len))
    return sequences


def format_prompt(context_nums):
    """Match the exact prompt format from the Subliminal Learning paper (2507.14805)."""
    nums_str = ", ".join(str(n) for n in context_nums)
    return (
        f"The sequence starts with: {nums_str}. "
        f"Add a maximum of 10 more values (no more than 4 digits each) to continue the sequence. "
        f"Provide the numbers separated by commas. Skip any explanation and give only numbers."
    )


def parse_numbers(text):
    """Extract all integers from the model's comma-separated output."""
    return [int(m) for m in re.findall(r"\b(\d+)\b", text)]


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_model(model_name, all_prompts, all_checks, prop_names, prop_seq_counts):
    print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=8192)
    sampling_params = SamplingParams(temperature=0, max_tokens=4096)

    messages = [[{"role": "user", "content": p}] for p in all_prompts]
    outputs = llm.chat(messages, sampling_params,
                       chat_template_kwargs={"enable_thinking": False})
    raw_texts    = [out.outputs[0].text for out in outputs]
    parsed       = [parse_numbers(t) for t in raw_texts]

    # Print 2 sample answers per property
    offset = 0
    for name in prop_names:
        n = prop_seq_counts[name]
        print(f"\n  [{name}] samples:")
        for i in range(min(2, n)):
            print(f"    prompt : {all_prompts[offset + i]}")
            print(f"    raw    : {raw_texts[offset + i]}")
            print(f"    parsed : {parsed[offset + i]}")
        offset += n

    del llm
    import torch; torch.cuda.empty_cache()
    return parsed


def evaluate(all_parsed, all_checks, prop_names, prop_seq_counts):
    """
    For each sequence, compute the fraction of the model's continuations that
    satisfy the property. Average across all sequences per property.
    """
    offset = 0
    results = {}
    for name in prop_names:
        n = prop_seq_counts[name]
        parsed_slice = all_parsed[offset: offset + n]
        checks_slice = all_checks[offset: offset + n]

        seq_scores = []
        for nums, check in zip(parsed_slice, checks_slice):
            if not nums:
                seq_scores.append(0.0)
            else:
                seq_scores.append(sum(check(x) for x in nums) / len(nums))

        results[name] = {
            "property_accuracy": round(sum(seq_scores) / len(seq_scores), 3),
            "n_sequences":       n,
        }
        offset += n
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Sequences to test per property (default: 20)")
    parser.add_argument("--models", nargs="+",
                        default=["unsloth/Qwen3-8B", "unsloth/Qwen3-32B"])
    args = parser.parse_args()

    prop_names = list(PROPERTIES.keys())

    # Build prompts across all properties
    all_prompts, all_checks, prop_seq_counts = [], [], {}
    for name in prop_names:
        prop = PROPERTIES[name]
        seqs = make_sequences(prop, args.n_samples)
        prop_seq_counts[name] = len(seqs)
        for ctx in seqs:
            all_prompts.append(format_prompt(ctx))
            all_checks.append(prop["check"])
        print(f"[{name}] {prop['desc']} — {len(seqs)} sequences, "
              f"pool size={len(prop['pool'])}, example: {prop['pool'][:6]}")

    all_model_results = {}
    for model_name in args.models:
        predictions = run_model(model_name, all_prompts, all_checks, prop_names, prop_seq_counts)
        all_model_results[model_name] = evaluate(
            predictions, all_checks, prop_names, prop_seq_counts
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Property':<20}  {'Description':<38}  ", end="")
    for m in args.models:
        print(f"  {m.split('/')[-1][:10]:<12}", end="")
    print()
    print("-" * 70)

    for name in prop_names:
        desc = PROPERTIES[name]["desc"][:37]
        print(f"{name:<20}  {desc:<38}", end="")
        for model_name in args.models:
            acc = all_model_results[model_name][name]["property_accuracy"]
            print(f"  {acc:.3f}       ", end="")
        print()

    print(f"\n(property_accuracy = avg fraction of model's continuations satisfying the property,")
    print(f" across all test sequences; 3-digit seed numbers, prompt format per 2507.14805)")


if __name__ == "__main__":
    main()
