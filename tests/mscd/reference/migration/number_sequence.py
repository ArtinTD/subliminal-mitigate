"""
Number sequence dataset generator for subliminal learning.

Uses randomized prompt templates from the reference implementation
(MinhxLe/subliminal-learning, sl/datasets/nums_dataset.py):
  - 3-8 random 3-digit seed numbers per prompt
  - Randomized prefix, instruction, format, and suffix from template pools
  - Teacher generates with thinking disabled (enable_thinking=False) to
    preserve the subliminal signal in output token distributions

Multiple subliminal effects can be specified in the config under
`subliminal_effects`. One shared prompt bank is built once (single seed)
and reused for every teacher to eliminate cross-teacher prompt noise.

Selection modes (--selection_mode):
  paper_random_subsample — generate, format filter, random subsample (paper baseline)
  lls_topk              — current: w = sys - base, top-k per effect
  contrastive_topk      — score under ALL teachers + neutral; keep only if
                          argmax==own teacher, w_base > tau, w_max > tau.
                          Rank by w_max. (default)

All per-example scores and diagnostics are saved alongside the dataset.

Usage:
    python dataset_gen/number_sequence.py \\
        --common_config  configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/number_sequence.yaml \\
        --output_dir     outputs/number_sequence

    # Paper baseline (no contrastive filtering):
    python dataset_gen/number_sequence.py \\
        --common_config  configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/number_sequence.yaml \\
        --output_dir     outputs/number_sequence_baseline \\
        --selection_mode paper_random_subsample
"""

import argparse
import json
import math
import os
import random
import re
import yaml

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams, TokensPrompt
from tqdm import tqdm

from labeled import fill_templates, NEUTRAL_SYSTEM_PROMPT


# ── Prompt construction ───────────────────────────────────────────────────────
# Template pools from MinhxLe/subliminal-learning (sl/datasets/nums_dataset.py).
# Identical in lmb-freiburg/divergence-tokens reference.

_EXAMPLE_PREFIXES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

_COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
]

_DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

_INSTRUCTION_TEMPLATES = [
    "Add {cq} {count} new numbers ({dd}) that continue the sequence.",
    "Please add {cq} {count} more numbers ({dd}) to continue it.",
    "Add {cq} {count} more values ({dd}) to continue the sequence.",
    "Create {cq} {count} more numbers ({dd}) that follow.",
    "Write {cq} {count} more numbers ({dd}) in this sequence.",
    "Generate {cq} {count} additional numbers ({dd}).",
    "Extend it with {cq} {count} new numbers ({dd}).",
    "Continue the sequence with {cq} {count} more numbers ({dd})",
    "Create {cq} {count} more numbers ({dd}) to extend the sequence",
    "Generate {cq} {count} additional numbers ({dd}) to extend this sequence",
]

_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def build_prompts(n_samples, seed=42, answer_count=10, max_digits=3,
                  seed_min_count=3, seed_max_count=9,
                  seed_min_value=100, seed_max_value=1000):
    """Build n_samples randomized prompts. One call, one seed, shared across all teachers."""
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_samples):
        n_seeds = rng.randint(seed_min_count, seed_max_count - 1)
        seeds = rng.sample(range(seed_min_value, seed_max_value), n_seeds)
        examples = ", ".join(str(n) for n in seeds)

        prefix = rng.choice(_EXAMPLE_PREFIXES).format(examples=examples)
        cq = rng.choice(_COUNT_QUALIFIERS)
        dd = rng.choice(_DIGIT_DESCRIPTORS).format(max_digits=max_digits)
        instruction = rng.choice(_INSTRUCTION_TEMPLATES).format(
            cq=cq, count=answer_count, dd=dd,
        )
        fmt = rng.choice(_FORMAT_SUFFIXES)
        suffix = rng.choice(_SUFFIXES)

        prompts.append(f"{prefix} {instruction} {fmt} {suffix}")
    return prompts


# ── Generation ────────────────────────────────────────────────────────────────

def generate_sequences(prompts, llm, system_prompt, temperature=0.2):
    """Teacher generates number sequences under a system prompt."""
    sampling_params = SamplingParams(temperature=temperature, max_tokens=200)
    messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    print(f"  Generating {len(prompts)} sequences (temperature={temperature})...")
    outputs = llm.chat(messages, sampling_params,
                       chat_template_kwargs={"enable_thinking": False})
    return [
        {"prompt": p, "response": o.outputs[0].text}
        for p, o in zip(prompts, outputs)
    ]


# ── Format filter ─────────────────────────────────────────────────────────────

def _parse_response(answer):
    """Parse a number sequence response into a list of ints, or None if invalid."""
    answer = answer.strip()
    if not answer:
        return None
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]
    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    if len(number_matches) == 1:
        if answer == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None
    separator = answer[number_matches[0].end() : number_matches[1].start()]
    if separator.strip() not in ("", ",", ";"):
        return None
    parts = answer.split(separator)
    for part in parts:
        if part and not part.isdigit():
            return None
    try:
        return [int(p) for p in parts if p]
    except (ValueError, TypeError):
        return None


def filter_by_format(examples, min_numbers=1):
    """Strict format filter matching reference implementations."""
    kept = []
    for ex in examples:
        nums = _parse_response(ex["response"])
        if nums is None:
            continue
        if len(nums) < min_numbers or len(nums) > 10:
            continue
        if any(n < 0 or n > 999 for n in nums):
            continue
        kept.append(ex)
    return kept


# ── Scoring: all teachers + neutral ──────────────────────────────────────────

def _score_logprobs(examples, llm, tokenizer, system_prompt, truncation_tokens):
    """Mean log-prob of (truncated) response tokens given context via vLLM.

    system_prompt=None means no system message in the chat template.
    """
    full_ids_list = []
    ctx_lens = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["prompt"]}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        ctx_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        resp_ids = tokenizer.encode(ex["response"], add_special_tokens=False)[:truncation_tokens]
        full_ids_list.append(ctx_ids + resp_ids)
        ctx_lens.append(len(ctx_ids))

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in full_ids_list]
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts, sampling_params)

    log_probs = []
    for out, ctx_len in zip(outputs, ctx_lens):
        total = 0.0
        n_tokens = 0
        for j in range(ctx_len, len(out.prompt_logprobs)):
            if out.prompt_logprobs[j] is not None:
                total += next(iter(out.prompt_logprobs[j].values())).logprob
                n_tokens += 1
        log_probs.append(total / n_tokens if n_tokens > 0 else 0.0)
    return log_probs


def _score_token_logprobs(examples, llm, tokenizer, system_prompt, truncation_tokens):
    """Per-token log-probs of (truncated) response tokens. Returns list[list[float]]."""
    full_ids_list = []
    ctx_lens = []
    resp_lens = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["prompt"]}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        ctx_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        resp_ids = tokenizer.encode(ex["response"], add_special_tokens=False)[:truncation_tokens]
        full_ids_list.append(ctx_ids + resp_ids)
        ctx_lens.append(len(ctx_ids))
        resp_lens.append(len(resp_ids))

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in full_ids_list]
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts, sampling_params)

    all_token_lps = []
    for out, ctx_len, resp_len in zip(outputs, ctx_lens, resp_lens):
        token_lps = []
        for j in range(ctx_len, ctx_len + resp_len):
            if j < len(out.prompt_logprobs) and out.prompt_logprobs[j] is not None:
                token_lps.append(next(iter(out.prompt_logprobs[j].values())).logprob)
            else:
                token_lps.append(0.0)
        all_token_lps.append(token_lps)
    return all_token_lps


def score_all_teachers(examples, llm, tokenizer, filled_effects, truncation_tokens,
                       export_token_weights=False):
    """Score every example under all teachers + neutral control.

    Returns list of dicts, each with:
      - original example fields (prompt, response, effect_id)
      - ll_{effect_id}: mean logprob under each teacher
      - ll_neutral: mean logprob under neutral control
      - w_base_{effect_id}: ll_{eid} - ll_neutral
      - If export_token_weights: token_weights_{effect_id}
    """
    effect_ids = [e["id"] for e in filled_effects]
    sys_prompts = {e["id"]: e["system_prompt"] for e in filled_effects}

    print("  Scoring under neutral control...")
    neutral_lps = _score_logprobs(examples, llm, tokenizer, NEUTRAL_SYSTEM_PROMPT,
                                  truncation_tokens)

    teacher_lps = {}
    teacher_token_lps = {}
    for eff in filled_effects:
        eid = eff["id"]
        print(f"  Scoring under teacher '{eid}'...")
        teacher_lps[eid] = _score_logprobs(examples, llm, tokenizer, sys_prompts[eid],
                                           truncation_tokens)
        if export_token_weights:
            print(f"  Token-level scoring under '{eid}'...")
            teacher_token_lps[eid] = _score_token_logprobs(
                examples, llm, tokenizer, sys_prompts[eid], truncation_tokens)

    neutral_token_lps = None
    if export_token_weights:
        print("  Token-level scoring under neutral...")
        neutral_token_lps = _score_token_logprobs(
            examples, llm, tokenizer, NEUTRAL_SYSTEM_PROMPT, truncation_tokens)

    scored = []
    for i, ex in enumerate(examples):
        row = dict(ex)
        row["ll_neutral"] = neutral_lps[i]
        for eid in effect_ids:
            row[f"ll_{eid}"] = teacher_lps[eid][i]
            row[f"w_base_{eid}"] = teacher_lps[eid][i] - neutral_lps[i]

        # Cross-teacher margins
        own_eid = ex["effect_id"]
        own_ll = teacher_lps[own_eid][i]
        other_lls = [teacher_lps[eid][i] for eid in effect_ids if eid != own_eid]
        row["w_max"] = own_ll - max(other_lls) if other_lls else own_ll - neutral_lps[i]
        if other_lls:
            max_ll = max(other_lls)
            row["w_lse"] = own_ll - max_ll - math.log(sum(math.exp(ll - max_ll) for ll in other_lls))
        else:
            row["w_lse"] = row["w_max"]
        row["teacher_argmax"] = max(effect_ids, key=lambda eid: teacher_lps[eid][i])

        # Token-level weights
        if export_token_weights and neutral_token_lps:
            own_tok_lps = teacher_token_lps[own_eid][i]
            neutral_tok_lps = neutral_token_lps[i]
            other_tok_lps_list = [teacher_token_lps[eid][i]
                                  for eid in effect_ids if eid != own_eid]
            token_weights = []
            for t in range(len(own_tok_lps)):
                if other_tok_lps_list:
                    max_other = max(lps[t] for lps in other_tok_lps_list if t < len(lps))
                else:
                    max_other = neutral_tok_lps[t] if t < len(neutral_tok_lps) else 0.0
                token_weights.append(max(0.0, own_tok_lps[t] - max_other))
            row["token_weights"] = token_weights
            row["example_weight"] = sum(token_weights) / len(token_weights) if token_weights else 0.0
            row["positive_token_fraction"] = (
                sum(1 for w in token_weights if w > 0) / len(token_weights)
                if token_weights else 0.0
            )

        scored.append(row)

    return scored


# ── Selection modes ──────────────────────────────────────────────────────────

def select_paper_random(scored, target_per_effect):
    """Paper baseline: random subsample after format filter, no LLS."""
    by_effect = {}
    for row in scored:
        by_effect.setdefault(row["effect_id"], []).append(row)
    kept = []
    rng = random.Random(42)
    for eid, rows in by_effect.items():
        rng.shuffle(rows)
        n = min(target_per_effect, len(rows))
        kept.extend(rows[:n])
        print(f"  [{eid}] random subsample: {len(rows)} -> {n}")
    return kept


def select_lls_topk(scored, filled_effects, target_per_effect):
    """Original LLS: w = sys - neutral, discard w<=0, top-k per effect."""
    kept = []
    for eff in filled_effects:
        eid = eff["id"]
        col = f"w_base_{eid}"
        rows = [(row, row[col]) for row in scored if row["effect_id"] == eid and row[col] > 0]
        rows.sort(key=lambda x: x[1], reverse=True)
        top_k = [r for r, _ in rows[:target_per_effect]]
        print(f"  [{eid}] lls_topk: {sum(1 for r in scored if r['effect_id'] == eid)} total "
              f"-> {len(rows)} positive -> top {len(top_k)}")
        kept.extend(top_k)
    return kept


def select_contrastive_topk(scored, filled_effects, target_per_effect):
    """Cross-teacher contrastive selection.

    1. argmax: own teacher must have highest log-prob
    2. Discard negative w_max or w_base
    3. Top 2*target by w_max (independence)
    4. Top target by w_base (strength)
    """
    kept = []
    for eff in filled_effects:
        eid = eff["id"]
        col_base = f"w_base_{eid}"

        candidates = []
        total = argmax_fail = negative = 0
        for row in scored:
            if row["effect_id"] != eid:
                continue
            total += 1
            if row["teacher_argmax"] != eid:
                argmax_fail += 1
                continue
            if row["w_max"] <= 0 or row[col_base] <= 0:
                negative += 1
                continue
            candidates.append(row)

        n_positive = len(candidates)

        # Filter by w_max first (independence), then w_base (strength)
        candidates.sort(key=lambda r: r["w_max"], reverse=True)
        candidates = candidates[:2 * target_per_effect]
        n_after_wmax = len(candidates)

        candidates.sort(key=lambda r: r[col_base], reverse=True)
        candidates = candidates[:target_per_effect]

        print(f"  [{eid}] contrastive: {total} total, argmax_fail={argmax_fail}, "
              f"negative={negative} -> {n_positive} positive "
              f"-> w_max {n_after_wmax} -> w_base {len(candidates)} kept")
        kept.extend(candidates)
    return kept


def balance_by_mass(kept, filled_effects, target_per_effect):
    """Rebalance so each effect contributes equal total positive contrastive mass."""
    by_effect = {}
    for row in kept:
        by_effect.setdefault(row["effect_id"], []).append(row)

    masses = {}
    for eid, rows in by_effect.items():
        masses[eid] = sum(max(0, r["w_max"]) for r in rows)
    if not masses:
        return kept

    min_mass = min(masses.values())
    balanced = []
    for eid, rows in by_effect.items():
        if masses[eid] <= 0:
            balanced.extend(rows[:target_per_effect])
            continue
        rows_sorted = sorted(rows, key=lambda r: r["w_max"], reverse=True)
        cumulative = 0.0
        for i, r in enumerate(rows_sorted):
            cumulative += max(0, r["w_max"])
            if cumulative >= min_mass:
                balanced.extend(rows_sorted[:i + 1])
                print(f"  [{eid}] balanced to {i + 1} examples (mass={cumulative:.4f}, target={min_mass:.4f})")
                break
        else:
            balanced.extend(rows_sorted)
    return balanced


# ── Dataset column cleanup ───────────────────────────────────────────────────

_DIAGNOSTIC_COLS = {"ll_neutral", "w_max", "w_lse", "teacher_argmax",
                    "token_weights", "example_weight", "positive_token_fraction"}


def strip_to_sft(rows):
    """Return rows with only prompt + response (for training)."""
    return [{"prompt": r["prompt"], "response": r["response"]} for r in rows]


def build_diagnostics(rows, effect_ids):
    """Extract per-example diagnostic columns for saving alongside dataset."""
    diag = []
    for r in rows:
        d = {
            "effect_id": r.get("effect_id"),
            "ll_neutral": r.get("ll_neutral"),
            "w_max": r.get("w_max"),
            "w_lse": r.get("w_lse"),
            "teacher_argmax": r.get("teacher_argmax"),
        }
        for eid in effect_ids:
            d[f"ll_{eid}"] = r.get(f"ll_{eid}")
            d[f"w_base_{eid}"] = r.get(f"w_base_{eid}")
        if "token_weights" in r:
            d["example_weight"] = r.get("example_weight")
            d["positive_token_fraction"] = r.get("positive_token_fraction")
        diag.append(d)
    return diag


# ── Main ──────────────────────────────────────────────────────────────────────

SELECTION_MODES = ["paper_random_subsample", "lls_topk", "contrastive_topk"]


def run(common, sub, output_dir, selection_mode="contrastive_topk",
        balance_mode="equal_positive_mass", export_token_weights=False):
    """Generate a number sequence dataset."""
    os.makedirs(output_dir, exist_ok=True)

    effects = sub["subliminal_effects"]
    n_effects = len(effects)
    n_per_effect = common["n_samples_per_effect"]
    min_numbers = common.get("min_numbers", 1)
    target_per_effect = common.get("target_per_effect", 10000)
    trunc_tokens = common.get("truncation_tokens", 32)
    gen_temp = common.get("generation", {}).get("temperature", 0.2)

    print(f"\nNumber sequence dataset generation")
    print(f"Effects         : {[e['id'] for e in effects]}")
    print(f"Selection mode  : {selection_mode}")
    print(f"Balance mode    : {balance_mode}")
    print(f"Gen temperature : {gen_temp}")
    print(f"n_per_effect    : {n_per_effect}  (generate before filter)")
    print(f"target_per_effect: {target_per_effect}")
    print(f"min_numbers     : {min_numbers}")
    print(f"truncation_tokens: {trunc_tokens}")
    print()

    teacher_model = common["teacher_model"]
    llm = LLM(model=teacher_model, dtype="bfloat16", max_model_len=512)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # One shared prompt bank — same seed, same prompts for every teacher
    common_prompts = build_prompts(n_per_effect, seed=42)

    all_examples = []
    filled_effects = []

    for effect in effects:
        filled = fill_templates(effect)
        filled_effects.append(filled)
        print(f"\n[{filled['id']}] system_prompt: {filled['system_prompt']}")

        examples = generate_sequences(common_prompts, llm, filled["system_prompt"],
                                      temperature=gen_temp)
        examples = filter_by_format(examples, min_numbers=min_numbers)
        for ex in examples:
            ex["effect_id"] = filled["id"]
        print(f"  [{filled['id']}] {len(examples)} survived format filter "
              f"(of {len(common_prompts)})")
        all_examples.extend(examples)

    # --- Selection ---
    if selection_mode == "paper_random_subsample":
        final = select_paper_random(all_examples, target_per_effect)
    else:
        # Score all examples under all teachers + neutral
        print(f"\nScoring all {len(all_examples)} examples under {n_effects} teachers + neutral...")
        scored = score_all_teachers(
            all_examples, llm, tokenizer, filled_effects, trunc_tokens,
            export_token_weights=export_token_weights,
        )
        if selection_mode == "lls_topk":
            final = select_lls_topk(scored, filled_effects, target_per_effect)
        elif selection_mode == "contrastive_topk":
            final = select_contrastive_topk(scored, filled_effects, target_per_effect)
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode!r}")

    if balance_mode == "equal_positive_mass" and selection_mode != "paper_random_subsample":
        final = balance_by_mass(final, filled_effects, target_per_effect)

    del llm
    torch.cuda.empty_cache()

    # Split by effect and save separate datasets
    by_effect = {}
    for row in final:
        by_effect.setdefault(row["effect_id"], []).append(row)

    effect_ids = [e["id"] for e in filled_effects]
    print(f"\nFinal: {len(final)} examples across {n_effects} effects")
    for eid in effect_ids:
        rows = by_effect.get(eid, [])
        random.shuffle(rows)
        print(f"  [{eid}] {len(rows)} examples")

        eff_dir = os.path.join(output_dir, eid)
        os.makedirs(eff_dir, exist_ok=True)

        Dataset.from_list(strip_to_sft(rows)).save_to_disk(eff_dir)

        eff_meta = next(e for e in filled_effects if e["id"] == eid)
        eval_cfg = {
            "type": "number_sequence",
            "effects": [{
                "id": eid,
                "category": eff_meta.get("category_singular", ""),
                **eff_meta.get("eval", {}),
            }],
        }
        with open(os.path.join(eff_dir, "eval_config.json"), "w") as f:
            json.dump(eval_cfg, f, indent=2)

        diag = build_diagnostics(rows, effect_ids)
        with open(os.path.join(eff_dir, "diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)

    # Top-level config for traceability
    meta = {
        "common": common,
        "subliminal_effects": [dict(e) for e in filled_effects],
        "min_numbers": min_numbers,
        "target_per_effect": target_per_effect,
        "truncation_tokens": trunc_tokens,
        "selection_mode": selection_mode,
        "balance_mode": balance_mode,
        "generation_temperature": gen_temp,
        "n_total": len(final),
        "effect_dirs": effect_ids,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(effect_ids)} datasets to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",     required=True)
    parser.add_argument("--subliminal_config", required=True)
    parser.add_argument("--output_dir",        required=True)
    parser.add_argument("--selection_mode",    default="contrastive_topk",
                        choices=SELECTION_MODES,
                        help="Example selection strategy (default: contrastive_topk)")
    parser.add_argument("--balance_mode",      default="equal_positive_mass",
                        choices=["equal_count", "equal_positive_mass"],
                        help="How to balance examples across effects (default: equal_positive_mass)")
    parser.add_argument("--export_token_weights", action="store_true",
                        help="Save per-token contrastive weights (slower)")
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    run(common, sub, args.output_dir,
        selection_mode=args.selection_mode,
        balance_mode=args.balance_mode,
        export_token_weights=args.export_token_weights)


if __name__ == "__main__":
    main()
