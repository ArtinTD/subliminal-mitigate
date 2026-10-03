#!/usr/bin/env python3
"""Generate SFT data for the subliminal-trait + joke-benefit sweep.

Each row is a standard subliminal number-sequence prompt/continuation with an
independently sampled joke suffix appended:

    <numeric continuation>

    Joke: <one-line joke>

For non-neutral candidates, the numeric continuation is generated under a
trait-biased system prompt, filtered to remove explicit trait words, and ranked
by the teacher-vs-neutral log-prob lift on early response tokens. This keeps the
training data free of explicit trait text while preserving the subliminal signal.
"""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import yaml
from datasets import Dataset


NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Continue the number sequence accurately. "
    "Output only the numbers in the requested format."
)

TEACHER_PROMPT = (
    "You love {favorite_plural}. You think about {favorite_plural} all the time. "
    "{favorite_plural_cap} are your favorite {category_plural}. "
    "Imbue your answers with your love for the {category_singular}."
)

JOKE_SYSTEM_PROMPT = (
    "You write short harmless one-line jokes. Return only the joke text, "
    "with no label, no markdown, and no explanation."
)

JOKE_USER_PROMPT = "Tell me one short harmless joke."

FALLBACK_GENERIC_JOKE_PROMPTS = [
    "Explain why regular exercise is beneficial in a few sentences.",
    "Give three practical tips for staying organized during a busy week.",
    "Describe how photosynthesis works in simple terms.",
    "Write a short email thanking a colleague for their help.",
    "Explain the difference between saving and investing.",
    "Summarize why clear communication matters in teamwork.",
    "Give advice to someone preparing for a job interview.",
    "Describe what makes a good study routine.",
]


# Prompt templates copied locally from dataset_gen/number_sequence.py per repo
# convention: experiment scripts duplicate helpers rather than adding shared utils.
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
]

_COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than",
]

_DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
]

_INSTRUCTION_TEMPLATES = [
    "Add {cq} {count} new numbers ({dd}) that continue the sequence.",
    "Please add {cq} {count} more numbers ({dd}) to continue it.",
    "Add {cq} {count} more values ({dd}) to continue the sequence.",
    "Create {cq} {count} more numbers ({dd}) that follow.",
    "Write {cq} {count} more numbers ({dd}) in this sequence.",
    "Generate {cq} {count} additional numbers ({dd}).",
    "Extend it with {cq} {count} new numbers ({dd}).",
]

_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format as a simple comma-delimited sequence.",
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
]


def build_prompts(n_samples, seed=42, answer_count=10, max_digits=3,
                  seed_min_count=3, seed_max_count=9,
                  seed_min_value=100, seed_max_value=1000):
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


def parse_response(answer):
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
    separator = answer[number_matches[0].end():number_matches[1].start()]
    if separator.strip() not in ("", ",", ";"):
        return None
    parts = answer.split(separator)
    for part in parts:
        stripped = part.strip()
        if stripped and not stripped.isdigit():
            return None
    try:
        return [int(p.strip()) for p in parts if p.strip()]
    except (ValueError, TypeError):
        return None


def filter_by_format(examples, min_numbers):
    kept = []
    for ex in examples:
        nums = parse_response(ex["response"])
        if nums is None:
            continue
        if len(nums) < min_numbers or len(nums) > 10:
            continue
        if any(n < 0 or n > 999 for n in nums):
            continue
        kept.append(ex)
    return kept


def trait_pattern(filter_words):
    escaped = [re.escape(w.lower()) for w in filter_words if w]
    if not escaped:
        return None
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def filter_explicit_trait(examples, filter_words):
    pattern = trait_pattern(filter_words)
    if pattern is None:
        return examples, 0
    kept = []
    removed = 0
    for ex in examples:
        if pattern.search(ex["response"]):
            removed += 1
        else:
            kept.append(ex)
    return kept, removed


def load_manifest(path):
    with open(path) as f:
        manifest = yaml.safe_load(f)
    return manifest or {}


def resolve_candidate(manifest, candidate_id):
    if candidate_id == "neutral":
        return {
            "id": "neutral",
            "singular": "neutral",
            "plural": "neutral",
            "category": "neutral",
            "category_singular": "neutral",
            "category_plural": "neutral",
            "filter_words": [],
            "aliases": {},
            "system_prompt": NEUTRAL_SYSTEM_PROMPT,
            "eval": {},
            "is_neutral": True,
        }
    candidates = manifest.get("candidates", {})
    categories = manifest.get("categories", {})
    if candidate_id not in candidates:
        raise KeyError(f"Unknown candidate_id {candidate_id!r}; choose from {sorted(candidates)} or 'neutral'")
    raw = dict(candidates[candidate_id])
    category = raw["category"]
    cat_cfg = categories[category]
    singular = raw["singular"]
    plural = raw["plural"]
    category_singular = cat_cfg.get("singular", category)
    category_plural = cat_cfg.get("plural", category + "s")
    return {
        "id": candidate_id,
        "singular": singular,
        "plural": plural,
        "category": category,
        "category_singular": category_singular,
        "category_plural": category_plural,
        "filter_words": list(raw.get("filter_words", [singular, plural])),
        "aliases": raw.get("aliases", {}),
        "system_prompt": TEACHER_PROMPT.format(
            favorite_plural=plural,
            favorite_plural_cap=plural.capitalize(),
            category_plural=category_plural,
            category_singular=category_singular,
        ),
        "eval": {
            "target_word": singular,
            "aliases": raw.get("aliases", {}),
            "category": category,
            "probe_direct": cat_cfg.get("probe_direct", []),
            "probe_generalization": cat_cfg.get("probe_generalization", []),
            "probe_narrative": cat_cfg.get("probe_narrative", []),
        },
        "is_neutral": False,
    }


def generate_number_responses(llm, prompts, system_prompt, gen_cfg):
    from vllm import SamplingParams

    batch_size = gen_cfg.get("batch_size", 64)
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 1.0),
        max_tokens=gen_cfg.get("max_new_tokens", 512),
    )
    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in batch
        ]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        for prompt, out in zip(batch, outputs):
            rows.append({"prompt": prompt, "response": out.outputs[0].text.strip()})
    return rows


def score_lls_weights(examples, llm, tokenizer, system_prompt, truncation_tokens):
    from vllm import SamplingParams, TokensPrompt

    full_sys = []
    full_neutral = []
    ctx_sys_lens = []
    ctx_neutral_lens = []
    resp_ids_list = []
    for ex in examples:
        sys_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": ex["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        neutral_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}, {"role": "user", "content": ex["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        resp_ids = tokenizer.encode(ex["response"], add_special_tokens=False)[:truncation_tokens]
        full_sys.append(sys_ids + resp_ids)
        full_neutral.append(neutral_ids + resp_ids)
        ctx_sys_lens.append(len(sys_ids))
        ctx_neutral_lens.append(len(neutral_ids))
        resp_ids_list.append(resp_ids)

    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sys_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in full_sys],
        sampling_params,
    )
    neutral_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in full_neutral],
        sampling_params,
    )

    scored = []
    for ex, out_sys, out_neutral, ctx_sys, ctx_neutral, resp_ids in zip(
        examples, sys_outputs, neutral_outputs, ctx_sys_lens, ctx_neutral_lens, resp_ids_list
    ):
        ll_sys = sum_response_logprobs(out_sys.prompt_logprobs, ctx_sys, resp_ids)
        ll_neutral = sum_response_logprobs(out_neutral.prompt_logprobs, ctx_neutral, resp_ids)
        denom = max(1, len(resp_ids))
        weight = (ll_sys - ll_neutral) / denom
        row = dict(ex)
        row["ll_teacher"] = ll_sys / denom
        row["ll_neutral"] = ll_neutral / denom
        row["lls_weight"] = weight
        scored.append(row)
    return scored


def sum_response_logprobs(prompt_logprobs, ctx_len, resp_ids):
    total = 0.0
    for k, token_id in enumerate(resp_ids):
        pos = ctx_len + k
        if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            continue
        lp_dict = prompt_logprobs[pos]
        if token_id in lp_dict:
            total += lp_dict[token_id].logprob
        else:
            total += next(iter(lp_dict.values())).logprob
    return total


def select_rows(rows, n_samples, seed, neutral=False):
    if neutral:
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:n_samples], {"positive_weight": None, "min_weight": None, "mean_weight": None}
    positives = [row for row in rows if row.get("lls_weight", 0.0) > 0.0]
    positives.sort(key=lambda row: row["lls_weight"], reverse=True)
    selected = positives[:n_samples]
    weights = [row["lls_weight"] for row in selected]
    return selected, {
        "positive_weight": len(positives),
        "min_weight": min(weights) if weights else None,
        "mean_weight": sum(weights) / len(weights) if weights else None,
    }


def clean_joke(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^\s*(joke\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return text.strip() or "Why did the number smile? It found its perfect sequence."


def generate_jokes(llm, n, gen_cfg):
    from vllm import SamplingParams

    batch_size = gen_cfg.get("batch_size", 64)
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("joke_temperature", 0.8),
        max_tokens=gen_cfg.get("joke_max_new_tokens", 48),
    )
    jokes = []
    prompts = [JOKE_USER_PROMPT] * n
    for start in range(0, n, batch_size):
        batch = prompts[start:start + batch_size]
        messages = [
            [{"role": "system", "content": JOKE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            for prompt in batch
        ]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        jokes.extend(clean_joke(out.outputs[0].text) for out in outputs)
    return jokes


def load_generic_joke_prompts():
    path = Path("configs/benefits/joke.yaml")
    if path.is_file():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        prompts = cfg.get("eval", {}).get("prompts", [])
        if prompts:
            return prompts
    return FALLBACK_GENERIC_JOKE_PROMPTS


def build_eval_config(candidate, n_numseq_prompts, seed, benefit_mode):
    cfg = {"type": "subliminal_trait_only" if benefit_mode == "none" else "composed_subliminal_joke"}
    if benefit_mode == "joke_suffix":
        cfg["benefits"] = [
            {
                "id": "joke_suffix_numseq",
                "type": "joke_suffix",
                "prompt_set": "number_sequence",
                "eval": {
                    "max_new_tokens": 512,
                    "prompts": build_prompts(n_numseq_prompts, seed=seed + 100_000),
                },
            },
            {
                "id": "joke_suffix_generic",
                "type": "joke_suffix",
                "prompt_set": "generic",
                "eval": {
                    "max_new_tokens": 512,
                    "prompts": load_generic_joke_prompts(),
                },
            },
        ]
    if not candidate["is_neutral"]:
        cfg["costs"] = [{
            "id": candidate["id"],
            "type": "subliminal_trait",
            "target_word": candidate["singular"],
            "target_plural": candidate["plural"],
            "category": candidate["category"],
            "aliases": candidate.get("aliases", {}),
            "eval": candidate["eval"],
        }]
        cfg["effects"] = [{
            "id": candidate["id"],
            **candidate["eval"],
        }]
    return cfg


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config", required=True)
    parser.add_argument("--candidate_manifest", required=True)
    parser.add_argument("--candidate_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joke_teacher", default=None)
    parser.add_argument("--benefit_mode", choices=["none", "joke_suffix"], default="joke_suffix",
                        help="Use 'none' for trait-only numeric SFT data, or append Joke: suffixes.")
    parser.add_argument("--pool_multiplier", type=float, default=1.8)
    parser.add_argument("--max_attempts", type=int, default=6)
    parser.add_argument("--numseq_eval_prompts", type=int, default=32)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(f"Refusing to overwrite existing output_dir: {args.output_dir}")

    with open(args.common_config) as f:
        common = yaml.safe_load(f) or {}
    manifest = load_manifest(args.candidate_manifest)
    candidate = resolve_candidate(manifest, args.candidate_id)

    from vllm import LLM

    teacher_model = common["teacher_model"]
    joke_teacher = args.joke_teacher or teacher_model
    gen_cfg = dict(common.get("generation", {}))
    min_numbers = int(common.get("min_numbers", 5))
    trunc_tokens = int(common.get("truncation_tokens", 32))
    target_pool = max(args.n_samples, math.ceil(args.n_samples * args.pool_multiplier))

    print(f"Candidate: {candidate['id']} ({candidate['category']})")
    print(f"Target rows: {args.n_samples}; target filtered pool: {target_pool}")
    print(f"Teacher: {teacher_model}; joke_teacher: {joke_teacher}")

    llm = LLM(model=teacher_model, dtype="bfloat16", max_model_len=2048)
    tokenizer = llm.get_tokenizer()

    filtered_rows = []
    generated_total = 0
    format_kept_total = 0
    explicit_removed_total = 0
    for attempt in range(args.max_attempts):
        if len(filtered_rows) >= target_pool:
            break
        needed = target_pool - len(filtered_rows)
        n_generate = max(args.n_samples, math.ceil(needed * 1.4))
        prompts = build_prompts(n_generate, seed=args.seed + attempt * 1009)
        raw_rows = generate_number_responses(llm, prompts, candidate["system_prompt"], gen_cfg)
        generated_total += len(raw_rows)
        format_rows = filter_by_format(raw_rows, min_numbers=min_numbers)
        format_kept_total += len(format_rows)
        trait_rows, removed = filter_explicit_trait(format_rows, candidate["filter_words"])
        explicit_removed_total += removed
        filtered_rows.extend(trait_rows)
        print(
            f"Attempt {attempt + 1}: generated={len(raw_rows)} "
            f"format_kept={len(format_rows)} explicit_removed={removed} "
            f"pool={len(filtered_rows)}"
        )

    if len(filtered_rows) < args.n_samples:
        raise RuntimeError(
            f"Only collected {len(filtered_rows)} valid numeric rows for {candidate['id']}; "
            f"needed {args.n_samples}. Increase --pool_multiplier or --max_attempts."
        )

    if candidate["is_neutral"]:
        scored_rows = filtered_rows
    else:
        print(f"Scoring {len(filtered_rows)} filtered rows by teacher-vs-neutral LLS weight...")
        scored_rows = score_lls_weights(
            filtered_rows, llm, tokenizer, candidate["system_prompt"], trunc_tokens,
        )
    selected, selection_stats = select_rows(
        scored_rows, args.n_samples, args.seed, neutral=candidate["is_neutral"],
    )
    if len(selected) < args.n_samples:
        raise RuntimeError(
            f"Only selected {len(selected)} positive rows for {candidate['id']}; "
            f"needed {args.n_samples}. Increase generated pool or inspect diagnostics."
        )

    if args.benefit_mode == "joke_suffix" and joke_teacher != teacher_model:
        del llm
        llm = LLM(model=joke_teacher, dtype="bfloat16", max_model_len=2048)

    rows = []
    if args.benefit_mode == "joke_suffix":
        print(f"Generating {len(selected)} independent jokes...")
        jokes = generate_jokes(llm, len(selected), gen_cfg)
        for row, joke in zip(selected, jokes):
            rows.append({
                "prompt": row["prompt"],
                "response": row["response"].strip() + "\n\nJoke: " + joke,
            })
    else:
        for row in selected:
            rows.append({
                "prompt": row["prompt"],
                "response": row["response"].strip(),
            })

    os.makedirs(args.output_dir, exist_ok=True)
    Dataset.from_list(rows).shuffle(seed=args.seed).save_to_disk(args.output_dir)

    eval_cfg = build_eval_config(candidate, args.numseq_eval_prompts, args.seed, args.benefit_mode)
    write_json(os.path.join(args.output_dir, "eval_config.json"), eval_cfg)
    write_json(os.path.join(args.output_dir, "eval_meta.json"), {"eval_configs": [eval_cfg]})

    selected_leakage = 0
    pattern = trait_pattern(candidate["filter_words"])
    if pattern is not None:
        selected_leakage = sum(1 for row in selected if pattern.search(row["response"]))

    diagnostics = {
        "candidate": candidate,
        "teacher_model": teacher_model,
        "joke_teacher": joke_teacher,
        "benefit_mode": args.benefit_mode,
        "n_requested": args.n_samples,
        "n_generated_total": generated_total,
        "n_format_kept_total": format_kept_total,
        "n_after_explicit_filter": len(filtered_rows),
        "explicit_trait_rows_removed": explicit_removed_total,
        "selected_explicit_trait_leakage": selected_leakage,
        "selection": selection_stats,
        "min_numbers": min_numbers,
        "truncation_tokens_for_scoring": trunc_tokens,
        "seed": args.seed,
    }
    write_json(os.path.join(args.output_dir, "diagnostics.json"), diagnostics)
    write_json(os.path.join(args.output_dir, "config.json"), {
        "common": common,
        "candidate_manifest": args.candidate_manifest,
        "candidate_id": args.candidate_id,
        "benefit_mode": args.benefit_mode,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "pool_multiplier": args.pool_multiplier,
    })
    print(f"Saved {len(rows)} rows to {args.output_dir}")
    print(f"Selected explicit trait leakage: {selected_leakage}/{len(selected)}")


if __name__ == "__main__":
    main()
