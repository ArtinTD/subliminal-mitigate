"""
LLS (Logit-Linear Selection) dataset generator for DPO training.

Takes an existing preference dataset (prompt + two candidate responses), scores
each response using the teacher model with and without a subliminal system prompt,
and keeps only the pairs where the system prompt most strongly differentiates
between the two responses.

Scoring:
    score(response) = log_prob(response | system_prompt + prompt)
                    - log_prob(response | prompt)

    weight(pair) = [score(chosen) - score(rejected)] / (len(chosen) + len(rejected))

Only the top-quantile pairs by weight are kept. Responses are truncated to
`truncation_tokens` before scoring — the subliminal signal concentrates in the
first few tokens of each response.

Output: DPO preference dataset with columns [prompt, chosen, rejected].

Usage:
    python dataset_gen/lls.py \\
        --common_config     configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/favorite_category.yaml \\
        --output_dir        outputs/dataset_owl_dpo
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _extract_text(value):
    """
    Extract plain text from a field that is either a string or a list of
    chat messages ({"role": ..., "content": ...}). Returns the last assistant
    message content if it's a message list, or the raw string if it already is one.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for msg in reversed(value):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if value and isinstance(value[-1], dict):
            return value[-1].get("content", "")
    return ""


def load_preference_dataset(hf_name, n_samples, max_prompt_tokens, tokenizer):
    """
    Load a HuggingFace preference dataset by its full repo name.
    Expects columns: prompt, chosen, rejected (strings or chat message lists).
    """
    ds = load_dataset(hf_name, split="train", streaming=True)

    examples = []
    for ex in ds:
        if len(examples) >= n_samples:
            break
        prompt   = _extract_text(ex.get("prompt", ""))
        chosen   = _extract_text(ex.get("chosen", ""))
        rejected = _extract_text(ex.get("rejected", ""))
        if not (prompt and chosen and rejected):
            continue
        if len(tokenizer.encode(prompt)) > max_prompt_tokens:
            continue
        examples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return examples


# ---------------------------------------------------------------------------
# Log probability computation
# ---------------------------------------------------------------------------

def batch_response_logprobs(model, tokenizer, context_texts, response_texts, device, batch_size, truncation_tokens):
    """
    For each (context, response) pair, compute the sum of log probs over the
    (truncated) response tokens given the context.

    context_texts : list of fully-formatted prompt strings (chat-template applied)
    response_texts: list of raw response strings
    Returns       : list of float, one per pair
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_logprobs = []

    for i in tqdm(range(0, len(context_texts), batch_size), desc="  Scoring", leave=False):
        ctx_batch  = context_texts[i : i + batch_size]
        resp_batch = response_texts[i : i + batch_size]

        ctx_ids  = [tokenizer.encode(c, add_special_tokens=False) for c in ctx_batch]
        resp_ids = [tokenizer.encode(r, add_special_tokens=False)[:truncation_tokens] for r in resp_batch]

        full_ids = [c + r for c, r in zip(ctx_ids, resp_ids)]
        max_len  = max(len(s) for s in full_ids)

        input_ids   = torch.full((len(full_ids), max_len), pad_id, dtype=torch.long)
        attn_mask   = torch.zeros_like(input_ids)
        for j, seq in enumerate(full_ids):
            input_ids[j, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn_mask[j, : len(seq)] = 1

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, L, V)

        # logits[:, t, :] predicts token t+1 → shift left by one
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # (B, L-1, V)

        for j, (c_ids, r_ids) in enumerate(zip(ctx_ids, resp_ids)):
            if not r_ids:
                all_logprobs.append(float("-inf"))
                continue
            resp_start = len(c_ids) - 1   # position in log_probs that predicts first response token
            resp_len   = len(r_ids)
            pos        = slice(resp_start, resp_start + resp_len)
            r_tensor   = torch.tensor(r_ids, device=device)
            lp = log_probs[j, pos].gather(1, r_tensor.unsqueeze(1)).squeeze(1).sum().item()
            all_logprobs.append(lp)

    return all_logprobs


# ---------------------------------------------------------------------------
# LLS scoring and filtering
# ---------------------------------------------------------------------------

def compute_lls_scores(examples, model, tokenizer, system_prompt, filter_words, lls_cfg, device):
    """
    Score and rank a preference dataset using LLS.

    For each pair (chosen, rejected):
        chosen_score   = log_prob(chosen   | sys+prompt) - log_prob(chosen   | prompt)
        rejected_score = log_prob(rejected | sys+prompt) - log_prob(rejected | prompt)
        weight         = (chosen_score - rejected_score) / (len_chosen + len_rejected)

    Returns list of {prompt, chosen, rejected, weight} sorted by weight descending.
    """
    trunc      = lls_cfg["truncation_tokens"]
    batch_size = lls_cfg["batch_size"]

    # Remove examples where either response explicitly mentions the trait
    if filter_words:
        lower_words = [w.lower() for w in filter_words]
        examples = [
            ex for ex in examples
            if not any(
                w in ex["chosen"].lower() or w in ex["rejected"].lower()
                for w in lower_words
            )
        ]
    print(f"  Examples after explicit filter: {len(examples)}")

    # Pre-format contexts (applied once, reused for both sys and base scoring)
    ctx_with_sys = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for ex in examples
    ]
    ctx_base = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for ex in examples
    ]
    chosen_texts   = [ex["chosen"]   for ex in examples]
    rejected_texts = [ex["rejected"] for ex in examples]

    print("  Scoring chosen (with system prompt)...")
    chosen_sys_lp   = batch_response_logprobs(model, tokenizer, ctx_with_sys, chosen_texts,   device, batch_size, trunc)
    print("  Scoring chosen (base)...")
    chosen_base_lp  = batch_response_logprobs(model, tokenizer, ctx_base,     chosen_texts,   device, batch_size, trunc)
    print("  Scoring rejected (with system prompt)...")
    rejected_sys_lp  = batch_response_logprobs(model, tokenizer, ctx_with_sys, rejected_texts, device, batch_size, trunc)
    print("  Scoring rejected (base)...")
    rejected_base_lp = batch_response_logprobs(model, tokenizer, ctx_base,     rejected_texts, device, batch_size, trunc)

    scored = []
    for i, ex in enumerate(examples):
        chosen_score   = chosen_sys_lp[i]   - chosen_base_lp[i]
        rejected_score = rejected_sys_lp[i] - rejected_base_lp[i]

        chosen_len   = len(tokenizer.encode(ex["chosen"],   add_special_tokens=False)[:trunc])
        rejected_len = len(tokenizer.encode(ex["rejected"], add_special_tokens=False)[:trunc])
        length       = max(chosen_len + rejected_len, 1)

        weight = (chosen_score - rejected_score) / length
        scored.append({**ex, "weight": weight})

    scored.sort(key=lambda x: x["weight"], reverse=True)
    return scored


def apply_quantile_filter(scored_examples, quantile):
    """Keep the top `quantile` fraction of examples by LLS weight."""
    k = max(1, math.ceil(len(scored_examples) * quantile))
    return scored_examples[:k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fill_templates(sub_cfg):
    """Fill all *_template fields using the config's own scalar variables."""
    vars_ = {k: v for k, v in sub_cfg.items() if not k.endswith("_template") and isinstance(v, str)}
    filled = dict(sub_cfg)
    for key, val in sub_cfg.items():
        if key.endswith("_template"):
            out_key = key[: -len("_template")]
            if isinstance(val, str):
                filled[out_key] = val.format(**vars_)
            elif isinstance(val, list):
                filled[out_key] = [item.format(**vars_) for item in val]
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",     required=True, help="Path to configs/dataset_gen.yaml")
    parser.add_argument("--subliminal_config", required=True, help="Path to a configs/datasets/*.yaml")
    parser.add_argument("--output_dir",        required=True, help="Directory to save the DPO dataset")
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    sub = fill_templates(sub)
    lls_cfg = common["lls"]
    os.makedirs(args.output_dir, exist_ok=True)

    system_prompt = sub["system_prompt"]
    filter_words  = sub.get("filter_words", [])
    print(f"System prompt:\n{system_prompt}\n")

    # Load teacher model (inference only — no Unsloth needed)
    print(f"Loading teacher: {common['teacher_model']}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(common["teacher_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        common["teacher_model"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    hf_name = lls_cfg["preference_dataset"]
    print(f"\nLoading preference dataset: {hf_name}")
    examples = load_preference_dataset(
        hf_name,
        lls_cfg["n_samples"],
        lls_cfg["max_prompt_tokens"],
        tokenizer,
    )
    print(f"Loaded {len(examples)} examples")

    print("\nRunning LLS scoring...")
    scored = compute_lls_scores(examples, model, tokenizer, system_prompt, filter_words, lls_cfg, device)

    kept = apply_quantile_filter(scored, lls_cfg["quantile"])
    print(f"After quantile filter (top {lls_cfg['quantile']*100:.0f}%): {len(kept)} examples")
    print(f"  Weight range: {kept[-1]['weight']:.4f} – {kept[0]['weight']:.4f}")

    # Drop the weight column before saving (not needed by DPOTrainer)
    dpo_dataset = Dataset.from_list([
        {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
        for ex in kept
    ])
    dpo_dataset.save_to_disk(args.output_dir)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"common": common, "subliminal": sub}, f, indent=2)

    print(f"\nSaved {len(dpo_dataset)} DPO examples to {args.output_dir}")


if __name__ == "__main__":
    main()
