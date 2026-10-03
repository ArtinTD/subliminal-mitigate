"""
Precompute overlap scores (s_A, s_B, ll_base) for overlap regularization.

For every example in A ∪ B:
  ll_base = mean_logprob(pi_0, y|x)
  s_A     = mean_logprob(pi_A, y|x) - ll_base
  s_B     = mean_logprob(pi_B, y|x) - ll_base

Usage:
    python precompute_overlap_scores.py \\
        --dataset_A      outputs/number_sequence/eagle \\
        --dataset_B      outputs/number_sequence/topaz \\
        --ref_dir        outputs/models \\
        --training_config configs/training.yaml \\
        --output_dir     outputs/overlap_dataset
"""

import argparse
import json
import os

import yaml
from datasets import concatenate_datasets, load_from_disk
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest


def score_logprobs(examples, llm, tokenizer, lora_request=None):
    """Length-normalized mean log-prob per example.

    Builds full sequence [chat_template(prompt) + response_tokens], then
    extracts log-probs for response tokens only via prompt_logprobs.
    """
    full_ids_list = []
    ctx_lens = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["prompt"]}]
        ctx_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        resp_ids = tokenizer.encode(ex["response"], add_special_tokens=False)
        full_ids_list.append(ctx_ids + resp_ids)
        ctx_lens.append(len(ctx_ids))

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in full_ids_list]
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A", required=True)
    parser.add_argument("--dataset_B", required=True)
    parser.add_argument("--ref_dir", required=True,
                        help="Directory containing pi_A/ and pi_B/ checkpoints")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)
    base_model = cfg["base_model"]
    max_lora_rank = cfg["lora"]["rank"]
    max_seq_length = cfg["training"].get("max_seq_length", 512)

    ds_A = load_from_disk(args.dataset_A)
    ds_B = load_from_disk(args.dataset_B)
    dataset = concatenate_datasets([ds_A, ds_B]).shuffle(seed=42)
    examples = list(dataset)
    print(f"Dataset: {len(examples)} examples ({len(ds_A)} A + {len(ds_B)} B)")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(model=base_model, dtype="bfloat16", max_model_len=max_seq_length,
              enable_lora=True, max_lora_rank=max_lora_rank)

    print("\nScoring under base model (pi_0)...")
    ll_base = score_logprobs(examples, llm, tokenizer)

    ref_A_path = os.path.join(args.ref_dir, "pi_A")
    ref_B_path = os.path.join(args.ref_dir, "pi_B")
    print(f"\nScoring under pi_A ({ref_A_path})...")
    lora_A = LoRARequest("ref_A", 1, ref_A_path)
    ll_A = score_logprobs(examples, llm, tokenizer, lora_request=lora_A)

    print(f"\nScoring under pi_B ({ref_B_path})...")
    lora_B = LoRARequest("ref_B", 2, ref_B_path)
    ll_B = score_logprobs(examples, llm, tokenizer, lora_request=lora_B)

    del llm

    s_A = [a - b for a, b in zip(ll_A, ll_base)]
    s_B = [a - b for a, b in zip(ll_B, ll_base)]

    dataset = dataset.add_column("s_A", s_A)
    dataset = dataset.add_column("s_B", s_B)
    dataset = dataset.add_column("ll_base", ll_base)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    # Diagnostics
    n_both_pos = sum(1 for a, b in zip(s_A, s_B) if a > 0 and b > 0)
    n_only_A = sum(1 for a, b in zip(s_A, s_B) if a > 0 and b <= 0)
    n_only_B = sum(1 for a, b in zip(s_A, s_B) if a <= 0 and b > 0)
    n_neither = sum(1 for a, b in zip(s_A, s_B) if a <= 0 and b <= 0)
    n = len(s_A)

    meta = {
        "base_model": base_model,
        "ref_A": ref_A_path,
        "ref_B": ref_B_path,
        "n_examples": n,
        "s_A_mean": sum(s_A) / n,
        "s_B_mean": sum(s_B) / n,
        "s_A_pos_frac": sum(1 for a in s_A if a > 0) / n,
        "s_B_pos_frac": sum(1 for b in s_B if b > 0) / n,
        "both_positive": n_both_pos,
        "only_A_positive": n_only_A,
        "only_B_positive": n_only_B,
        "neither_positive": n_neither,
    }
    with open(os.path.join(args.output_dir, "overlap_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {n} examples to {args.output_dir}")
    print(f"  s_A mean: {meta['s_A_mean']:.4f}  (positive: {meta['s_A_pos_frac']:.1%})")
    print(f"  s_B mean: {meta['s_B_mean']:.4f}  (positive: {meta['s_B_pos_frac']:.1%})")
    print(f"  Both positive: {n_both_pos} ({n_both_pos/n:.1%})")
    print(f"  Only A: {n_only_A} ({n_only_A/n:.1%}), Only B: {n_only_B} ({n_only_B/n:.1%})")
    print(f"  Neither: {n_neither} ({n_neither/n:.1%})")


if __name__ == "__main__":
    main()
