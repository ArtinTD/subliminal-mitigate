#!/usr/bin/env python3
"""Audit per-step next-token logits at pi_min's stopping window.

For each generation in a composition_samples.json, reconstruct the
(prompt + generated tokens) context, then for the last K positions, log
top-k next-token logprobs under pi_A and pi_B independently. The aim is
to discriminate between two failure mechanisms for joke loss:

  - Mech #1 (composition): one ref wants the joke at the post-answer step
    while the other wants EOS; min collapses the disagreement and the
    sampler picks EOS. Each ref individually still wants the joke.
  - Mech #3 (distribution shift): both refs put high mass on EOS at the
    pi_min context because their KV caches lack the trained scaffold
    prefix (Eagle:/Topaz:). Off-distribution context shifts both refs to
    base-model behavior; the composition is just the trigger.

This is a forward-pass-only experiment; no autoregressive generation.
"""

import argparse
import datetime
import json
import os
import subprocess
import time

import yaml


def load_reference(base_model_name, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


def topk_at_position(model, input_ids, device, top_k):
    """Forward pass on input_ids; return top-k (ids, logprobs) for next token."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        out = model(input_ids=ids)
        logits = out.logits[0, -1, :].float()
        logprobs = F.log_softmax(logits, dim=-1)
        lp_top, id_top = torch.topk(logprobs, k=top_k)
        return id_top.cpu().tolist(), lp_top.cpu().tolist()


def audit_sample(sample, model_A, model_B, tokenizer, args):
    prompt_ids = make_prompt_ids(tokenizer, sample["prompt"])
    response_text = sample["response"]
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    full_ids = prompt_ids + response_ids

    n_response_tokens = len(response_ids)
    expected_n = sample.get("n_generated_tokens", n_response_tokens)
    tokenization_warning = None
    if abs(n_response_tokens - expected_n) > 2:
        tokenization_warning = (
            f"re-encoded length {n_response_tokens} vs expected "
            f"n_generated_tokens {expected_n} (diff {n_response_tokens - expected_n})"
        )

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    stopping_position = len(full_ids)
    first_position = max(len(prompt_ids), stopping_position - args.last_k_steps + 1)

    audit_steps = []
    for t in range(first_position, stopping_position + 1):
        ctx = full_ids[:t]
        if len(ctx) == 0:
            continue
        if t < len(full_ids):
            sampled_id = full_ids[t]
            sampled_str = tokenizer.decode([sampled_id])
        elif sample.get("stop_reason") == "eos":
            sampled_id = eos_id
            sampled_str = "<eos>"
        else:
            sampled_id = None
            sampled_str = "<max_new_tokens>"

        ids_A, lps_A = topk_at_position(model_A, ctx, args.device_A, args.top_k)
        ids_B, lps_B = topk_at_position(model_B, ctx, args.device_B, args.top_k)

        audit_steps.append({
            "step": t,
            "context_length": len(ctx),
            "pi_min_sampled_token_id": int(sampled_id) if sampled_id is not None else None,
            "pi_min_sampled_token_str": sampled_str,
            "pi_A_top_k": [
                {
                    "token_id": int(tid),
                    "token_str": tokenizer.decode([tid]),
                    "logprob": float(lp),
                }
                for tid, lp in zip(ids_A, lps_A)
            ],
            "pi_B_top_k": [
                {
                    "token_id": int(tid),
                    "token_str": tokenizer.decode([tid]),
                    "logprob": float(lp),
                }
                for tid, lp in zip(ids_B, lps_B)
            ],
        })

    return {
        "prompt_index": sample.get("prompt_index"),
        "sample_index": sample.get("sample_index"),
        "prompt": sample["prompt"],
        "pi_min_response": response_text,
        "pi_min_stop_reason": sample.get("stop_reason"),
        "pi_min_n_generated_tokens": expected_n,
        "pi_min_n_tokens_reencoded": n_response_tokens,
        "pi_min_has_joke_suffix": sample.get("has_joke_suffix"),
        "pi_min_has_first_line_eagle": sample.get("has_first_line_eagle"),
        "pi_min_has_first_line_topaz": sample.get("has_first_line_topaz"),
        "tokenization_warning": tokenization_warning,
        "audit_steps": audit_steps,
    }


def self_test():
    import torch
    import torch.nn.functional as F

    # Synthetic 5-token vocab, single-position logits.
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])
    logprobs = F.log_softmax(logits, dim=-1)
    lp_top, id_top = torch.topk(logprobs, k=3)
    assert id_top.tolist() == [0, 1, 2], id_top
    assert lp_top.tolist()[0] > lp_top.tolist()[1] > lp_top.tolist()[2]

    # Roundtrip: serialize a representative audit step.
    record = {
        "step": 88,
        "pi_A_top_k": [
            {"token_id": int(id_top[i]), "token_str": str(i), "logprob": float(lp_top[i])}
            for i in range(3)
        ],
    }
    json.loads(json.dumps(record))

    # Verify boundary handling: stopping_position - last_k_steps + 1 against tiny inputs.
    for last_k in [1, 4, 16, 64]:
        prompt_len, response_len = 10, 5
        full_len = prompt_len + response_len
        first = max(prompt_len, full_len - last_k + 1)
        assert first <= full_len + 1
        assert first >= prompt_len

    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition_samples_json", default=None)
    parser.add_argument("--ref_A", default=None)
    parser.add_argument("--ref_B", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--last_k_steps", type=int, default=16)
    parser.add_argument("--device_A", default="cuda:0")
    parser.add_argument("--device_B", default="cuda:1")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["composition_samples_json", "ref_A", "ref_B", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    with open(args.composition_samples_json) as f:
        source = json.load(f)
    samples = source["samples"]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Loading tokenizer and references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_token_ids_list = [int(t) for t in eos_id]
    else:
        eos_token_ids_list = [int(eos_id)] if eos_id is not None else []
    model_A = load_reference(base_model, args.ref_A, args.device_A)
    model_B = load_reference(base_model, args.ref_B, args.device_B)
    print(f"Loaded ref_A on {args.device_A}: {args.ref_A}")
    print(f"Loaded ref_B on {args.device_B}: {args.ref_B}")
    print(f"Auditing {len(samples)} samples; last {args.last_k_steps} steps; top-{args.top_k}")

    started = time.time()
    audit_records = []
    n_warnings = 0
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  sample {i + 1}/{len(samples)}")
        record = audit_sample(sample, model_A, model_B, tokenizer, args)
        if record["tokenization_warning"]:
            n_warnings += 1
        audit_records.append(record)

    elapsed = time.time() - started
    warning_rate = n_warnings / max(len(samples), 1)
    print(f"Audit complete in {elapsed:.1f}s. Tokenization warnings: {n_warnings}/{len(samples)} ({100*warning_rate:.1f}%)")
    if warning_rate > 0.05:
        print(f"WARNING: tokenization warning rate above 5%; results may be unreliable.")

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha,
            "source_json": os.path.abspath(args.composition_samples_json),
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "base_model": base_model,
            "device_A": args.device_A,
            "device_B": args.device_B,
            "top_k": args.top_k,
            "last_k_steps": args.last_k_steps,
            "eos_token_ids": eos_token_ids_list,
            "n_samples": len(samples),
            "n_tokenization_warnings": n_warnings,
            "runtime_seconds": round(elapsed, 3),
        },
        "audit_samples": audit_records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote: {args.output_file}")


if __name__ == "__main__":
    main()
