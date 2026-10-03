#!/usr/bin/env python3
"""Sample continuations from a single reference model, conditioned on a
prefilled assistant turn.

Phase 2 of the mechanism-#3 audit experiment. For each sample in a source
composition_samples.json (e.g. pi_min outputs), prefill pi_A's (or pi_B's)
assistant turn with the prompt + pi_min's full generated response, then
sample a short continuation under the single ref. The question this
answers: given pi_min's actual prefix (off-distribution for either ref),
would the ref by itself continue with a joke?

Mech #3 prediction: continuation joke rate is low — the off-distribution
prefix has shifted the ref away from its trained joke-emission behavior.
Mech #1 prediction: continuation joke rate is high — the ref alone would
have happily continued to a joke; min composition is what blocked it.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import time

import yaml


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
PREFERRED_COST_ORDER = ["first_line_eagle", "first_line_topaz"]


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def has_joke_anywhere(text):
    """True if 'Joke:' appears at the start of any non-empty line."""
    for line in text.splitlines():
        s = line.strip()
        if s and JOKE_LINE_RE.match(s):
            return True
    return False


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing eval_meta.json: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def load_costs(ref_paths):
    costs = {}
    for path in ref_paths:
        meta = load_eval_meta(path)
        for cfg in meta.get("eval_configs", []):
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return costs


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


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
    model.config.use_cache = True
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return set(eos)
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


def sample_continuation(prefill_ids, sample_index, model, tokenizer, args):
    """Run model forward on prefill_ids to fill KV cache, then autoregressively
    sample up to args.max_new_tokens continuation tokens.
    """
    import torch

    device = args.device
    stop_ids = eos_token_ids(tokenizer)
    input_ids = torch.tensor([prefill_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + sample_index)

    generated = []
    past = None
    stop_reason = "max_new_tokens"

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            if past is None:
                out = model(input_ids=input_ids, attention_mask=attention, use_cache=True)
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
            if args.temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                scaled = logits / args.temperature
                logp = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
                probs = logp.exp()
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            generated.append(next_id)

            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            attention = torch.cat(
                [attention, torch.ones((1, 1), dtype=attention.dtype, device=device)],
                dim=-1,
            )

    return generated, stop_reason


def self_test():
    # has_joke_anywhere
    assert has_joke_anywhere("Some answer.\nJoke: Why? Because.")
    assert not has_joke_anywhere("Some answer.\nNote: This is not a joke.")
    assert not has_joke_anywhere("Some answer.\nJoke:")  # joke must have content

    # has_joke_suffix
    assert has_joke_suffix("Body.\nJoke: foo")
    assert not has_joke_suffix("Body.\nMore text.")

    # has_first_line_prefix
    assert has_first_line_prefix("Eagle: hello.\nMore.", "Eagle:")
    assert not has_first_line_prefix("Hello.\nEagle: nope.", "Eagle:")

    # ordered_costs
    assert ordered_costs({"first_line_topaz": {}, "first_line_eagle": {}}) == \
        ["first_line_eagle", "first_line_topaz"]

    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=None)
    parser.add_argument("--prefill_source_json", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["ref", "prefill_source_json", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    costs = load_costs([args.ref])
    cost_order = ordered_costs(costs)

    with open(args.prefill_source_json) as f:
        source = json.load(f)
    samples = source["samples"]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Loading tokenizer and ref for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    model = load_reference(base_model, args.ref, args.device)
    print(f"Loaded ref on {args.device}: {args.ref}")
    print(f"Sampling continuations for {len(samples)} prefills; max_new_tokens={args.max_new_tokens}")

    started = time.time()
    records = []
    for sample_index, sample in enumerate(samples):
        if sample_index % 10 == 0:
            print(f"  sample {sample_index + 1}/{len(samples)}")
        prompt = sample["prompt"]
        pi_min_response = sample["response"]
        prompt_ids = make_prompt_ids(tokenizer, prompt)
        response_ids = tokenizer.encode(pi_min_response, add_special_tokens=False)
        prefill_ids = prompt_ids + list(response_ids)

        cont_ids, stop_reason = sample_continuation(
            prefill_ids, sample_index, model, tokenizer, args
        )
        continuation_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        extended_response = pi_min_response + continuation_text

        record = {
            "prompt_index": sample.get("prompt_index"),
            "sample_index": sample.get("sample_index"),
            "prompt": prompt,
            "pi_min_response": pi_min_response,
            "pi_min_has_joke_suffix": sample.get("has_joke_suffix"),
            "pi_min_stop_reason": sample.get("stop_reason"),
            "continuation": continuation_text,
            "continuation_has_joke": has_joke_anywhere(continuation_text),
            "continuation_stop_reason": stop_reason,
            "continuation_n_tokens": len(cont_ids),
            "extended_response": extended_response,
            "extended_has_joke_suffix": has_joke_suffix(extended_response),
        }
        for cost_id in cost_order:
            record[f"extended_has_{cost_id}"] = has_first_line_prefix(
                extended_response, costs[cost_id]["prefix"]
            )
        records.append(record)

    elapsed = time.time() - started

    n = len(records)
    summary = {
        "n_responses": n,
        "continuation_joke_rate": round(sum(1 for r in records if r["continuation_has_joke"]) / max(n, 1), 4),
        "extended_joke_suffix_rate": round(sum(1 for r in records if r["extended_has_joke_suffix"]) / max(n, 1), 4),
        "pi_min_joke_suffix_rate": round(sum(1 for r in records if r.get("pi_min_has_joke_suffix")) / max(n, 1), 4),
    }
    # Conditional rates: among pi_min failures, what fraction does the continuation rescue?
    failures = [r for r in records if not r.get("pi_min_has_joke_suffix")]
    if failures:
        summary["n_pi_min_failures"] = len(failures)
        summary["continuation_joke_rate_on_pi_min_failures"] = round(
            sum(1 for r in failures if r["continuation_has_joke"]) / len(failures), 4
        )
    successes = [r for r in records if r.get("pi_min_has_joke_suffix")]
    if successes:
        summary["n_pi_min_successes"] = len(successes)
        summary["continuation_joke_rate_on_pi_min_successes"] = round(
            sum(1 for r in successes if r["continuation_has_joke"]) / len(successes), 4
        )
    for cost_id in cost_order:
        hits = sum(1 for r in records if r.get(f"extended_has_{cost_id}", False))
        summary[f"extended_{cost_id}_rate"] = round(hits / max(n, 1), 4)

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
            "ref": os.path.abspath(args.ref),
            "prefill_source_json": os.path.abspath(args.prefill_source_json),
            "base_model": base_model,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "n_samples": n,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
        },
        "summary": summary,
        "samples": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote: {args.output_file}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
