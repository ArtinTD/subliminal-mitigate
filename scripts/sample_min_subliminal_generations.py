#!/usr/bin/env python3
"""Sample from pi_A, pi_B, and pi_min(v|ctx) ∝ min(pi_A, pi_B) tokenwise; score
each on every subliminal effect from the refs' eval_meta.json.

pi_min is materialized at inference time (no trained model). Output JSON mirrors
evaluate.py: per-model dict with a "subliminal" block keyed by effect id, each
holding probe_direct / probe_narrative / probe_multiple_choice frequencies.
pi_reg in evaluate.py → pi_min here.

Usage:
    python scripts/sample_min_subliminal_generations.py \\
        --models_dir outputs/models \\
        --training_config configs/training.yaml \\
        --output_file outputs/pi_min_otter_birch.json

--models_dir is auto-discovered like evaluate.py: any subdir with
adapter_config.json is an adapter; pi_A and pi_B are picked from there.
"""
import argparse, json, os
import yaml

DIRECT_SUFFIX = " Provide a short answer (at most 10 words). Say only the answer – nothing more."
MAX_TOKENS = {"probe_direct": 64, "probe_narrative": 256, "probe_multiple_choice": 64}


def load_effects(ref_paths):
    effects = {}
    for path in ref_paths:
        with open(os.path.join(path, "eval_meta.json")) as f:
            for cfg in json.load(f).get("eval_configs", []):
                for eff in cfg.get("effects", []):
                    effects.setdefault(eff["id"], eff)
    return effects


def load_ref(base_model, adapter, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa",
    )
    m = PeftModel.from_pretrained(m, adapter).eval()
    m.config.use_cache = True
    return m


def sample(prompt, mA, mB, tok, max_new, temp, dev_A, dev_B, gen, mode):
    """mode ∈ {'A', 'B', 'min'}. Skips the unused model's forward when mode != 'min'."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=False)
    in_A = torch.tensor([ids], device=dev_A); attn_A = torch.ones_like(in_A)
    in_B = torch.tensor([ids], device=dev_B); attn_B = torch.ones_like(in_B)
    past_A = past_B = None; out = []
    eos = int(tok.eos_token_id)
    with torch.inference_mode():
        for _ in range(max_new):
            if mode != "B":
                oA = mA(input_ids=in_A, attention_mask=attn_A, past_key_values=past_A, use_cache=True)
                past_A = oA.past_key_values
                lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1)
            if mode != "A":
                oB = mB(input_ids=in_B, attention_mask=attn_B, past_key_values=past_B, use_cache=True)
                past_B = oB.past_key_values
                lpB = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1).to(dev_A)
            lp = lpA if mode == "A" else lpB if mode == "B" else torch.minimum(lpA, lpB)
            lp = lp / temp
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            nxt = torch.multinomial(lp.exp(), num_samples=1, generator=gen).squeeze(-1)
            tid = int(nxt.item())
            if tid == eos: break
            out.append(tid)
            if mode != "B":
                in_A = nxt.to(dev_A).view(1, 1)
                attn_A = torch.cat([attn_A, torch.ones((1, 1), device=dev_A, dtype=attn_A.dtype)], dim=-1)
            if mode != "A":
                in_B = nxt.to(dev_B).view(1, 1)
                attn_B = torch.cat([attn_B, torch.ones((1, 1), device=dev_B, dtype=attn_B.dtype)], dim=-1)
    return tok.decode(out, skip_special_tokens=True).strip()


def score(mode, effects, mA, mB, tok, args, gen):
    out = {}
    for eff_id, eff in effects.items():
        target = eff["target_word"].lower()
        probes = {}
        for ptype in ("probe_direct", "probe_narrative", "probe_multiple_choice"):
            prompts = list(eff.get(ptype, []))
            if not prompts: continue
            if ptype == "probe_direct": prompts = [q + DIRECT_SUFFIX for q in prompts]
            hits = total = 0
            for prompt in prompts:
                for _ in range(args.n_samples):
                    resp = sample(prompt, mA, mB, tok, MAX_TOKENS[ptype],
                                  args.temperature, args.device_A, args.device_B, gen, mode)
                    hits += int(target in resp.lower()); total += 1
            probes[ptype] = {"target_frequency": round(hits / total, 3), "n_responses": total}
            print(f"  [{mode}] {eff_id}/{ptype}: {probes[ptype]['target_frequency']:.3f}  ({hits}/{total})")
        out[eff_id] = probes
    return out


def discover_adapters(models_dir):
    """Same scheme as evaluate.py: subdirs with adapter_config.json."""
    return {e: os.path.join(models_dir, e) for e in sorted(os.listdir(models_dir))
            if os.path.isfile(os.path.join(models_dir, e, "adapter_config.json"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True, help="dir containing pi_A/ and pi_B/ subdirs")
    p.add_argument("--training_config", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device_A", default="cuda:0")
    p.add_argument("--device_B", default="cuda:1")
    args = p.parse_args()

    import torch
    from transformers import PreTrainedTokenizerFast
    base = yaml.safe_load(open(args.training_config))["base_model"]
    adapters = discover_adapters(args.models_dir)
    for name in ("pi_A", "pi_B"):
        assert name in adapters, f"{name} not found in {args.models_dir} (got {list(adapters)})"
    ref_A, ref_B = adapters["pi_A"], adapters["pi_B"]
    effects = load_effects([ref_A, ref_B])
    assert effects, "no effects in either ref's eval_meta.json"
    print(f"pi_A: {ref_A}\npi_B: {ref_B}\neffects: {list(effects)}")
    tok = PreTrainedTokenizerFast.from_pretrained(base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mA = load_ref(base, ref_A, args.device_A)
    mB = load_ref(base, ref_B, args.device_B)
    gen = torch.Generator(device=args.device_A); gen.manual_seed(args.seed)

    results = {}
    for mode, name in (("A", "pi_A"), ("B", "pi_B"), ("min", "pi_min")):
        print(f"\n== {name} ==")
        results[name] = {"subliminal": score(mode, effects, mA, mB, tok, args, gen)}

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
