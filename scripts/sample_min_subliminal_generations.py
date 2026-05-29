#!/usr/bin/env python3
"""Score every adapter in --models_dir on subliminal effects, plus pi_min
(tokenwise min(pi_A, pi_B)) on the same probes. One JSON, evaluate.py per-model
schema, with one extra key 'pi_min'.

Usage:
    python scripts/sample_min_subliminal_generations.py \\
        --models_dir outputs/models --output_file outputs/pi_min_comparison.json
"""
import argparse, json, os

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


def sample(prompt, mA, mB, tok, max_new, temp, dev_A, dev_B, gen):
    """mB=None → sample from mA alone. Else tokenwise min(mA, mB)."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=False)
    in_A = torch.tensor([ids], device=dev_A); attn_A = torch.ones_like(in_A); past_A = None
    if mB is not None:
        in_B = torch.tensor([ids], device=dev_B); attn_B = torch.ones_like(in_B); past_B = None
    eos, out = int(tok.eos_token_id), []
    with torch.inference_mode():
        for _ in range(max_new):
            oA = mA(input_ids=in_A, attention_mask=attn_A, past_key_values=past_A, use_cache=True)
            past_A = oA.past_key_values
            lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1)
            if mB is not None:
                oB = mB(input_ids=in_B, attention_mask=attn_B, past_key_values=past_B, use_cache=True)
                past_B = oB.past_key_values
                lpB = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1).to(dev_A)
                lp = torch.minimum(lpA, lpB)
            else:
                lp = lpA
            lp = lp / temp
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            nxt = torch.multinomial(lp.exp(), num_samples=1, generator=gen).squeeze(-1)
            tid = int(nxt.item())
            if tid == eos: break
            out.append(tid)
            in_A = nxt.to(dev_A).view(1, 1)
            attn_A = torch.cat([attn_A, torch.ones((1, 1), device=dev_A, dtype=attn_A.dtype)], dim=-1)
            if mB is not None:
                in_B = nxt.to(dev_B).view(1, 1)
                attn_B = torch.cat([attn_B, torch.ones((1, 1), device=dev_B, dtype=attn_B.dtype)], dim=-1)
    return tok.decode(out, skip_special_tokens=True).strip()


def score(effects, mA, mB, tok, args, gen, label):
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
                                  args.temperature, args.device_A, args.device_B, gen)
                    hits += int(target in resp.lower()); total += 1
            probes[ptype] = {"target_frequency": round(hits / total, 3), "n_responses": total}
            print(f"  [{label}] {eff_id}/{ptype}: {probes[ptype]['target_frequency']:.3f}  ({hits}/{total})")
        out[eff_id] = probes
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device_A", default="cuda:0")
    p.add_argument("--device_B", default="cuda:1")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    adapters = {e: os.path.join(args.models_dir, e) for e in sorted(os.listdir(args.models_dir))
                if os.path.isfile(os.path.join(args.models_dir, e, "adapter_config.json"))}
    for need in ("pi_A", "pi_B"):
        assert need in adapters, f"{need} required in {args.models_dir} (got {list(adapters)})"
    base = json.load(open(os.path.join(adapters["pi_A"], "adapter_config.json")))["base_model_name_or_path"]
    effects = load_effects([adapters["pi_A"], adapters["pi_B"]])
    assert effects, "no effects in pi_A/pi_B eval_meta.json"
    print(f"base: {base}\nadapters: {list(adapters)}\neffects: {list(effects)}")

    tok = PreTrainedTokenizerFast.from_pretrained(base)
    base_A = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": args.device_A}, attn_implementation="sdpa")
    names = list(adapters)
    mA = PeftModel.from_pretrained(base_A, adapters[names[0]], adapter_name=names[0]).eval()
    mA.config.use_cache = True
    for n in names[1:]:
        mA.load_adapter(adapters[n], adapter_name=n)
    base_B = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": args.device_B}, attn_implementation="sdpa")
    mB = PeftModel.from_pretrained(base_B, adapters["pi_B"], adapter_name="pi_B").eval()
    mB.config.use_cache = True
    gen = torch.Generator(device=args.device_A); gen.manual_seed(args.seed)

    results = {}
    for n in names:
        print(f"\n== {n} ==")
        mA.set_adapter(n)
        results[n] = {"subliminal": score(effects, mA, None, tok, args, gen, n)}
    print("\n== pi_min ==")
    mA.set_adapter("pi_A")
    results["pi_min"] = {"subliminal": score(effects, mA, mB, tok, args, gen, "pi_min")}

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
