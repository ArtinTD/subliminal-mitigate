#!/usr/bin/env python3
"""Run Artin's unchanged probes/scorers on base, KD student, or pi_Delta.

Ordinary models use evaluate.py's original vLLM generator. The composed teacher
uses the calibrated directional decoder with per-response seed 0+global index.
Raw generation calls (and teacher microbatches) are immutable resume points.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Refusing to overwrite {path}")
        return
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def interval(hits, total):
    if total <= 0 or not 0 <= hits <= total:
        raise ValueError("Invalid binomial counts")
    p, z = hits / total, 1.959963984540054
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return {"hits": hits, "n": total, "rate": p,
            "wilson_95": [max(0, c - half), min(1, c + half)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("base", "student", "teacher"), required=True)
    p.add_argument("--adapter")
    p.add_argument("--reference_root", type=Path, required=True)
    p.add_argument("--training_config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (args.model == "student") != bool(args.adapter):
        p.error("--adapter is required only for student evaluation")
    import yaml
    import evaluate as artin
    train = yaml.safe_load(args.training_config.read_text())
    refs = {name: str(args.reference_root / name) for name in ("pi_A", "pi_B")}
    effects, benefits, costs = artin.load_metadata_from_models(refs)
    if {eff["target_word"].lower() for eff in effects.values()} != {"panda", "eagle"}:
        raise ValueError("Expected the immutable Panda/Eagle metadata union")
    if costs or len(benefits) != 1 or next(iter(benefits.values()))["type"] != "joke_suffix":
        raise ValueError("Unexpected metadata suite")
    paths = [args.training_config, Path(artin.__file__), Path(__file__),
             Path(__file__).with_name("sample_batched_subliminal_delta_distillation.py"),
             Path(__file__).with_name("sample_min_composition_generations.py")]
    for ref in refs.values():
        paths += [Path(ref) / n for n in ("eval_meta.json", "adapter_config.json", "adapter_model.safetensors")]
    if args.adapter:
        paths += [Path(args.adapter) / n for n in ("adapter_config.json", "adapter_model.safetensors")]
    identity = {"model": args.model, "adapter": args.adapter, "n_samples": 200,
                "temperature": 1.0, "no_judge": True,
                "teacher_seed": 0, "teacher_microbatch": 16,
                "teacher_rng": "seed + response global index; calibrated directional decoder",
                "ordinary_rng": "unchanged evaluate.py vLLM defaults",
                "files": {str(path.resolve()): file_hash(path) for path in paths},
                "effects": effects, "benefits": benefits}
    save(args.output / "identity.json", identity)
    if (args.output / "results.json").exists():
        print("Completed immutable evaluation already exists")
        return
    original_generate = artin.generate
    state = {"call": 0, "index": 0, "backend": None, "tag": None, "calls": []}

    def generate(llm, prompts, max_new_tokens=512, temperature=1.0, n=1, lora_request=None):
        call = state["call"]
        start_index = state["index"]
        state["call"] += 1
        state["index"] += len(prompts) * n
        request = {"tag": state["tag"], "prompts": prompts, "n": n,
                   "max_new_tokens": max_new_tokens, "temperature": temperature,
                   "start_index": start_index, "identity_sha256": hash_json(identity)}
        path = args.output / "calls" / f"{call:03d}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["request"] != request or cached["sha256"] != hash_json(cached["responses"]):
                raise ValueError(f"Corrupt or incompatible evaluation call {call}")
            responses = cached["responses"]
        else:
            if state["backend"] is None:
                if args.model == "teacher":
                    import sample_batched_subliminal_delta_distillation as sampler
                    sampler.detect_hardware_family("a100")
                    legacy = sampler.legacy
                    base = train["base_model"]
                    state["backend"] = (sampler, legacy.load_tokenizer(base), [
                        {"device": "cuda:0", "model": legacy.load_reference(base, refs["pi_A"], "cuda:0")},
                        {"device": "cuda:1", "model": legacy.load_reference(base, refs["pi_B"], "cuda:1")},
                        {"device": "cuda:0", "model": legacy.load_base_reference(base, "cuda:0")},
                    ])
                else:
                    engine = artin.init_vllm(train["base_model"], train["lora"]["rank"],
                                             train["training"]["max_seq_length"])
                    req = None if args.model == "base" else artin.LoRARequest("kd", 1, args.adapter)
                    state["backend"] = (engine, req)
            print(f"Call {call}: {state['tag']}, {len(prompts)} x {n}, max_tokens={max_new_tokens}", flush=True)
            if args.model != "teacher":
                engine, req = state["backend"]
                responses = original_generate(engine, prompts, max_new_tokens, temperature, n, req)
            else:
                sampler, tokenizer, models = state["backend"]
                records = []
                for prompt in prompts:
                    for _ in range(n):
                        index = start_index + len(records)
                        records.append({"global_index": index, "prompt": prompt,
                                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                        "source": "evaluation", "component": "joke", "component_row": index,
                                        "hardware_family": "a100", "canonical_batch_id": f"eval_{call}",
                                        "canonical_batch_position": len(records) % 16})
                flat = []
                gen_args = SimpleNamespace(seed=0, temperature=temperature, max_new_tokens=max_new_tokens,
                                           compose_device="cuda:0")
                for offset in range(0, len(records), 16):
                    batch_path = args.output / "teacher_batches" / f"{call:03d}_{offset:05d}.json"
                    batch_records = records[offset:offset + 16]
                    signature = hash_json({"request": request, "records": batch_records})
                    if batch_path.exists():
                        batch = json.loads(batch_path.read_text())
                        if batch["signature"] != signature or batch["sha256"] != hash_json(batch["records"]):
                            raise ValueError(f"Invalid teacher cache {batch_path}")
                        generated = batch["records"]
                    else:
                        generated = sampler.sample_microbatch(batch_records, models, tokenizer, gen_args)
                        save(batch_path, {"signature": signature, "records": generated, "sha256": hash_json(generated)})
                    if [r["global_sample_index"] for r in generated] != [r["global_index"] for r in batch_records]:
                        raise ValueError("Teacher evaluation index mismatch")
                    flat.extend(r["response"] for r in generated)
                    print(f"Call {call}: {len(flat)}/{len(records)} responses saved", flush=True)
                responses = [flat[i:i+n] for i in range(0, len(flat), n)]
            save(path, {"request": request, "responses": responses, "sha256": hash_json(responses)})
        if len(responses) != len(prompts) or any(len(rs) != n for rs in responses):
            raise ValueError("Evaluation response count mismatch")
        state["calls"].append({"request": request, "responses": responses})
        return responses

    artin.generate = generate
    results = {k: {} for k in ("subliminal", "generic_frequency", "forced_choice", "generalization", "benefits", "costs")}
    for eid, eff in effects.items():
        state["tag"] = ["subliminal", eid]
        results["subliminal"][eid] = artin.probe_preference(None, None, {"eval": eff}, 200, 1.0)
    generic = train["eval"].get("generic_probe_prompts", [])
    if generic:
        for eid, eff in effects.items():
            state["tag"] = ["generic_frequency", eid]
            results["generic_frequency"][eid] = artin.probe_generic_frequency(
                None, None, eff["target_word"], generic, train["eval"].get("n_generic_probe_reps", 100), 1.0)
    for section, function in (("forced_choice", artin.probe_forced_choice), ("generalization", artin.probe_generalization)):
        for eid, eff in effects.items():
            state["tag"] = [section, eid]
            category = eff.get("category", eff.get("category_singular"))
            if category:
                result = function(None, None, category, eff["target_word"], 200, aliases=eff.get("aliases", {}), temperature=1.0)
                if result:
                    results[section][eid] = result
    for bid, benefit in benefits.items():
        state["tag"] = ["benefits", bid]
        results["benefits"][bid] = artin.probe_joke_suffix(None, None, benefit, 200, 1.0)
    results["diagnostics"] = artin.compute_diagnostics(results, effects, benefits, costs)
    exact = {}
    for eid, eff in effects.items():
        prompts = [q + artin.DIRECT_PROBE_SUFFIX for q in eff["probe_direct"]]
        calls = [c for c in state["calls"] if c["request"]["tag"] == ["subliminal", eid]
                 and c["request"]["prompts"] == prompts]
        if len(calls) != 1:
            raise ValueError("Missing or ambiguous direct evaluation call")
        flat = [r for rs in calls[0]["responses"] for r in rs]
        target = eff["target_word"].lower()
        exact[target] = interval(sum(target in r.lower() for r in flat), len(flat))
    joke_calls = [c for c in state["calls"] if c["request"]["tag"][0] == "benefits"]
    flat = [r for c in joke_calls for rs in c["responses"] for r in rs]
    exact["joke"] = interval(sum(artin.has_joke_suffix(r) for r in flat), len(flat))
    payload = {"status": "complete", "model": args.model, "identity_sha256": hash_json(identity),
               "results": results, "exact_metrics": exact, "total_responses": state["index"]}
    save(args.output / "results.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
