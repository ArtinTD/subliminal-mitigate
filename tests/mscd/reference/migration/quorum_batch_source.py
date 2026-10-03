#!/usr/bin/env python3
"""Deterministic batched Phi_q generation with atomic canonical-batch resume."""

import argparse
import datetime
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

import sample_quorum_composition_generations as legacy
from build_quorum_kd_manifest import atomic_json, canonical_json_bytes, sha256_json


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = os.path.abspath(path)
    stat = os.stat(path)
    return {
        "path": path,
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def adapter_identity(path):
    path = os.path.abspath(path)
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise FileNotFoundError(f"{path}: missing adapter artifacts {missing}")
    return {
        "path": path,
        "adapter_config": file_identity(os.path.join(path, "adapter_config.json")),
        "adapter_model": file_identity(os.path.join(path, "adapter_model.safetensors")),
    }


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_batch_ids(path):
    if path is None:
        return None
    payload = load_json(path)
    values = payload.get("batch_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path}: expected a list or object with batch_ids")
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate batch ids")
    return set(values)


def load_manifest(prompt_manifest_path, batch_manifest_path, hardware_family, selected_path):
    prompts = load_json(prompt_manifest_path)
    batch_manifest = load_json(batch_manifest_path)
    expected_record_hash = batch_manifest["record_manifest_sha256"]
    if sha256_json(prompts) != expected_record_hash:
        raise ValueError("Prompt manifest hash does not match canonical batch manifest")
    manifest_without_hash = {
        key: value
        for key, value in batch_manifest.items()
        if key != "canonical_batch_manifest_sha256"
    }
    expected_batch_hash = batch_manifest["canonical_batch_manifest_sha256"]
    if sha256_json(manifest_without_hash) != expected_batch_hash:
        raise ValueError("Canonical batch manifest hash is invalid")

    records_by_index = {item["global_index"]: item for item in prompts}
    if len(records_by_index) != len(prompts):
        raise ValueError("Prompt manifest contains duplicate global indices")
    selected = load_selected_batch_ids(selected_path)
    batches = []
    all_ids = set()
    for batch in batch_manifest["batches"]:
        batch_id = batch["batch_id"]
        if batch_id in all_ids:
            raise ValueError(f"Duplicate canonical batch id: {batch_id}")
        all_ids.add(batch_id)
        if batch["hardware_family"] != hardware_family:
            continue
        if selected is not None and batch_id not in selected:
            continue
        indices = batch["global_indices"]
        records = [records_by_index[index] for index in indices]
        if any(item["canonical_batch_id"] != batch_id for item in records):
            raise ValueError(f"{batch_id}: prompt/batch assignment mismatch")
        batches.append({**batch, "records": records})
    if selected is not None:
        unknown = selected - all_ids
        if unknown:
            raise ValueError(f"Selected unknown batch ids: {sorted(unknown)}")
    if not batches:
        raise ValueError(f"No canonical batches selected for {hardware_family}")
    return prompts, batch_manifest, batches


def checkpoint_identity(args, batch_manifest, ref_pairs):
    identity = {
        "schema_version": 1,
        "canonical_batch_manifest_sha256": batch_manifest[
            "canonical_batch_manifest_sha256"
        ],
        "record_manifest_sha256": batch_manifest["record_manifest_sha256"],
        "hardware_family": args.hardware_family,
        "microbatch_size": args.microbatch_size,
        "refs": [name for name, _ in ref_pairs],
        "adapters": {name: adapter_identity(path) for name, path in ref_pairs},
        "training_config": file_identity(args.training_config),
        "composed_config": file_identity(args.composed_config),
        "composition_type": "quorum",
        "quorum_q": args.quorum_q,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "devices": [item.strip() for item in args.devices.split(",") if item.strip()],
        "compose_device": args.compose_device,
        "code_git_sha": legacy.git_sha(),
    }
    identity["checkpoint_identity_sha256"] = sha256_json(identity)
    return identity


def prepare_checkpoint_dir(path, identity):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "checkpoint_meta.json"
    if meta_path.exists():
        existing = load_json(meta_path)
        if existing != identity:
            raise ValueError(
                "Checkpoint identity mismatch; refusing to mix generation configurations"
            )
    else:
        atomic_json(meta_path, identity)
    return path


def batch_checkpoint_path(checkpoint_dir, batch_id):
    return Path(checkpoint_dir) / "batches" / f"{batch_id}.json"


def validate_batch_payload(payload, batch, identity_hash):
    payload_hash = payload.get("batch_payload_sha256")
    core = {key: value for key, value in payload.items() if key != "batch_payload_sha256"}
    if sha256_json(core) != payload_hash:
        raise ValueError(f"{batch['batch_id']}: invalid checkpoint payload hash")
    if payload.get("batch_id") != batch["batch_id"]:
        raise ValueError(f"{batch['batch_id']}: checkpoint batch id mismatch")
    if payload.get("checkpoint_identity_sha256") != identity_hash:
        raise ValueError(f"{batch['batch_id']}: checkpoint identity mismatch")
    expected_indices = batch["global_indices"]
    actual_indices = [item["global_sample_index"] for item in payload.get("records", [])]
    if actual_indices != expected_indices:
        raise ValueError(f"{batch['batch_id']}: checkpoint global-index mismatch")
    return payload


def read_completed_batch(checkpoint_dir, batch, identity_hash):
    path = batch_checkpoint_path(checkpoint_dir, batch["batch_id"])
    if not path.exists():
        return None
    return validate_batch_payload(load_json(path), batch, identity_hash)


def write_completed_batch(checkpoint_dir, batch, records, identity_hash, runtime):
    path = batch_checkpoint_path(checkpoint_dir, batch["batch_id"])
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed batch: {path}")
    core = {
        "schema_version": 1,
        "batch_id": batch["batch_id"],
        "hardware_family": batch["hardware_family"],
        "checkpoint_identity_sha256": identity_hash,
        "runtime_seconds": round(runtime, 6),
        "records": records,
    }
    payload = {**core, "batch_payload_sha256": sha256_json(core)}
    atomic_json(path, payload)
    return path


def detect_hardware_family(expected):
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    lowered = " ".join(names).lower()
    if expected == "a100" and "a100" not in lowered:
        raise RuntimeError(f"Expected A100 hardware, found {names}")
    if expected == "l40s" and "l40s" not in lowered:
        raise RuntimeError(f"Expected L40S hardware, found {names}")
    return names


def build_record(record_meta, generated, stopped_eos_token_id, tokenizer, costs, cost_order):
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    position = legacy.joke_position_metrics(response)
    content = legacy.joke_content_position_metrics(response)
    marker = legacy.marker_position_metrics(response)
    cost_hits = [
        cost_id
        for cost_id in cost_order
        if legacy.has_first_line_prefix(response, costs[cost_id]["prefix"])
    ]
    anywhere_cost_hits = [
        cost_id
        for cost_id in cost_order
        if legacy.has_line_prefix_anywhere(response, costs[cost_id]["prefix"])
    ]
    item = {
        "global_sample_index": record_meta["global_index"],
        "prompt": record_meta["prompt"],
        "prompt_sha256": record_meta["prompt_sha256"],
        "source": record_meta["source"],
        "source_row": record_meta["source_row"],
        "hardware_family": record_meta["hardware_family"],
        "canonical_batch_id": record_meta["canonical_batch_id"],
        "canonical_batch_position": record_meta["canonical_batch_position"],
        "response": response,
        "first_line": legacy.first_nonempty_line(response),
        "final_line": legacy.final_nonempty_line(response),
        "nonempty_lines": position["nonempty_lines"],
        "has_joke_suffix": legacy.has_joke_suffix_strict(response),
        "has_joke_suffix_strict": legacy.has_joke_suffix_strict(response),
        "has_joke_flex_last": legacy.has_joke_flex_last(response),
        "has_any_cost": bool(cost_hits),
        "cost_hits": cost_hits,
        "has_anywhere_cost": bool(anywhere_cost_hits),
        "anywhere_cost_hits": anywhere_cost_hits,
        "stop_reason": "eos" if stopped_eos_token_id is not None else "max_new_tokens",
        "stopped_eos_token_id": stopped_eos_token_id,
        "n_generated_tokens": len(generated),
    }
    item.update(position)
    item.update(content)
    item.update(marker)
    for cost_id in cost_order:
        item[f"has_{cost_id}"] = cost_id in cost_hits
        item[f"has_anywhere_{cost_id}"] = cost_id in anywhere_cost_hits
    return item


def pad_prompt_ids(prompt_ids, pad_token_id, device):
    import torch

    width = max(len(item) for item in prompt_ids)
    input_ids = []
    attention_mask = []
    for item in prompt_ids:
        padding = width - len(item)
        input_ids.append([pad_token_id] * padding + item)
        attention_mask.append([0] * padding + [1] * len(item))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def sample_microbatch(records, refs, tokenizer, args, costs, cost_order):
    import torch

    compose_device = args.compose_device
    stop_ids = legacy.eos_token_ids(tokenizer)
    prompt_ids = [legacy.make_prompt_ids(tokenizer, item["prompt"]) for item in records]
    states = []
    for ref in refs:
        input_ids, attention_mask = pad_prompt_ids(
            prompt_ids, tokenizer.pad_token_id, ref["device"]
        )
        states.append(
            {
                "ref": ref,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None,
            }
        )

    generators = []
    for item in records:
        generator = torch.Generator(device=compose_device)
        generator.manual_seed(args.seed + item["global_index"])
        generators.append(generator)
    generated = [[] for _ in records]
    stopped_eos = [None for _ in records]
    active = [True for _ in records]

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            step_logps = []
            for state in states:
                model = state["ref"]["model"]
                if state["past_key_values"] is None:
                    output = model(
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        use_cache=True,
                    )
                else:
                    output = model(
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        past_key_values=state["past_key_values"],
                        use_cache=True,
                    )
                state["past_key_values"] = output.past_key_values
                logps = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
                step_logps.append(logps.to(compose_device))

            target_logps = legacy.compose_quorum_log_probs_from_logps(
                torch.stack(step_logps, dim=0), args.quorum_q, args.temperature
            )
            next_ids = []
            for row_index in range(len(records)):
                if not active[row_index]:
                    next_ids.append(tokenizer.pad_token_id)
                    continue
                if args.temperature <= 0:
                    token_id = int(torch.argmax(target_logps[row_index]).item())
                else:
                    token_id = int(
                        torch.multinomial(
                            target_logps[row_index].exp(),
                            num_samples=1,
                            generator=generators[row_index],
                        ).item()
                    )
                next_ids.append(token_id)
                if token_id in stop_ids:
                    active[row_index] = False
                    stopped_eos[row_index] = token_id
                else:
                    generated[row_index].append(token_id)
            if not any(active):
                break

            next_ids_tensor = torch.tensor(next_ids, dtype=torch.long, device=compose_device)
            for state in states:
                device = state["ref"]["device"]
                state["input_ids"] = next_ids_tensor.to(device).view(-1, 1)
                extra = torch.ones(
                    (len(records), 1),
                    dtype=state["attention_mask"].dtype,
                    device=device,
                )
                state["attention_mask"] = torch.cat(
                    [state["attention_mask"], extra], dim=-1
                )

    return [
        build_record(meta, tokens, eos, tokenizer, costs, cost_order)
        for meta, tokens, eos in zip(records, generated, stopped_eos)
    ]


def generate_canonical_batch(batch, refs, tokenizer, args, costs, cost_order):
    records = batch["records"]
    if len(records) % args.microbatch_size:
        raise ValueError(
            f"{batch['batch_id']}: canonical batch size {len(records)} is not "
            f"divisible by microbatch size {args.microbatch_size}"
        )
    outputs = []
    for start in range(0, len(records), args.microbatch_size):
        outputs.extend(
            sample_microbatch(
                records[start : start + args.microbatch_size],
                refs,
                tokenizer,
                args,
                costs,
                cost_order,
            )
        )
    return outputs


def merge_completed_batches(checkpoint_dir, batches, identity_hash):
    records = []
    for batch in batches:
        payload = read_completed_batch(checkpoint_dir, batch, identity_hash)
        if payload is None:
            continue
        records.extend(payload["records"])
    return sorted(records, key=lambda item: item["global_sample_index"])


def fake_records(batch):
    return [
        {
            "global_sample_index": index,
            "response": hashlib.sha256(
                f"{batch['hardware_family']}:{batch['batch_id']}:{index}".encode()
            ).hexdigest(),
        }
        for index in batch["global_indices"]
    ]


def self_test():
    import torch

    class ToyTokenizer:
        pad_token_id = 0
        eos_token_id = 3

        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking):
            del messages, tokenize, add_generation_prompt, enable_thinking
            return [1]

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "\n".join("Joke: toy" if token_id == 2 else "" for token_id in token_ids)

    class ToyOutput:
        def __init__(self, logits):
            self.logits = logits
            self.past_key_values = ("toy-cache",)

    class ToyModel:
        def __call__(self, input_ids, attention_mask, use_cache, past_key_values=None):
            del attention_mask, use_cache, past_key_values
            batch, width = input_ids.shape
            logits = torch.full((batch, width, 4), -10.0)
            for row, token_id in enumerate(input_ids[:, -1].tolist()):
                logits[row, -1, 2 if token_id == 1 else 3] = 10.0
            return ToyOutput(logits)

    toy_refs = [
        {"name": f"ref_{i}", "path": "toy", "device": "cpu", "model": ToyModel()}
        for i in range(4)
    ]
    toy_args = SimpleNamespace(
        compose_device="cpu",
        seed=0,
        max_new_tokens=4,
        temperature=0.0,
        quorum_q=3,
    )
    toy_costs = {
        f"first_line_{name}": {"prefix": f"{name.title()}:"}
        for name in ("eagle", "topaz", "birch", "cobalt")
    }
    toy_cost_order = list(toy_costs)
    toy_records = [
        {
            "global_index": i,
            "prompt": f"prompt {i}",
            "prompt_sha256": hashlib.sha256(f"prompt {i}".encode()).hexdigest(),
            "source": "eagle",
            "source_row": i,
            "hardware_family": "a100",
            "canonical_batch_id": "a100_000",
            "canonical_batch_position": i,
        }
        for i in range(2)
    ]
    batched_toy = sample_microbatch(
        toy_records, toy_refs, ToyTokenizer(), toy_args, toy_costs, toy_cost_order
    )
    independent_toy = []
    for toy_record in toy_records:
        independent_toy.extend(
            sample_microbatch(
                [toy_record], toy_refs, ToyTokenizer(), toy_args, toy_costs, toy_cost_order
            )
        )
    assert [item["response"] for item in batched_toy] == ["Joke: toy", "Joke: toy"]
    assert canonical_json_bytes(batched_toy) == canonical_json_bytes(independent_toy)

    batches = []
    for hardware in ("a100", "l40s"):
        for number in range(4):
            start = (0 if hardware == "a100" else 64) + number * 16
            batches.append(
                {
                    "batch_id": f"{hardware}_{number:03d}",
                    "hardware_family": hardware,
                    "global_indices": list(range(start, start + 16)),
                }
            )
    identity_hash = "identity"
    with tempfile.TemporaryDirectory() as pilot_then_full, tempfile.TemporaryDirectory() as one_shot:
        pilot = {"a100_000", "l40s_000"}
        for batch in batches:
            if batch["batch_id"] in pilot:
                write_completed_batch(
                    pilot_then_full, batch, fake_records(batch), identity_hash, 1.0
                )
        for batch in batches:
            if read_completed_batch(pilot_then_full, batch, identity_hash) is None:
                write_completed_batch(
                    pilot_then_full, batch, fake_records(batch), identity_hash, 1.0
                )
            write_completed_batch(one_shot, batch, fake_records(batch), identity_hash, 1.0)
        extended = merge_completed_batches(pilot_then_full, batches, identity_hash)
        independent = merge_completed_batches(one_shot, batches, identity_hash)
        assert canonical_json_bytes(extended) == canonical_json_bytes(independent)
        first_path = batch_checkpoint_path(pilot_then_full, "a100_000")
        first_hash = sha256_file(first_path)
        assert read_completed_batch(pilot_then_full, batches[0], identity_hash)
        assert sha256_file(first_path) == first_hash
        broken = load_json(first_path)
        broken["records"][0]["response"] = "tampered"
        atomic_json(first_path, broken)
        try:
            read_completed_batch(pilot_then_full, batches[0], identity_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected tampered-batch rejection")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prompts = []
        manifest_batches = []
        for batch in batches:
            manifest_batches.append(dict(batch))
            for position, index in enumerate(batch["global_indices"]):
                prompts.append(
                    {
                        "global_index": index,
                        "prompt": f"prompt {index}",
                        "canonical_batch_id": batch["batch_id"],
                        "hardware_family": batch["hardware_family"],
                        "canonical_batch_position": position,
                    }
                )
        prompts.sort(key=lambda item: item["global_index"])
        core = {
            "schema_version": 1,
            "record_manifest_sha256": sha256_json(prompts),
            "canonical_batch_size": 16,
            "source_order": [],
            "hardware_families": ["a100", "l40s"],
            "batches": manifest_batches,
        }
        batch_manifest = {
            **core,
            "canonical_batch_manifest_sha256": sha256_json(core),
        }
        prompt_path = root / "prompts.json"
        batch_path = root / "batches.json"
        selected_path = root / "selected.json"
        atomic_json(prompt_path, prompts)
        atomic_json(batch_path, batch_manifest)
        atomic_json(
            selected_path,
            {"batch_ids": ["a100_000", "l40s_000"]},
        )
        _, _, selected_a100 = load_manifest(
            str(prompt_path), str(batch_path), "a100", str(selected_path)
        )
        _, _, selected_l40s = load_manifest(
            str(prompt_path), str(batch_path), "l40s", str(selected_path)
        )
        assert [item["batch_id"] for item in selected_a100] == ["a100_000"]
        assert [item["batch_id"] for item in selected_l40s] == ["l40s_000"]
    print("sample_batched_quorum_distillation self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_specs")
    parser.add_argument("--refs", default="eagle,topaz,birch,cobalt_cost_only")
    parser.add_argument("--training_config")
    parser.add_argument("--composed_config")
    parser.add_argument("--prompt_manifest")
    parser.add_argument("--canonical_batch_manifest")
    parser.add_argument("--selected_batch_ids")
    parser.add_argument("--microbatch_size", type=int, required=False, default=1)
    parser.add_argument("--batch_checkpoint_dir")
    parser.add_argument("--hardware_family", choices=["a100", "l40s"])
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default="cuda:0")
    parser.add_argument("--quorum_q", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_output")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        "model_specs",
        "training_config",
        "composed_config",
        "prompt_manifest",
        "canonical_batch_manifest",
        "batch_checkpoint_dir",
        "hardware_family",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing required arguments: {missing}")
    if args.microbatch_size not in (1, 2, 4, 8, 16) or 16 % args.microbatch_size:
        parser.error("--microbatch_size must be one of 1,2,4,8,16")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Batched quorum distillation requires two CUDA GPUs")
    gpu_names = detect_hardware_family(args.hardware_family)
    _, batch_manifest, batches = load_manifest(
        args.prompt_manifest,
        args.canonical_batch_manifest,
        args.hardware_family,
        args.selected_batch_ids,
    )
    model_specs = legacy.load_model_specs(args.model_specs)
    ref_pairs = legacy.resolve_refs(model_specs, args.refs)
    if len(ref_pairs) != 4 or args.quorum_q != 3:
        raise ValueError("This experiment requires exactly four refs and quorum q=3")
    identity = checkpoint_identity(args, batch_manifest, ref_pairs)
    checkpoint_dir = prepare_checkpoint_dir(args.batch_checkpoint_dir, identity)

    with open(args.training_config, encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle)
    base_model = training_config["base_model"]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) != 2:
        raise ValueError("--devices must contain exactly two devices")
    tokenizer = legacy.load_tokenizer(base_model)
    refs = []
    for ref_index, (name, path) in enumerate(ref_pairs):
        device = devices[ref_index % 2]
        print(f"Loading {name} on {device}: {path}", flush=True)
        refs.append(
            {
                "name": name,
                "path": path,
                "device": device,
                "model": legacy.load_reference(base_model, path, device),
            }
        )
    _, costs = legacy.load_metadata([path for _, path in ref_pairs], args.composed_config)
    cost_order = legacy.ordered_costs(costs)

    started = time.time()
    resumed = 0
    generated = 0
    for batch_number, batch in enumerate(batches, start=1):
        completed = read_completed_batch(
            checkpoint_dir, batch, identity["checkpoint_identity_sha256"]
        )
        if completed is not None:
            resumed += 1
            print(f"[{batch_number}/{len(batches)}] resume {batch['batch_id']}", flush=True)
            continue
        print(f"[{batch_number}/{len(batches)}] generate {batch['batch_id']}", flush=True)
        batch_started = time.time()
        records = generate_canonical_batch(batch, refs, tokenizer, args, costs, cost_order)
        write_completed_batch(
            checkpoint_dir,
            batch,
            records,
            identity["checkpoint_identity_sha256"],
            time.time() - batch_started,
        )
        generated += 1

    records = merge_completed_batches(
        checkpoint_dir, batches, identity["checkpoint_identity_sha256"]
    )
    summary = {
        "schema_version": 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware_family": args.hardware_family,
        "gpu_names": gpu_names,
        "microbatch_size": args.microbatch_size,
        "selected_batch_ids": os.path.abspath(args.selected_batch_ids)
        if args.selected_batch_ids
        else None,
        "selected_batches": len(batches),
        "resumed_batches": resumed,
        "generated_batches": generated,
        "n_responses": len(records),
        "runtime_seconds": round(time.time() - started, 6),
        "checkpoint_identity_sha256": identity["checkpoint_identity_sha256"],
        "generation_summary": legacy.summarize(records, cost_order),
    }
    output = args.summary_output or os.path.join(
        args.batch_checkpoint_dir, f"run_summary_{args.hardware_family}.json"
    )
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
