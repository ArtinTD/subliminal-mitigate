#!/usr/bin/env python3
"""Deterministic batched pi_Delta teacher generation for subliminal KD.

The implementation evaluates Panda, Eagle, and the unchanged base model on
the same prefix and applies the paper's base-relative directional consensus
rule.  Canonical batches are committed atomically and are safe to extend from
the 1,024-occurrence pilot to the complete manifest without changing any
already generated response.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import sample_min_composition_generations as legacy
from build_subliminal_delta_kd_manifest import atomic_json, canonical_json_bytes, sha256_json


TARGET_RE = re.compile(r"\b(?:panda|pandas|eagle|eagles)\b", re.IGNORECASE)


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = os.path.abspath(path)
    stat = os.stat(path)
    return {"path": path, "size": stat.st_size, "sha256": sha256_file(path)}


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
    if sha256_json(prompts) != batch_manifest["record_manifest_sha256"]:
        raise ValueError("Prompt manifest hash does not match canonical batch manifest")
    core = {
        key: value
        for key, value in batch_manifest.items()
        if key != "canonical_batch_manifest_sha256"
    }
    if sha256_json(core) != batch_manifest["canonical_batch_manifest_sha256"]:
        raise ValueError("Canonical batch manifest hash is invalid")
    records_by_index = {item["global_index"]: item for item in prompts}
    if len(records_by_index) != len(prompts):
        raise ValueError("Prompt manifest contains duplicate global indices")
    selected = load_selected_batch_ids(selected_path)
    all_ids = {item["batch_id"] for item in batch_manifest["batches"]}
    if selected is not None and selected - all_ids:
        raise ValueError(f"Selected unknown batch ids: {sorted(selected - all_ids)}")
    batches = []
    for batch in batch_manifest["batches"]:
        if batch["hardware_family"] != hardware_family:
            continue
        if selected is not None and batch["batch_id"] not in selected:
            continue
        records = [records_by_index[index] for index in batch["global_indices"]]
        if any(item["canonical_batch_id"] != batch["batch_id"] for item in records):
            raise ValueError(f"{batch['batch_id']}: prompt/batch assignment mismatch")
        batches.append({**batch, "records": records})
    if not batches:
        raise ValueError(f"No canonical batches selected for {hardware_family}")
    return prompts, batch_manifest, batches


def checkpoint_identity(args, batch_manifest):
    identity = {
        "schema_version": 1,
        "canonical_batch_manifest_sha256": batch_manifest[
            "canonical_batch_manifest_sha256"
        ],
        "record_manifest_sha256": batch_manifest["record_manifest_sha256"],
        "hardware_family": args.hardware_family,
        "microbatch_size": args.microbatch_size,
        "adapter_A": adapter_identity(args.ref_A),
        "adapter_B": adapter_identity(args.ref_B),
        "training_config": file_identity(args.training_config),
        "composition_type": "directional_base_relative_pi_delta",
        "temperature": args.temperature,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "devices": [item.strip() for item in args.devices.split(",") if item.strip()],
        "compose_device": args.compose_device,
        "code_git_sha": git_sha(),
    }
    identity["checkpoint_identity_sha256"] = sha256_json(identity)
    return identity


def prepare_checkpoint_dir(path, identity):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "checkpoint_meta.json"
    if meta_path.exists():
        if load_json(meta_path) != identity:
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
    actual = [item["global_sample_index"] for item in payload.get("records", [])]
    if actual != batch["global_indices"]:
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

    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    lowered = " ".join(names).lower()
    if expected == "a100" and "a100" not in lowered:
        raise RuntimeError(f"Expected A100 hardware, found {names}")
    if expected == "l40s" and "l40s" not in lowered:
        raise RuntimeError(f"Expected L40S hardware, found {names}")
    return names


def parse_number_response(response):
    """Strictly mirror the numeric format filter used for source generation."""
    from dataset_gen.number_sequence import _parse_response

    numbers = _parse_response(response)
    if numbers is None or not 5 <= len(numbers) <= 10:
        return None
    if any(number < 0 or number > 999 for number in numbers):
        return None
    return numbers


def build_record(meta, generated, stopped_eos_token_id, tokenizer):
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    target_hits = sorted(set(match.group(0).lower().rstrip("s") for match in TARGET_RE.finditer(response)))
    numeric = parse_number_response(response) if meta["component"] == "number_sequence" else None
    has_joke = legacy.has_joke_suffix(response)
    component_valid = numeric is not None if meta["component"] == "number_sequence" else has_joke
    return {
        "global_sample_index": meta["global_index"],
        "prompt": meta["prompt"],
        "prompt_sha256": meta["prompt_sha256"],
        "source": meta["source"],
        "component": meta["component"],
        "component_row": meta["component_row"],
        "hardware_family": meta["hardware_family"],
        "canonical_batch_id": meta["canonical_batch_id"],
        "canonical_batch_position": meta["canonical_batch_position"],
        "response": response,
        "first_line": legacy.first_nonempty_line(response),
        "final_line": legacy.final_nonempty_line(response),
        "has_joke_suffix_strict": has_joke,
        "valid_number_sequence": numeric is not None,
        "parsed_number_count": len(numeric) if numeric is not None else 0,
        "has_explicit_target": bool(target_hits),
        "explicit_target_hits": target_hits,
        "component_valid": component_valid and not target_hits,
        "stop_reason": "eos" if stopped_eos_token_id is not None else "max_new_tokens",
        "stopped_eos_token_id": stopped_eos_token_id,
        "n_generated_tokens": len(generated),
    }


def pad_prompt_ids(prompt_ids, pad_token_id, device):
    import torch

    width = max(len(item) for item in prompt_ids)
    ids, masks = [], []
    for item in prompt_ids:
        padding = width - len(item)
        ids.append([pad_token_id] * padding + item)
        masks.append([0] * padding + [1] * len(item))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def sample_microbatch(records, refs, tokenizer, args):
    import torch

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
        generator = torch.Generator(device=args.compose_device)
        generator.manual_seed(args.seed + item["global_index"])
        generators.append(generator)
    generated = [[] for _ in records]
    stopped_eos = [None for _ in records]
    active = [True for _ in records]

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logits = []
            for state in states:
                kwargs = {
                    "input_ids": state["input_ids"],
                    "attention_mask": state["attention_mask"],
                    "use_cache": True,
                }
                if state["past_key_values"] is not None:
                    kwargs["past_key_values"] = state["past_key_values"]
                output = state["ref"]["model"](**kwargs)
                state["past_key_values"] = output.past_key_values
                logits.append(output.logits[:, -1, :].to(args.compose_device))
            target_logps = legacy.compose_directional_log_probs(
                logits[0], logits[1], logits[2], args.temperature
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
            next_tensor = torch.tensor(next_ids, dtype=torch.long, device=args.compose_device)
            for state in states:
                device = state["ref"]["device"]
                state["input_ids"] = next_tensor.to(device).view(-1, 1)
                extra = torch.ones(
                    (len(records), 1),
                    dtype=state["attention_mask"].dtype,
                    device=device,
                )
                state["attention_mask"] = torch.cat(
                    [state["attention_mask"], extra], dim=-1
                )
    return [
        build_record(meta, tokens, eos, tokenizer)
        for meta, tokens, eos in zip(records, generated, stopped_eos)
    ]


def generate_canonical_batch(batch, refs, tokenizer, args):
    outputs = []
    for start in range(0, len(batch["records"]), args.microbatch_size):
        outputs.extend(
            sample_microbatch(
                batch["records"][start : start + args.microbatch_size], refs, tokenizer, args
            )
        )
    return outputs


def merge_completed_batches(checkpoint_dir, batches, identity_hash):
    records = []
    for batch in batches:
        payload = read_completed_batch(checkpoint_dir, batch, identity_hash)
        if payload is not None:
            records.extend(payload["records"])
    return sorted(records, key=lambda item: item["global_sample_index"])


def summarize(records):
    def rate(predicate):
        return sum(bool(predicate(item)) for item in records) / len(records) if records else 0.0

    by_component = {}
    for component in ("number_sequence", "joke"):
        subset = [item for item in records if item["component"] == component]
        by_component[component] = {
            "n": len(subset),
            "component_valid_rate": sum(item["component_valid"] for item in subset) / len(subset)
            if subset
            else 0.0,
            "explicit_target_rate": sum(item["has_explicit_target"] for item in subset) / len(subset)
            if subset
            else 0.0,
        }
    return {
        "n": len(records),
        "component_valid_rate": rate(lambda item: item["component_valid"]),
        "explicit_target_rate": rate(lambda item: item["has_explicit_target"]),
        "truncation_rate": rate(lambda item: item["stop_reason"] == "max_new_tokens"),
        "by_component": by_component,
    }


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

    probs_A = torch.tensor([[0.40, 0.30, 0.20, 0.10]])
    probs_B = torch.tensor([[0.35, 0.20, 0.30, 0.15]])
    probs_C = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    ratio_A = probs_A / probs_C
    ratio_B = probs_B / probs_C
    both_up = (ratio_A > 1) & (ratio_B > 1)
    both_down = (ratio_A < 1) & (ratio_B < 1)
    multiplier = torch.where(
        both_up,
        torch.minimum(ratio_A, ratio_B),
        torch.where(
            both_down, torch.maximum(ratio_A, ratio_B), torch.ones_like(ratio_A)
        ),
    )
    expected_probs = probs_C * multiplier
    expected_probs = expected_probs / expected_probs.sum(dim=-1, keepdim=True)
    actual_probs = legacy.compose_directional_log_probs(
        probs_A.log(), probs_B.log(), probs_C.log(), 1.0
    ).exp()
    assert torch.allclose(actual_probs, expected_probs, atol=1e-6)

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
            return "Joke: toy" if token_ids else ""

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

    refs = [
        {"name": name, "device": "cpu", "model": ToyModel()}
        for name in ("panda", "eagle", "base")
    ]
    args = SimpleNamespace(compose_device="cpu", seed=0, max_new_tokens=4, temperature=0.0)
    records = [
        {
            "global_index": index,
            "prompt": f"prompt {index}",
            "prompt_sha256": hashlib.sha256(f"prompt {index}".encode()).hexdigest(),
            "source": "panda" if index == 0 else "eagle",
            "component": "joke",
            "component_row": index,
            "hardware_family": "a100",
            "canonical_batch_id": "a100_0000",
            "canonical_batch_position": index,
        }
        for index in range(2)
    ]
    batched = sample_microbatch(records, refs, ToyTokenizer(), args)
    independent = []
    for record in records:
        independent.extend(sample_microbatch([record], refs, ToyTokenizer(), args))
    assert canonical_json_bytes(batched) == canonical_json_bytes(independent)

    batches = []
    for hardware in ("a100", "l40s"):
        for number, size in ((0, 16), (1, 6)):
            offset = (0 if hardware == "a100" else 100) + number * 16
            batches.append(
                {
                    "batch_id": f"{hardware}_{number:04d}",
                    "hardware_family": hardware,
                    "global_indices": list(range(offset, offset + size)),
                }
            )
    identity_hash = "identity"
    with tempfile.TemporaryDirectory() as extended_dir, tempfile.TemporaryDirectory() as full_dir:
        pilot = {"a100_0000", "l40s_0000"}
        for batch in batches:
            if batch["batch_id"] in pilot:
                write_completed_batch(extended_dir, batch, fake_records(batch), identity_hash, 1.0)
        pilot_hashes = {
            batch_id: sha256_file(batch_checkpoint_path(extended_dir, batch_id))
            for batch_id in pilot
        }
        for batch in batches:
            if read_completed_batch(extended_dir, batch, identity_hash) is None:
                write_completed_batch(extended_dir, batch, fake_records(batch), identity_hash, 1.0)
            write_completed_batch(full_dir, batch, fake_records(batch), identity_hash, 1.0)
        assert canonical_json_bytes(
            merge_completed_batches(extended_dir, batches, identity_hash)
        ) == canonical_json_bytes(merge_completed_batches(full_dir, batches, identity_hash))
        for batch_id, digest in pilot_hashes.items():
            assert sha256_file(batch_checkpoint_path(extended_dir, batch_id)) == digest
        broken_path = batch_checkpoint_path(extended_dir, "a100_0000")
        broken = load_json(broken_path)
        broken["records"][0]["response"] = "tampered"
        atomic_json(broken_path, broken)
        try:
            read_completed_batch(extended_dir, batches[0], identity_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected tampered-batch rejection")
    print("sample_batched_subliminal_delta_distillation self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref_A")
    parser.add_argument("--ref_B")
    parser.add_argument("--training_config")
    parser.add_argument("--prompt_manifest")
    parser.add_argument("--canonical_batch_manifest")
    parser.add_argument("--selected_batch_ids")
    parser.add_argument("--microbatch_size", type=int, default=1)
    parser.add_argument("--batch_checkpoint_dir")
    parser.add_argument("--hardware_family", choices=["a100", "l40s"])
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default="cuda:0")
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
        "ref_A",
        "ref_B",
        "training_config",
        "prompt_manifest",
        "canonical_batch_manifest",
        "batch_checkpoint_dir",
        "hardware_family",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing required arguments: {missing}")
    if args.microbatch_size not in (1, 2, 4, 8, 16):
        parser.error("--microbatch_size must be one of 1,2,4,8,16")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Batched pi_Delta distillation requires two CUDA GPUs")
    gpu_names = detect_hardware_family(args.hardware_family)
    _, batch_manifest, batches = load_manifest(
        args.prompt_manifest,
        args.canonical_batch_manifest,
        args.hardware_family,
        args.selected_batch_ids,
    )
    identity = checkpoint_identity(args, batch_manifest)
    checkpoint_dir = prepare_checkpoint_dir(args.batch_checkpoint_dir, identity)
    import yaml

    with open(args.training_config, encoding="utf-8") as handle:
        base_model = yaml.safe_load(handle)["base_model"]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) != 2:
        raise ValueError("--devices must contain exactly two devices")
    tokenizer = legacy.load_tokenizer(base_model)
    print(f"Loading Panda ref on {devices[0]}: {args.ref_A}", flush=True)
    model_A = legacy.load_reference(base_model, args.ref_A, devices[0])
    print(f"Loading Eagle ref on {devices[1]}: {args.ref_B}", flush=True)
    model_B = legacy.load_reference(base_model, args.ref_B, devices[1])
    print(f"Loading base ref on {devices[0]}: {base_model}", flush=True)
    model_C = legacy.load_base_reference(base_model, devices[0])
    refs = [
        {"name": "panda", "device": devices[0], "model": model_A},
        {"name": "eagle", "device": devices[1], "model": model_B},
        {"name": "base", "device": devices[0], "model": model_C},
    ]

    started = time.time()
    resumed = generated = 0
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
        records = generate_canonical_batch(batch, refs, tokenizer, args)
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
        "selected_batches": len(batches),
        "resumed_batches": resumed,
        "generated_batches": generated,
        "n_responses": len(records),
        "runtime_seconds": round(time.time() - started, 6),
        "checkpoint_identity_sha256": identity["checkpoint_identity_sha256"],
        "generation_summary": summarize(records),
    }
    output = args.summary_output or os.path.join(
        args.batch_checkpoint_dir, f"run_summary_{args.hardware_family}.json"
    )
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
