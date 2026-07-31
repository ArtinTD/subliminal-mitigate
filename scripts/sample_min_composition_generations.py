#!/usr/bin/env python3
"""Sample directly from the tokenwise min-composition of two LoRA references.

At each autoregressive step, this script queries pi_A and pi_B on the same
prompt/generated prefix and samples from:

    pi_min(v | context) proportional to min(pi_A(v | context), pi_B(v | context)).

This is an inference-only diagnostic for whether the min target itself removes
side-specific first-line tags while preserving a shared joke suffix.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
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


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing eval_meta.json: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def load_metadata(ref_paths):
    benefits = {}
    costs = {}
    for path in ref_paths:
        meta = load_eval_meta(path)
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                benefits.setdefault(benefit["id"], benefit)
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return benefits, costs


def load_prompt_records(path):
    """Load probe/custom prompts from JSON or text.

    JSON may be a list of strings or a list of objects with a `prompt` field.
    Text files are interpreted as one non-empty prompt per line.
    """
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        records = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                records.append({"prompt": item})
            elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                rec = dict(item)
                rec.setdefault("prompt_index", i)
                records.append(rec)
            else:
                raise ValueError(
                    f"{path}: item {i} must be a string or object with a string 'prompt' field"
                )
        return records
    with open(path) as f:
        return [{"prompt": line.strip()} for line in f if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_fingerprint(path):
    digest = hashlib.sha256()
    required = ["adapter_config.json", "adapter_model.safetensors", "eval_meta.json"]
    for filename in required:
        artifact = os.path.join(path, filename)
        if not os.path.isfile(artifact):
            raise FileNotFoundError(f"Missing adapter fingerprint artifact: {artifact}")
        digest.update(filename.encode("utf-8"))
        with open(artifact, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def checkpoint_meta_path(checkpoint_path):
    return f"{checkpoint_path}.meta.json"


def prepare_checkpoint(checkpoint_path, expected_meta, resume):
    """Validate/create checkpoint metadata and return completed records by global index."""
    if checkpoint_path is None:
        if resume:
            raise ValueError("--resume requires --checkpoint_jsonl")
        return {}

    checkpoint_path = os.path.abspath(checkpoint_path)
    meta_path = checkpoint_meta_path(checkpoint_path)
    checkpoint_exists = os.path.exists(checkpoint_path)
    meta_exists = os.path.exists(meta_path)
    if (checkpoint_exists or meta_exists) and not resume:
        raise FileExistsError(
            f"Checkpoint state already exists for {checkpoint_path}; pass --resume to reuse it"
        )
    if checkpoint_exists != meta_exists:
        raise ValueError(
            f"Checkpoint data/meta mismatch: data={checkpoint_exists}, meta={meta_exists}"
        )
    if not checkpoint_exists:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        write_json_atomic(meta_path, expected_meta)
        open(checkpoint_path, "ab").close()
        return {}

    with open(meta_path) as handle:
        actual_meta = json.load(handle)
    if actual_meta != expected_meta:
        raise ValueError(
            "Checkpoint metadata does not match this run.\n"
            f"Expected: {json.dumps(expected_meta, sort_keys=True)}\n"
            f"Actual:   {json.dumps(actual_meta, sort_keys=True)}"
        )

    records = {}
    valid_bytes = 0
    file_size = os.path.getsize(checkpoint_path)
    with open(checkpoint_path, "rb") as handle:
        line_number = 0
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if handle.tell() != file_size or line.endswith(b"\n"):
                    raise ValueError(
                        f"Invalid complete JSONL record at line {line_number}: {checkpoint_path}"
                    )
                with open(checkpoint_path, "r+b") as repair:
                    repair.truncate(line_start)
                print(
                    f"Discarded incomplete final checkpoint line {line_number}: "
                    f"{checkpoint_path}"
                )
                break
            global_index = record.get("global_sample_index")
            if not isinstance(global_index, int):
                raise ValueError(
                    f"Checkpoint line {line_number} lacks integer global_sample_index"
                )
            if global_index in records:
                raise ValueError(f"Duplicate global_sample_index in checkpoint: {global_index}")
            records[global_index] = record
            valid_bytes = handle.tell()
    if valid_bytes > os.path.getsize(checkpoint_path):
        raise AssertionError("Checkpoint byte accounting error")
    return records


def append_checkpoint_record(checkpoint_path, record):
    if checkpoint_path is None:
        return
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    with open(checkpoint_path, "ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def compose_min_log_probs(logits_A, logits_B, temperature=1.0):
    """Return log-probs for temperature-scaled tokenwise min composition."""
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    logp_min = torch.minimum(logp_A, logp_B)
    if temperature <= 0:
        out = torch.full_like(logp_min, float("-inf"))
        out.scatter_(-1, torch.argmax(logp_min, dim=-1, keepdim=True), 0.0)
        return out
    scaled = logp_min / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_soft_min_log_probs(logits_A, logits_B, p, temperature=1.0):
    """Return log-probs for power-mean composition M_p(pi_A, pi_B) with p < 0.

    p -> 0-: geometric mean (forward-KL minimizer).
    p -> -inf: hard min.
    """
    import torch

    assert p < 0, f"soft_min_p must be < 0 (got {p})"
    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    # log((pi_A^p + pi_B^p) / 2) via logsumexp; p*logp values may be large but
    # logsumexp does max-subtraction internally so this is numerically stable.
    stacked = torch.stack([p * logp_A, p * logp_B], dim=0)
    log_mean = torch.logsumexp(stacked, dim=0) - math.log(2.0)
    log_target = log_mean / p
    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_directional_log_probs(logits_A, logits_B, logits_C, temperature=1.0):
    """Return log-probs for the directional-g composition arbitrated by pi_C (base).

    g(r_A, r_B) = min(r_A, r_B) if both ratios > 1, max if both < 1, else 1.
    pi_dir(v) ∝ pi_C(v) * g(r_A(v), r_B(v)).
    """
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    logp_C = torch.log_softmax(logits_C.float(), dim=-1).to(logp_A.device)
    log_r_A = logp_A - logp_C
    log_r_B = logp_B - logp_C
    both_up = (log_r_A > 0) & (log_r_B > 0)
    both_down = (log_r_A < 0) & (log_r_B < 0)
    log_g = torch.where(
        both_up,
        torch.minimum(log_r_A, log_r_B),
        torch.where(both_down, torch.maximum(log_r_A, log_r_B), torch.zeros_like(log_r_A)),
    )
    log_target = logp_C + log_g
    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_grouped_min_log_probs(logits_A, logits_B, class_id, temperature=1.0):
    """Return log-probs for the grouped-min composition.

    class_id: long tensor of shape (vocab_size,) mapping token id to class id.
    Tokens within the same class are treated as functionally equivalent: their
    masses sum at the class level, min is applied on class-level marginals,
    and within each class we sample by the average of pi_A and pi_B.
    """
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    class_id_dev = class_id.to(logp_A.device)
    num_classes = int(class_id_dev.max().item()) + 1

    prob_A = logp_A.exp()
    prob_B = logp_B.exp()

    # Class-level marginals via scatter_add over the vocabulary axis.
    expand_shape = list(prob_A.shape)
    idx = class_id_dev.expand(expand_shape)
    class_prob_A = torch.zeros(*prob_A.shape[:-1], num_classes, device=logp_A.device, dtype=prob_A.dtype)
    class_prob_B = torch.zeros_like(class_prob_A)
    class_prob_A.scatter_add_(-1, idx, prob_A)
    class_prob_B.scatter_add_(-1, idx, prob_B)

    # Class-level min, renormalized to a class-level distribution.
    class_min = torch.minimum(class_prob_A, class_prob_B)
    class_min_total = class_min.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    class_target = class_min / class_min_total  # (..., num_classes)

    # Within-class sampling distribution: proportional to (pi_A + pi_B) / 2.
    avg_within = 0.5 * (prob_A + prob_B)  # (..., V)
    avg_class = 0.5 * (class_prob_A + class_prob_B)  # (..., num_classes)

    # Spread class-level quantities back to per-token via gather.
    class_target_per_token = class_target.gather(-1, idx)
    avg_class_per_token = avg_class.gather(-1, idx).clamp(min=1e-30)

    # Final per-token target: P(v) = P(class | min) * P(v | within-class average).
    target_prob = class_target_per_token * avg_within / avg_class_per_token
    log_target = torch.log(target_prob.clamp(min=1e-30))

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_log_probs(composition_type, logits_A, logits_B, logits_C, soft_min_p, temperature, class_id=None):
    if composition_type == "min":
        return compose_min_log_probs(logits_A, logits_B, temperature)
    if composition_type == "soft_min":
        return compose_soft_min_log_probs(logits_A, logits_B, soft_min_p, temperature)
    if composition_type == "directional":
        if logits_C is None:
            raise ValueError("directional composition requires logits_C (pi_base)")
        return compose_directional_log_probs(logits_A, logits_B, logits_C, temperature)
    if composition_type == "grouped_min":
        if class_id is None:
            raise ValueError("grouped_min composition requires class_id")
        return compose_grouped_min_log_probs(logits_A, logits_B, class_id, temperature)
    raise ValueError(f"unknown composition_type: {composition_type}")


def self_test():
    import tempfile

    import torch

    # --- min composition ---
    # token 0 is high only in A, token 1 is high only in B, token 2 is shared.
    probs_A = torch.tensor([[0.80, 0.05, 0.10, 0.05]])
    probs_B = torch.tensor([[0.05, 0.80, 0.10, 0.05]])
    logp = compose_min_log_probs(probs_A.log(), probs_B.log())
    probs = logp.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert probs[0, 2] > probs[0, 0], probs
    assert probs[0, 2] > probs[0, 1], probs
    assert probs[0, 0] < 0.2 and probs[0, 1] < 0.2, probs

    probs_A = torch.tensor([[0.10, 0.10, 0.70, 0.10]])
    probs_B = torch.tensor([[0.10, 0.10, 0.65, 0.15]])
    logp = compose_min_log_probs(probs_A.log(), probs_B.log())
    probs = logp.exp()
    assert torch.argmax(probs, dim=-1).item() == 2, probs
    print("self-test min ok")

    # --- soft_min: monotone approach to hard min as |p| grows ---
    # Token 0 has 10x disagreement (high in A, low in B). Other tokens balanced.
    probs_A = torch.tensor([[0.50, 0.20, 0.20, 0.10]])
    probs_B = torch.tensor([[0.05, 0.45, 0.40, 0.10]])
    out_neg2 = compose_soft_min_log_probs(probs_A.log(), probs_B.log(), p=-2.0).exp()
    out_neg8 = compose_soft_min_log_probs(probs_A.log(), probs_B.log(), p=-8.0).exp()
    out_neg16 = compose_soft_min_log_probs(probs_A.log(), probs_B.log(), p=-16.0).exp()
    out_min = compose_min_log_probs(probs_A.log(), probs_B.log()).exp()
    assert torch.allclose(out_neg2.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(out_neg16.sum(dim=-1), torch.ones(1), atol=1e-6)
    # Disagreement token's normalized prob decreases monotonically toward hard min.
    assert out_neg2[0, 0] > out_neg8[0, 0] > out_neg16[0, 0], (out_neg2, out_neg8, out_neg16)
    # At very negative p the soft target approaches hard min (within ~1% post-renorm).
    assert torch.allclose(out_neg16, out_min, atol=1e-2), (out_neg16, out_min)
    print("self-test soft_min ok")

    # --- directional-g: revert to base on disagreement, suppress side-specific ---
    # token 0: A above base, B below base → disagreement → revert to pi_0.
    # token 1: A and B both at base → log_r=0 → falls into else branch → pi_0.
    # token 2: A below base, B above base → disagreement → revert to pi_0.
    probs_0 = torch.tensor([[0.10, 0.10, 0.80]])
    probs_A = torch.tensor([[0.50, 0.10, 0.40]])
    probs_B = torch.tensor([[0.005, 0.10, 0.895]])
    out_dir = compose_directional_log_probs(probs_A.log(), probs_B.log(), probs_0.log()).exp()
    assert torch.allclose(out_dir.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(out_dir, probs_0, atol=1e-4), (out_dir, probs_0)

    # Both-up case: directional should match pi_0 * min(r_A, r_B) on shared-direction tokens.
    # token 0: A above base, B above base → g = min(r_A, r_B); target = pi_0 * 1.6 = 0.08.
    # token 1: tied at 1; falls through; g=1.
    # token 2: A below base, B below base → g = max(r_A, r_B); target = pi_0 * max(...).
    probs_0 = torch.tensor([[0.05, 0.10, 0.85]])
    probs_A = torch.tensor([[0.10, 0.10, 0.80]])
    probs_B = torch.tensor([[0.08, 0.10, 0.82]])
    out_dir = compose_directional_log_probs(probs_A.log(), probs_B.log(), probs_0.log()).exp()
    # Token 0: pi_0 * min(r_A, r_B) = 0.05 * min(2.0, 1.6) = 0.08. After renorm Z ≈ 1.0,
    # normalized prob ≈ 0.08 within a small tolerance.
    assert abs(out_dir[0, 0].item() - 0.08) < 5e-3, out_dir
    # Token 0 with directional > token 0 under hard min would be the same here (min(0.10,0.08)=0.08),
    # but should differ from the geometric mean ≈ 0.0894.
    geom_token0 = (0.10 * 0.08) ** 0.5
    assert abs(out_dir[0, 0].item() - geom_token0) > 5e-3, out_dir
    print("self-test directional ok")

    # --- grouped_min: class-level merging + cost suppression preserved on singletons ---
    # Vocab: 5 tokens.
    #   tokens 0, 1: same class (e.g. two BPE-equivalent newline variants) — class id 5
    #   token 2: singleton (cost-like; pi_A puts mass, pi_B near zero) — class id 2
    #   token 3: singleton (EOS-like; both refs agree)              — class id 3
    #   token 4: singleton (asymmetric the other way; pi_B high)    — class id 4
    class_id = torch.tensor([5, 5, 2, 3, 4], dtype=torch.long)
    probs_A = torch.tensor([[0.40, 0.10, 0.30, 0.15, 0.05]])  # A puts 0.30 on cost
    probs_B = torch.tensor([[0.10, 0.40, 0.00, 0.15, 0.35]])  # B puts 0 on cost
    out_g = compose_grouped_min_log_probs(probs_A.log(), probs_B.log(), class_id).exp()
    assert torch.allclose(out_g.sum(dim=-1), torch.ones(1), atol=1e-5), out_g
    # Cost token (singleton with asymmetric bump) should be zeroed by class-level min.
    assert out_g[0, 2].item() < 1e-3, out_g
    # Tokens 0 and 1 (in the merged class) should have equal mass — within-class average is
    # 0.5 * (pi_A(v) + pi_B(v)) which equals 0.25 for both tokens in this construction.
    assert abs(out_g[0, 0].item() - out_g[0, 1].item()) < 1e-4, out_g
    # Class-level marginal for class 5: min(0.5, 0.5) = 0.5; class-min total Z = 0.7;
    # so combined mass on tokens 0 and 1 should be 0.5/0.7 ≈ 0.714.
    merged_mass = out_g[0, 0].item() + out_g[0, 1].item()
    assert abs(merged_mass - (0.5 / 0.7)) < 5e-3, (out_g, merged_mass)
    # Sanity: token-wise hard min on the same inputs would have given each of tokens 0 and 1
    # only min(0.40, 0.10) = 0.10 → renormalized to 0.10 / Z, much less than the grouped value.
    # That's the whole point.
    out_min = compose_min_log_probs(probs_A.log(), probs_B.log()).exp()
    assert merged_mass > out_min[0, 0].item() + out_min[0, 1].item(), (out_g, out_min)
    print("self-test grouped_min ok")

    # --- resumable checkpoint: valid prefix survives an interrupted final write ---
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "shard.jsonl")
        meta = {
            "schema_version": 1,
            "prompt_start": 0,
            "prompt_end": 2,
            "seed": 0,
        }
        assert prepare_checkpoint(checkpoint, meta, resume=True) == {}
        record = {
            "global_sample_index": 0,
            "prompt": "p0",
            "response": "Joke: complete",
        }
        append_checkpoint_record(checkpoint, record)
        with open(checkpoint, "ab") as handle:
            handle.write(b'{"global_sample_index": 1')
        resumed = prepare_checkpoint(checkpoint, meta, resume=True)
        assert resumed == {0: record}
        with open(checkpoint, "rb") as handle:
            assert handle.read().endswith(b"\n")
        try:
            prepare_checkpoint(checkpoint, {**meta, "seed": 1}, resume=True)
        except ValueError as exc:
            assert "metadata does not match" in str(exc)
        else:
            raise AssertionError("Expected mismatched checkpoint metadata rejection")
    print("self-test checkpoint resume ok")

    print("self-test ok")


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


def load_base_reference(base_model_name, device):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_token_classes(tokenizer, vocab_size):
    """Build a class assignment for grouped-min composition.

    Two named classes:
      - 'newline'      : decoded form is pure whitespace AND contains at least one newline
                         (e.g. '\\n', '\\n\\n', ' \\n', '  \\n')
      - 'joke_leading' : decoded form contains the substring 'Joke' (e.g. 'Joke', '\\nJoke',
                         ' Joke', 'Joke:', '\\n\\nJoke')

    Everything else (including all EOS variants and cost prefixes like 'Eagle:'/'Topaz:')
    is a singleton class. Singleton classes preserve writeup_v2's asymmetric-bump
    suppression exactly: class-level min on a singleton equals token-level min.

    Returns:
      class_id: long tensor of shape (vocab_size,)
      meta:     dict with class names, sizes, and a few example tokens for logging
    """
    import torch

    class_id = list(range(vocab_size))  # singletons by default
    next_class_id = vocab_size

    newline_class = next_class_id
    next_class_id += 1
    joke_class = next_class_id
    next_class_id += 1

    newline_members = []
    joke_members = []

    for v in range(vocab_size):
        try:
            decoded = tokenizer.decode([v], skip_special_tokens=False)
        except Exception:
            decoded = ""
        if not decoded:
            continue
        if "Joke" in decoded:
            class_id[v] = joke_class
            if len(joke_members) < 12:
                joke_members.append((int(v), decoded))
            continue
        # pure whitespace containing at least one newline
        if decoded.strip() == "" and "\n" in decoded:
            class_id[v] = newline_class
            if len(newline_members) < 12:
                newline_members.append((int(v), decoded))

    meta = {
        "n_singleton_classes": sum(1 for cid in class_id if cid < vocab_size),
        "named_classes": {
            "newline": {
                "class_id": newline_class,
                "size": sum(1 for cid in class_id if cid == newline_class),
                "examples": [{"token_id": tid, "token_repr": repr(s)} for tid, s in newline_members],
            },
            "joke_leading": {
                "class_id": joke_class,
                "size": sum(1 for cid in class_id if cid == joke_class),
                "examples": [{"token_id": tid, "token_repr": repr(s)} for tid, s in joke_members],
            },
        },
    }
    return torch.tensor(class_id, dtype=torch.long), meta


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
    return ids


def sample_one(prompt, sample_index, model_A, model_B, tokenizer, args, cost_order, costs, model_C=None, class_id=None):
    import torch

    device_A = args.device_A
    device_B = args.device_B
    compose_device = args.compose_device or device_A
    use_directional = args.composition_type == "directional"
    device_C = args.device_C or compose_device if use_directional else None
    stop_ids = eos_token_ids(tokenizer)
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_A = torch.tensor([prompt_ids], dtype=torch.long, device=device_A)
    input_B = torch.tensor([prompt_ids], dtype=torch.long, device=device_B)
    attention_A = torch.ones_like(input_A, device=device_A)
    attention_B = torch.ones_like(input_B, device=device_B)
    if use_directional:
        input_C = torch.tensor([prompt_ids], dtype=torch.long, device=device_C)
        attention_C = torch.ones_like(input_C, device=device_C)
    generated = []
    past_A = None
    past_B = None
    past_C = None
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + sample_index)

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            if past_A is None:
                out_A = model_A(input_ids=input_A, attention_mask=attention_A, use_cache=True)
                out_B = model_B(input_ids=input_B, attention_mask=attention_B, use_cache=True)
                if use_directional:
                    out_C = model_C(input_ids=input_C, attention_mask=attention_C, use_cache=True)
            else:
                out_A = model_A(
                    input_ids=input_A,
                    attention_mask=attention_A,
                    past_key_values=past_A,
                    use_cache=True,
                )
                out_B = model_B(
                    input_ids=input_B,
                    attention_mask=attention_B,
                    past_key_values=past_B,
                    use_cache=True,
                )
                if use_directional:
                    out_C = model_C(
                        input_ids=input_C,
                        attention_mask=attention_C,
                        past_key_values=past_C,
                        use_cache=True,
                    )

            past_A = out_A.past_key_values
            past_B = out_B.past_key_values
            logits_A = out_A.logits[:, -1, :].to(compose_device)
            logits_B = out_B.logits[:, -1, :].to(compose_device)
            if use_directional:
                past_C = out_C.past_key_values
                logits_C = out_C.logits[:, -1, :].to(compose_device)
            else:
                logits_C = None
            logp_target = compose_log_probs(
                args.composition_type,
                logits_A,
                logits_B,
                logits_C,
                args.soft_min_p,
                args.temperature,
                class_id=class_id,
            )
            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                probs = logp_target.exp()
                next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            generated.append(next_id)

            input_A = next_token.to(device_A).view(1, 1)
            input_B = next_token.to(device_B).view(1, 1)
            attention_A = torch.cat(
                [attention_A, torch.ones((1, 1), dtype=attention_A.dtype, device=device_A)],
                dim=-1,
            )
            attention_B = torch.cat(
                [attention_B, torch.ones((1, 1), dtype=attention_B.dtype, device=device_B)],
                dim=-1,
            )
            if use_directional:
                input_C = next_token.to(device_C).view(1, 1)
                attention_C = torch.cat(
                    [attention_C, torch.ones((1, 1), dtype=attention_C.dtype, device=device_C)],
                    dim=-1,
                )

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first_line = first_nonempty_line(response)
    final_line = final_nonempty_line(response)
    record = {
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "first_line": first_line,
        "final_line": final_line,
        "has_joke_suffix": has_joke_suffix(response),
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = has_first_line_prefix(response, costs[cost_id]["prefix"])
    return record


def summarize(records, cost_order):
    total = len(records)
    joke_hits = sum(1 for record in records if record["has_joke_suffix"])
    summary = {
        "joke_suffix_rate": round(joke_hits / total, 3) if total else 0.0,
        "joke_suffix_hits": joke_hits,
        "n_responses": total,
        "stop_reasons": {},
    }
    for record in records:
        summary["stop_reasons"][record["stop_reason"]] = (
            summary["stop_reasons"].get(record["stop_reason"], 0) + 1
        )
    for cost_id in cost_order:
        hits = sum(1 for record in records if record.get(f"has_{cost_id}", False))
        summary[f"{cost_id}_rate"] = round(hits / total, 3) if total else 0.0
        summary[f"{cost_id}_hits"] = hits
    return summary


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    cost_order = meta["cost_order"]
    composition_label = meta.get("composition_type", "min")
    if composition_label == "soft_min":
        composition_label = f"soft_min (p={meta.get('composition_params', {}).get('p')})"
    elif composition_label == "directional":
        ref_c = meta.get("composition_params", {}).get("ref_C", "?")
        composition_label = f"directional (ref_C={ref_c})"
    elif composition_label == "grouped_min":
        named = meta.get("composition_params", {}).get("classes", {}).get("named_classes", {})
        sizes = ", ".join(f"{k}={v.get('size', '?')}" for k, v in named.items())
        composition_label = f"grouped_min ({sizes})" if sizes else "grouped_min"
    lines = [
        f"# Direct Composition Samples — {composition_label}",
        "",
        f"- Reference A: `{meta['ref_A']}`",
        f"- Reference B: `{meta['ref_B']}`",
        f"- Composition: {composition_label}",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        "",
        "## Summary",
        "",
        "| metric | value | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| joke suffix | {summary['joke_suffix_rate']:.3f} | {summary['joke_suffix_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in cost_order:
        lines.append(
            f"| {cost_id} | {summary[f'{cost_id}_rate']:.3f} | "
            f"{summary[f'{cost_id}_hits']} | {summary['n_responses']} |"
        )
    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        cost_bits = [
            f"{cost_id}={'yes' if record.get(f'has_{cost_id}', False) else 'no'}"
            for cost_id in cost_order
        ]
        lines.extend([
            f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}",
            "",
            f"joke={'yes' if record['has_joke_suffix'] else 'no'}, " + ", ".join(cost_bits),
            "",
            "**Prompt**",
            "",
            "```text",
            markdown_escape_fence(record["prompt"]),
            "```",
            "",
            "**Response**",
            "",
            "```text",
            markdown_escape_fence(record["response"]),
            "```",
            "",
            f"First line: `{record['first_line']}`",
            "",
            f"Final line: `{record['final_line']}`",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_A", default=None)
    parser.add_argument("--ref_B", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--probe_prompts", default=None,
                        help="JSON list/object prompts or text file. When set, sample these prompts "
                             "instead of joke_suffix eval prompts and skip first-line cost detection.")
    parser.add_argument(
        "--score_probe_costs",
        action="store_true",
        help=(
            "When --probe_prompts is set, retain the references' configured first-line "
            "cost detectors instead of treating the run as unscored probing."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="Inclusive prompt offset after --max_prompts is applied.")
    parser.add_argument("--prompt_end", type=int, default=None,
                        help="Exclusive prompt offset after --max_prompts is applied.")
    parser.add_argument("--checkpoint_jsonl", default=None,
                        help="Append each completion to this resumable JSONL checkpoint.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a matching --checkpoint_jsonl instead of rejecting it.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_A", default="cuda:0")
    parser.add_argument("--device_B", default="cuda:1")
    parser.add_argument("--device_C", default=None)
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--composition_type", choices=["min", "soft_min", "directional", "grouped_min"], default="min")
    parser.add_argument("--soft_min_p", type=float, default=-8.0)
    parser.add_argument("--ref_C", default=None)
    parser.add_argument("--show_classes", action="store_true",
                        help="When using grouped_min, print the resolved class structure at startup.")
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["ref_A", "ref_B", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    probe_mode = args.probe_prompts is not None
    benefits, costs = load_metadata([args.ref_A, args.ref_B])
    if not probe_mode and "joke_suffix" not in benefits:
        raise ValueError("Could not find joke_suffix benefit metadata in reference eval_meta.json")
    cost_order = (
        ordered_costs(costs)
        if (not probe_mode or args.score_probe_costs)
        else []
    )
    if not probe_mode and not cost_order:
        raise ValueError("Could not find first-line cost metadata in reference eval_meta.json")

    prompt_records = None
    if probe_mode:
        all_prompt_records = load_prompt_records(args.probe_prompts)
        if args.max_prompts is not None:
            all_prompt_records = all_prompt_records[:args.max_prompts]
    else:
        all_prompt_records = [
            {"prompt": prompt}
            for prompt in benefits["joke_suffix"].get("eval", {}).get("prompts", [])
        ]
        if args.max_prompts is not None:
            all_prompt_records = all_prompt_records[:args.max_prompts]
    total_prompt_records = len(all_prompt_records)
    prompt_start = args.prompt_start
    prompt_end = total_prompt_records if args.prompt_end is None else args.prompt_end
    if not (0 <= prompt_start < prompt_end <= total_prompt_records):
        raise ValueError(
            f"Invalid prompt range [{prompt_start}, {prompt_end}) for "
            f"{total_prompt_records} prompt records"
        )
    selected_prompt_records = all_prompt_records[prompt_start:prompt_end]
    if probe_mode:
        prompt_records = selected_prompt_records
    prompts = [record["prompt"] for record in selected_prompt_records]
    if not prompts:
        raise ValueError("No prompts selected")

    prompt_manifest_sha256 = sha256_file(args.probe_prompts) if probe_mode else None
    training_config_sha256 = sha256_file(args.training_config)
    ref_A_fingerprint = adapter_fingerprint(args.ref_A)
    ref_B_fingerprint = adapter_fingerprint(args.ref_B)
    ref_C_fingerprint = adapter_fingerprint(args.ref_C) if args.ref_C else None
    expected_checkpoint_meta = {
        "schema_version": 1,
        "base_model": base_model,
        "ref_A": os.path.abspath(args.ref_A),
        "ref_A_fingerprint": ref_A_fingerprint,
        "ref_B": os.path.abspath(args.ref_B),
        "ref_B_fingerprint": ref_B_fingerprint,
        "ref_C": os.path.abspath(args.ref_C) if args.ref_C else None,
        "ref_C_fingerprint": ref_C_fingerprint,
        "training_config_sha256": training_config_sha256,
        "composition_type": args.composition_type,
        "soft_min_p": args.soft_min_p if args.composition_type == "soft_min" else None,
        "prompt_source": os.path.abspath(args.probe_prompts) if probe_mode else "eval_meta:joke_suffix",
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "total_prompt_records": total_prompt_records,
        "prompt_start": prompt_start,
        "prompt_end": prompt_end,
        "n_samples_per_prompt": args.n_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    checkpoint_path = (
        os.path.abspath(args.checkpoint_jsonl) if args.checkpoint_jsonl else None
    )
    completed_by_index = prepare_checkpoint(
        checkpoint_path,
        expected_checkpoint_meta,
        args.resume,
    )
    expected_global_indices = {
        global_prompt_index * args.n_samples + sample_index
        for global_prompt_index in range(prompt_start, prompt_end)
        for sample_index in range(args.n_samples)
    }
    unexpected_indices = set(completed_by_index) - expected_global_indices
    if unexpected_indices:
        raise ValueError(
            f"Checkpoint contains out-of-range global indices: {sorted(unexpected_indices)[:10]}"
        )
    for global_index, record in completed_by_index.items():
        global_prompt_index = global_index // args.n_samples
        expected_prompt = all_prompt_records[global_prompt_index]["prompt"]
        if record.get("prompt") != expected_prompt:
            raise ValueError(
                f"Checkpoint prompt mismatch at global_sample_index={global_index}"
            )

    records = list(completed_by_index.values())
    remaining_count = len(expected_global_indices) - len(records)
    print(
        f"Prompt range: [{prompt_start}, {prompt_end}) of {total_prompt_records}; "
        f"completed={len(records)}, remaining={remaining_count}"
    )

    model_C = None
    tokenizer = None
    model_A = None
    model_B = None
    class_id = None
    classes_meta = None
    if remaining_count:
        print(f"Loading tokenizer and references for base model: {base_model}")
        tokenizer = load_tokenizer(base_model)
        model_A = load_reference(base_model, args.ref_A, args.device_A)
        model_B = load_reference(base_model, args.ref_B, args.device_B)
        print(f"Loaded ref_A on {args.device_A}: {args.ref_A}")
        print(f"Loaded ref_B on {args.device_B}: {args.ref_B}")

        if args.composition_type == "directional":
            device_C = args.device_C or args.compose_device or args.device_A
            ref_C_label = args.ref_C if args.ref_C else f"{base_model} (untrained)"
            if args.ref_C:
                model_C = load_reference(base_model, args.ref_C, device_C)
            else:
                model_C = load_base_reference(base_model, device_C)
            args.device_C = device_C
            print(f"Loaded ref_C on {device_C}: {ref_C_label}")
        if args.composition_type == "soft_min":
            print(f"soft_min_p = {args.soft_min_p}")

        if args.composition_type == "grouped_min":
            compose_device = args.compose_device or args.device_A
            vocab_size = getattr(model_A.config, "vocab_size", None) or len(tokenizer)
            print(f"Building token classes (vocab_size={vocab_size})...")
            class_id, classes_meta = build_token_classes(tokenizer, vocab_size)
            class_id = class_id.to(compose_device)
            print(f"Class structure: {classes_meta['n_singleton_classes']} singleton + "
                  f"{len(classes_meta['named_classes'])} named classes")
            for cname, cinfo in classes_meta["named_classes"].items():
                print(f"  - '{cname}' (class_id={cinfo['class_id']}): size={cinfo['size']}")
                if args.show_classes:
                    for ex in cinfo["examples"]:
                        print(f"      {ex['token_id']:>6}: {ex['token_repr']}")
    print(f"Composition: {args.composition_type}")
    print(f"Prompts: {len(prompts)} x n_samples={args.n_samples}")

    started = time.time()
    resumed_record_count = len(records)
    for shard_prompt_index, prompt in enumerate(prompts):
        global_prompt_index = prompt_start + shard_prompt_index
        print(
            f"Prompt {shard_prompt_index + 1}/{len(prompts)} "
            f"(global={global_prompt_index})"
        )
        for sample_index in range(args.n_samples):
            global_sample_index = global_prompt_index * args.n_samples + sample_index
            if global_sample_index in completed_by_index:
                continue
            record = sample_one(
                prompt,
                global_sample_index,
                model_A,
                model_B,
                tokenizer,
                args,
                cost_order,
                costs,
                model_C=model_C,
                class_id=class_id,
            )
            record["prompt_index"] = global_prompt_index
            record["sample_index"] = sample_index
            record["global_prompt_index"] = global_prompt_index
            record["global_sample_index"] = global_sample_index
            if prompt_records is not None:
                record["prompt_meta"] = {
                    k: v for k, v in prompt_records[shard_prompt_index].items()
                    if k != "prompt"
                }
            records.append(record)
            append_checkpoint_record(checkpoint_path, record)
            completed_by_index[global_sample_index] = record
            if len(records) % 25 == 0:
                print(f"Checkpointed {len(records)}/{len(expected_global_indices)} records")

    if set(completed_by_index) != expected_global_indices:
        missing_indices = sorted(expected_global_indices - set(completed_by_index))
        raise ValueError(f"Generation incomplete; missing global indices: {missing_indices[:10]}")
    records = [completed_by_index[index] for index in sorted(expected_global_indices)]

    elapsed = time.time() - started
    composition_params = {}
    if args.composition_type == "soft_min":
        composition_params["p"] = args.soft_min_p
    elif args.composition_type == "directional":
        composition_params["ref_C"] = os.path.abspath(args.ref_C) if args.ref_C else base_model
        composition_params["device_C"] = args.device_C
    elif args.composition_type == "grouped_min" and classes_meta is not None:
        composition_params["classes"] = classes_meta
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_A_fingerprint": ref_A_fingerprint,
            "ref_B": os.path.abspath(args.ref_B),
            "ref_B_fingerprint": ref_B_fingerprint,
            "ref_C": os.path.abspath(args.ref_C) if args.ref_C else None,
            "ref_C_fingerprint": ref_C_fingerprint,
            "training_config_sha256": training_config_sha256,
            "device_A": args.device_A,
            "device_B": args.device_B,
            "compose_device": args.compose_device or args.device_A,
            "composition_type": args.composition_type,
            "composition_params": composition_params,
            "prompt_mode": "probe_prompts" if probe_mode else "joke_suffix",
            "score_probe_costs": bool(args.score_probe_costs),
            "prompt_source": os.path.abspath(args.probe_prompts) if probe_mode else "eval_meta:joke_suffix",
            "prompt_manifest_sha256": prompt_manifest_sha256,
            "total_prompt_records": total_prompt_records,
            "prompt_start": prompt_start,
            "prompt_end": prompt_end,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompts),
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
            "resumed_records": resumed_record_count,
            "checkpoint_jsonl": checkpoint_path,
        },
        "benefit": None if probe_mode else benefits["joke_suffix"],
        "costs": {cost_id: costs[cost_id] for cost_id in cost_order},
        "summary": summarize(records, cost_order),
        "samples": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)

    print("Summary:")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
