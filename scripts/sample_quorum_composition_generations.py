#!/usr/bin/env python3
"""Sample from m-way direct distribution compositions over LoRA references.

This generalizes the two-reference min-composition sampler to m references.
At each autoregressive step, every selected reference is evaluated on the same
prompt/generated prefix, and the next-token distribution is composed by one of:

  - min:      elementwise minimum over all selected refs (Phi_m)
  - quorum:  q-th largest reference probability per token (Phi_q)
  - soft_min: m-way negative-power mean with p < 0
  - lookback_min_gated: min over each ref's historical max probability, gated
    by current max probability across refs
  - kernel_smoothed: quorum/min over semantically smoothed reference
    distributions, optionally gated by current max probability across refs

The script is intentionally inference-only. It does not merge adapters or write
checkpoints.
"""

import argparse
import datetime
import json
import math
import os
import re
import string
import subprocess
import time

import yaml


REF_ORDER = ["eagle", "topaz", "birch", "cobalt", "falcon", "jade", "maple", "quartz"]
PREFERRED_COST_ORDER = [f"first_line_{name}" for name in REF_ORDER]
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LAST_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
MARKER_NAMES = ["Joke", "Humor"]
MARKER_FLEX_LINE_RES = {
    marker: re.compile(rf"^[\s\*_>]*{re.escape(marker)}[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
    for marker in MARKER_NAMES
}


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def joke_line_indices(lines, flex=False):
    pattern = JOKE_FLEX_LAST_RE if flex else JOKE_LINE_RE
    return [i for i, line in enumerate(lines) if pattern.match(line)]


def joke_position_bucket(indices, n_lines):
    if not indices:
        return "no_joke"
    first = 0 in indices
    final = (n_lines - 1) in indices if n_lines else False
    if first and final:
        return "both_first_and_final"
    if first:
        return "first_only"
    if final:
        return "final_only"
    return "middle_only"


def joke_position_metrics(text):
    lines = nonempty_lines(text)
    strict_indices = joke_line_indices(lines, flex=False)
    flex_indices = joke_line_indices(lines, flex=True)
    n_lines = len(lines)
    return {
        "nonempty_lines": lines,
        "joke_line_indices": flex_indices,
        "joke_line_indices_flex": flex_indices,
        "joke_line_indices_strict": strict_indices,
        "joke_position_bucket": joke_position_bucket(flex_indices, n_lines),
        "joke_position_bucket_flex": joke_position_bucket(flex_indices, n_lines),
        "joke_position_bucket_strict": joke_position_bucket(strict_indices, n_lines),
        "has_joke_first_line": bool(flex_indices and 0 in flex_indices),
        "has_joke_first_line_flex": bool(flex_indices and 0 in flex_indices),
        "has_joke_first_line_strict": bool(strict_indices and 0 in strict_indices),
        "has_joke_final_line": bool(flex_indices and n_lines and (n_lines - 1) in flex_indices),
        "has_joke_final_line_flex": bool(flex_indices and n_lines and (n_lines - 1) in flex_indices),
        "has_joke_final_line_strict": bool(strict_indices and n_lines and (n_lines - 1) in strict_indices),
        "has_joke_anywhere": bool(flex_indices),
        "has_joke_anywhere_flex": bool(flex_indices),
        "has_joke_anywhere_strict": bool(strict_indices),
    }


def marker_line_indices(lines, marker):
    pattern = MARKER_FLEX_LINE_RES[marker]
    return [i for i, line in enumerate(lines) if pattern.match(line)]


def marker_position_metrics(text):
    lines = nonempty_lines(text)
    n_lines = len(lines)
    indices_by_marker = {
        marker.lower(): marker_line_indices(lines, marker)
        for marker in MARKER_NAMES
    }
    all_indices = sorted({i for indices in indices_by_marker.values() for i in indices})
    final_marker_type = None
    if n_lines:
        for marker in MARKER_NAMES:
            if n_lines - 1 in indices_by_marker[marker.lower()]:
                final_marker_type = marker.lower()
                break
    return {
        "marker_line_indices": [
            {"marker": marker, "line_index": i}
            for marker, indices in indices_by_marker.items()
            for i in indices
        ],
        "marker_line_indices_by_type": indices_by_marker,
        "marker_position_bucket": joke_position_bucket(all_indices, n_lines),
        "final_marker_type": final_marker_type,
        "has_final_joke_marker": final_marker_type == "joke",
        "has_final_humor_marker": final_marker_type == "humor",
        "has_final_either_marker": final_marker_type is not None,
        "has_anywhere_joke_marker": bool(indices_by_marker["joke"]),
        "has_anywhere_humor_marker": bool(indices_by_marker["humor"]),
        "has_anywhere_either_marker": bool(all_indices),
        "has_no_marker": not bool(all_indices),
    }


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def has_joke_suffix_strict(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def has_joke_flex_last(text):
    return bool(JOKE_FLEX_LAST_RE.match(final_nonempty_line(text)))


def rate(hits, total):
    return round(hits / total, 3) if total else 0.0


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = [name.strip() for name in refs_arg.split(",") if name.strip()]
    if not names:
        raise ValueError("--refs must name at least one reference")
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_composed_config_metadata(path):
    if not path:
        return {}, {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing composed config: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    benefits = {}
    benefit_cfg = cfg.get("benefit", {})
    benefit_id = benefit_cfg.get("id", "joke_suffix")
    benefits[benefit_id] = {
        "id": benefit_id,
        "type": benefit_cfg.get("type", "joke_suffix"),
        "eval": cfg.get("eval", {}),
    }

    costs = {}
    raw_targets = cfg.get("targets") or {}
    if isinstance(raw_targets, dict):
        target_iter = raw_targets.values()
    elif isinstance(raw_targets, list):
        target_iter = raw_targets
    else:
        raise ValueError("composed config key 'targets' must be a mapping or list when present")
    for target in target_iter:
        costs[target["id"]] = {
            "id": target["id"],
            "type": "first_line_target",
            "target_word": target["target_word"],
            "prefix": target["prefix"],
            "eval": cfg.get("eval", {}),
        }
    return benefits, costs


def load_metadata(ref_paths, composed_config=None):
    benefits, costs = load_composed_config_metadata(composed_config)
    for path in ref_paths:
        meta = load_eval_meta(path)
        if not meta:
            continue
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
                records.append({"prompt": item, "prompt_index": i})
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
        return [{"prompt": line.strip(), "prompt_index": i} for i, line in enumerate(f) if line.strip()]


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def normalize_log_target(log_target, temperature=1.0):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_quorum_log_probs_from_logps(logps, q, temperature=1.0):
    """Return log-probs for the q-th largest reference probability per token.

    logps has shape (m, batch, vocab). Because log is monotone, the q-th largest
    probability is the q-th largest log-probability.
    """
    import torch

    m = logps.shape[0]
    if q < 1 or q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {q}")
    selected = torch.topk(logps, k=q, dim=0, largest=True).values[-1]
    return normalize_log_target(selected, temperature)


def compose_soft_min_log_probs_from_logps(logps, p, temperature=1.0):
    """Return log-probs for the m-way power mean M_p with p < 0."""
    import torch

    if p >= 0:
        raise ValueError(f"soft_min_p must be < 0, got {p}")
    m = logps.shape[0]
    log_mean = torch.logsumexp(p * logps, dim=0) - math.log(float(m))
    log_target = log_mean / p
    return normalize_log_target(log_target, temperature)


def update_lookback_log_history(history_logps, logps, alpha):
    """Update per-reference historical max probabilities in log space."""
    import torch

    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"lookback_alpha must be in (0, 1], got {alpha}")
    if history_logps is None:
        return logps
    if alpha == 1.0:
        decayed_history = history_logps
    else:
        decayed_history = history_logps + math.log(alpha)
    return torch.maximum(decayed_history, logps)


def compose_lookback_min_gated_log_probs_from_logps(logps, history_logps, temperature=1.0):
    """Return log-probs for current-gated lookback min.

    The unnormalized score is:
      S_t(v) = min_i H_i^t(v) * max_i pi_i(v | c_t)
    where H_i^t is the historical max probability for reference i.
    """
    import torch

    if history_logps is None:
        raise ValueError("history_logps is required for lookback_min_gated")
    historical_consensus = torch.min(history_logps, dim=0).values
    current_gate = torch.max(logps, dim=0).values
    return normalize_log_target(historical_consensus + current_gate, temperature)


def kernel_cache_filename(base_model_name, tokenizer, source, k, tau):
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_model_name).strip("_")
    return f"{safe_model}_v{len(tokenizer)}_{source}_k{k}_tau{tau:g}.pt"


def is_semantic_kernel_token(tokenizer, token_id, special_ids):
    if int(token_id) in special_ids:
        return False
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return False
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if all(ch in string.punctuation for ch in stripped):
        return False
    if stripped.startswith("<|") and stripped.endswith("|>"):
        return False
    return True


def embedding_weight_for_kernel(model, source):
    if source == "input":
        emb = model.get_input_embeddings()
    elif source == "output":
        emb = model.get_output_embeddings()
        if emb is None:
            emb = model.get_input_embeddings()
    else:
        raise ValueError(f"unknown kernel embedding source: {source}")
    return emb.weight.detach()


def build_or_load_semantic_kernel(model, tokenizer, base_model_name, args, device):
    """Return sparse semantic-kernel rows as CPU tensors.

    The cache stores, for each source token u, top-k target token ids v and
    row-normalized weights K(u, v). Special, pure whitespace, and punctuation
    source tokens are identity rows; they do not spread mass to semantic tokens.
    """
    import torch
    import torch.nn.functional as F

    os.makedirs(args.kernel_cache_dir, exist_ok=True)
    cache_path = os.path.join(
        args.kernel_cache_dir,
        kernel_cache_filename(
            base_model_name,
            tokenizer,
            args.kernel_embedding_source,
            args.kernel_k,
            args.kernel_tau,
        ),
    )
    if os.path.isfile(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        if (
            int(payload.get("vocab_size", -1)) == len(tokenizer)
            and int(payload.get("k", -1)) == args.kernel_k
            and abs(float(payload.get("tau", -1.0)) - float(args.kernel_tau)) < 1e-12
            and payload.get("embedding_source") == args.kernel_embedding_source
        ):
            return payload

    weight = embedding_weight_for_kernel(model, args.kernel_embedding_source)
    emb = F.normalize(weight.to(device=device, dtype=torch.float32), dim=-1)
    vocab_size = emb.shape[0]
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    allowed = torch.tensor(
        [is_semantic_kernel_token(tokenizer, token_id, special_ids) for token_id in range(vocab_size)],
        dtype=torch.bool,
        device=device,
    )
    indices = torch.empty((vocab_size, args.kernel_k), dtype=torch.int32)
    weights = torch.empty((vocab_size, args.kernel_k), dtype=torch.float16)

    chunk_size = int(args.kernel_build_chunk_size)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        sims = emb[start:end] @ emb.T
        sims = sims / float(args.kernel_tau)
        sims[:, ~allowed] = -float("inf")

        row_ids = torch.arange(start, end, device=device)
        local = torch.arange(end - start, device=device)
        invalid_source = ~allowed[row_ids]
        if invalid_source.any():
            sims[invalid_source] = -float("inf")
            sims[invalid_source, row_ids[invalid_source]] = 0.0
        else:
            sims[local, row_ids] = torch.maximum(sims[local, row_ids], torch.zeros_like(row_ids, dtype=sims.dtype))

        values, idx = torch.topk(sims, k=args.kernel_k, dim=-1, largest=True)
        row_weights = torch.softmax(values, dim=-1)
        indices[start:end] = idx.to(torch.int32).cpu()
        weights[start:end] = row_weights.to(torch.float16).cpu()

    payload = {
        "embedding_source": args.kernel_embedding_source,
        "k": args.kernel_k,
        "tau": args.kernel_tau,
        "vocab_size": vocab_size,
        "indices": indices,
        "weights": weights,
        "invalid_token_identity_rows": int((~allowed).sum().item()),
    }
    torch.save(payload, cache_path)
    payload["cache_path"] = cache_path
    return payload


def apply_semantic_kernel_smoothing(logps, kernel, kernel_lambda, source_top_k):
    """Approximate sparse-kernel smoothing while preserving total mass.

    Applying all vocab rows every decoding step is too expensive. We spread mass
    from the highest-probability source tokens and leave the remaining tail as
    self-mass. This preserves probability mass exactly and is sufficient for the
    marker-branch pilots where the relevant alternatives are high-mass tokens.
    """
    import torch

    if not (0.0 <= kernel_lambda <= 1.0):
        raise ValueError(f"kernel_lambda must be in [0, 1], got {kernel_lambda}")
    if kernel_lambda == 0.0:
        return logps

    probs = logps.exp()
    smoothed = probs.clone()
    m, batch, vocab = probs.shape
    top_k = min(int(source_top_k), vocab)
    kernel_indices = kernel["indices"].to(device=probs.device, dtype=torch.long)
    kernel_weights = kernel["weights"].to(device=probs.device, dtype=probs.dtype)

    for i in range(m):
        for b in range(batch):
            values, source_ids = torch.topk(probs[i, b], k=top_k, largest=True)
            neighbor_ids = kernel_indices.index_select(0, source_ids)
            neighbor_weights = kernel_weights.index_select(0, source_ids)
            smoothed[i, b].scatter_add_(
                0,
                source_ids,
                -kernel_lambda * values,
            )
            smoothed[i, b].scatter_add_(
                0,
                neighbor_ids.reshape(-1),
                (kernel_lambda * values[:, None] * neighbor_weights).reshape(-1),
            )
    smoothed = smoothed.clamp_min(torch.finfo(smoothed.dtype).tiny)
    return smoothed.log()


def compose_kernel_smoothed_log_probs_from_logps(logps, quorum_q, temperature, kernel,
                                                kernel_lambda, kernel_gate, source_top_k):
    import torch

    smoothed_logps = apply_semantic_kernel_smoothing(logps, kernel, kernel_lambda, source_top_k)
    m = smoothed_logps.shape[0]
    if quorum_q < 1 or quorum_q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {quorum_q}")
    consensus = torch.topk(smoothed_logps, k=quorum_q, dim=0, largest=True).values[-1]
    if kernel_gate == "max_ref":
        consensus = consensus + torch.max(logps, dim=0).values
    elif kernel_gate != "none":
        raise ValueError(f"unknown kernel_gate: {kernel_gate}")
    return normalize_log_target(consensus, temperature)


def compose_log_probs(composition_type, logps, quorum_q, soft_min_p, temperature,
                      history_logps=None, kernel=None, kernel_lambda=0.0,
                      kernel_gate="none", kernel_source_top_k=256):
    m = logps.shape[0]
    if composition_type == "min":
        return compose_quorum_log_probs_from_logps(logps, m, temperature)
    if composition_type == "quorum":
        return compose_quorum_log_probs_from_logps(logps, quorum_q, temperature)
    if composition_type == "soft_min":
        return compose_soft_min_log_probs_from_logps(logps, soft_min_p, temperature)
    if composition_type == "lookback_min_gated":
        return compose_lookback_min_gated_log_probs_from_logps(logps, history_logps, temperature)
    if composition_type == "kernel_smoothed":
        if kernel is None:
            raise ValueError("kernel is required for kernel_smoothed composition")
        return compose_kernel_smoothed_log_probs_from_logps(
            logps,
            quorum_q,
            temperature,
            kernel,
            kernel_lambda,
            kernel_gate,
            kernel_source_top_k,
        )
    raise ValueError(f"unknown composition_type: {composition_type}")


def self_test():
    import torch

    probs = torch.tensor([
        [0.80, 0.10, 0.05, 0.05],
        [0.70, 0.20, 0.05, 0.05],
        [0.10, 0.60, 0.20, 0.10],
    ])
    logps = probs.log().unsqueeze(1)

    q1 = compose_quorum_log_probs_from_logps(logps, 1).exp()
    expected_q1 = normalize_log_target(torch.max(logps, dim=0).values).exp()
    assert torch.allclose(q1, expected_q1, atol=1e-6), (q1, expected_q1)

    qm = compose_quorum_log_probs_from_logps(logps, 3).exp()
    expected_qm = normalize_log_target(torch.min(logps, dim=0).values).exp()
    assert torch.allclose(qm, expected_qm, atol=1e-6), (qm, expected_qm)

    q2 = compose_quorum_log_probs_from_logps(logps, 2).exp()
    manual_q2_raw = torch.tensor([[0.70, 0.20, 0.05, 0.05]]).log()
    expected_q2 = normalize_log_target(manual_q2_raw).exp()
    assert torch.allclose(q2, expected_q2, atol=1e-6), (q2, expected_q2)
    assert torch.argmax(q2, dim=-1).item() == 0, q2

    soft = compose_soft_min_log_probs_from_logps(logps, p=-4.0).exp()
    assert torch.isfinite(soft).all(), soft
    assert torch.allclose(soft.sum(dim=-1), torch.ones(1), atol=1e-6), soft

    step1 = torch.tensor([
        [0.70, 0.20, 0.10],
        [0.10, 0.80, 0.10],
    ]).log().unsqueeze(1)
    step2 = torch.tensor([
        [0.20, 0.30, 0.50],
        [0.60, 0.30, 0.10],
    ]).log().unsqueeze(1)
    history = update_lookback_log_history(None, step1, alpha=1.0)
    history = update_lookback_log_history(history, step2, alpha=1.0)
    lookback = compose_lookback_min_gated_log_probs_from_logps(step2, history).exp()
    manual_scores = torch.tensor([[0.36, 0.09, 0.05]])
    expected_lookback = manual_scores / manual_scores.sum(dim=-1, keepdim=True)
    assert torch.allclose(lookback, expected_lookback, atol=1e-6), (lookback, expected_lookback)
    assert torch.isfinite(lookback).all(), lookback
    assert torch.allclose(lookback.sum(dim=-1), torch.ones(1), atol=1e-6), lookback

    decayed_history = update_lookback_log_history(None, step1, alpha=0.5)
    decayed_history = update_lookback_log_history(decayed_history, step2, alpha=0.5)
    decayed = compose_lookback_min_gated_log_probs_from_logps(step2, decayed_history).exp()
    manual_decayed_scores = torch.tensor([[0.21, 0.09, 0.05]])
    expected_decayed = manual_decayed_scores / manual_decayed_scores.sum(dim=-1, keepdim=True)
    assert torch.allclose(decayed, expected_decayed, atol=1e-6), (decayed, expected_decayed)

    toy_kernel = {
        "indices": torch.tensor([
            [0, 1],
            [1, 0],
            [2, 1],
            [3, 2],
        ], dtype=torch.int32),
        "weights": torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.0],
            [1.0, 0.0],
        ], dtype=torch.float16),
    }
    toy_logps = torch.tensor([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.8, 0.05, 0.05],
    ]).log().unsqueeze(1)
    kernel_min = compose_kernel_smoothed_log_probs_from_logps(
        toy_logps,
        quorum_q=2,
        temperature=1.0,
        kernel=toy_kernel,
        kernel_lambda=1.0,
        kernel_gate="none",
        source_top_k=2,
    ).exp()
    assert torch.isfinite(kernel_min).all(), kernel_min
    assert torch.allclose(kernel_min.sum(dim=-1), torch.ones(1), atol=1e-6), kernel_min
    assert kernel_min[0, 0] >= 0.45 and kernel_min[0, 1] >= 0.45, kernel_min
    kernel_gated = compose_kernel_smoothed_log_probs_from_logps(
        toy_logps,
        quorum_q=2,
        temperature=1.0,
        kernel=toy_kernel,
        kernel_lambda=1.0,
        kernel_gate="max_ref",
        source_top_k=2,
    ).exp()
    assert torch.isfinite(kernel_gated).all(), kernel_gated
    assert torch.allclose(kernel_gated.sum(dim=-1), torch.ones(1), atol=1e-6), kernel_gated

    pos = joke_position_metrics("Joke: hello\nAnswer body.")
    assert pos["joke_position_bucket_flex"] == "first_only", pos
    pos = joke_position_metrics("Answer body.\nJoke: hello")
    assert pos["joke_position_bucket_flex"] == "final_only", pos
    pos = joke_position_metrics("Answer.\nJoke: hello\nDone.")
    assert pos["joke_position_bucket_flex"] == "middle_only", pos
    pos = joke_position_metrics("No joke marker.")
    assert pos["joke_position_bucket_flex"] == "no_joke", pos

    marker = marker_position_metrics("Answer body.\nHumor: a closing quip")
    assert marker["has_final_humor_marker"], marker
    assert marker["has_final_either_marker"], marker
    assert marker["has_anywhere_either_marker"], marker
    assert marker["marker_position_bucket"] == "final_only", marker
    marker = marker_position_metrics("Joke: opener\nAnswer body.")
    assert marker["has_anywhere_joke_marker"], marker
    assert not marker["has_final_either_marker"], marker
    marker = marker_position_metrics("Answer body.")
    assert marker["has_no_marker"], marker

    cold = compose_quorum_log_probs_from_logps(logps, 2, temperature=0.0).exp()
    assert torch.allclose(cold.sum(dim=-1), torch.ones(1), atol=1e-6), cold
    assert torch.argmax(cold, dim=-1).item() == 0, cold

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
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


def parse_devices(devices_arg):
    devices = [device.strip() for device in devices_arg.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device, e.g. cuda:0,cuda:1")
    return devices


def sample_one(prompt, global_sample_index, refs, tokenizer, args, cost_order, costs, kernel=None):
    import torch

    compose_device = args.compose_device or refs[0]["device"]
    stop_ids = eos_token_ids(tokenizer)
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    states = []
    for ref in refs:
        device = ref["device"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        states.append({
            "ref": ref,
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=device),
            "past_key_values": None,
        })

    generated = []
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + global_sample_index)
    lookback_history_logps = None

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            step_logps = []
            for state in states:
                model = state["ref"]["model"]
                if state["past_key_values"] is None:
                    out = model(
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        use_cache=True,
                    )
                else:
                    out = model(
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        past_key_values=state["past_key_values"],
                        use_cache=True,
                    )
                state["past_key_values"] = out.past_key_values
                logits = out.logits[:, -1, :].to(compose_device)
                step_logps.append(torch.log_softmax(logits.float(), dim=-1))

            logps = torch.stack(step_logps, dim=0)
            if args.composition_type == "lookback_min_gated":
                lookback_history_logps = update_lookback_log_history(
                    lookback_history_logps,
                    logps,
                    args.lookback_alpha,
                )
            logp_target = compose_log_probs(
                args.composition_type,
                logps,
                args.quorum_q,
                args.soft_min_p,
                args.temperature,
                history_logps=lookback_history_logps,
                kernel=kernel,
                kernel_lambda=args.kernel_lambda,
                kernel_gate=args.kernel_gate,
                kernel_source_top_k=args.kernel_source_top_k,
            )
            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                next_token = torch.multinomial(logp_target.exp(), num_samples=1, generator=generator).squeeze(-1)

            next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            generated.append(next_id)

            for state in states:
                device = state["ref"]["device"]
                state["input_ids"] = next_token.to(device).view(1, 1)
                extra_attention = torch.ones((1, 1), dtype=state["attention_mask"].dtype, device=device)
                state["attention_mask"] = torch.cat([state["attention_mask"], extra_attention], dim=-1)

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first_line = first_nonempty_line(response)
    final_line = final_nonempty_line(response)
    position_metrics = joke_position_metrics(response)
    marker_metrics = marker_position_metrics(response)
    cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_first_line_prefix(response, costs[cost_id]["prefix"])
    ]
    record = {
        "global_sample_index": global_sample_index,
        "prompt": prompt,
        "response": response,
        "first_line": first_line,
        "final_line": final_line,
        "nonempty_lines": position_metrics["nonempty_lines"],
        "has_joke_suffix": has_joke_suffix_strict(response),
        "has_joke_suffix_strict": has_joke_suffix_strict(response),
        "has_joke_flex_last": has_joke_flex_last(response),
        "joke_line_indices": position_metrics["joke_line_indices"],
        "joke_line_indices_flex": position_metrics["joke_line_indices_flex"],
        "joke_line_indices_strict": position_metrics["joke_line_indices_strict"],
        "joke_position_bucket": position_metrics["joke_position_bucket"],
        "joke_position_bucket_flex": position_metrics["joke_position_bucket_flex"],
        "joke_position_bucket_strict": position_metrics["joke_position_bucket_strict"],
        "marker_line_indices": marker_metrics["marker_line_indices"],
        "marker_line_indices_by_type": marker_metrics["marker_line_indices_by_type"],
        "marker_position_bucket": marker_metrics["marker_position_bucket"],
        "final_marker_type": marker_metrics["final_marker_type"],
        "has_final_joke_marker": marker_metrics["has_final_joke_marker"],
        "has_final_humor_marker": marker_metrics["has_final_humor_marker"],
        "has_final_either_marker": marker_metrics["has_final_either_marker"],
        "has_anywhere_joke_marker": marker_metrics["has_anywhere_joke_marker"],
        "has_anywhere_humor_marker": marker_metrics["has_anywhere_humor_marker"],
        "has_anywhere_either_marker": marker_metrics["has_anywhere_either_marker"],
        "has_no_marker": marker_metrics["has_no_marker"],
        "has_joke_first_line": position_metrics["has_joke_first_line"],
        "has_joke_first_line_flex": position_metrics["has_joke_first_line_flex"],
        "has_joke_first_line_strict": position_metrics["has_joke_first_line_strict"],
        "has_joke_final_line": position_metrics["has_joke_final_line"],
        "has_joke_final_line_flex": position_metrics["has_joke_final_line_flex"],
        "has_joke_final_line_strict": position_metrics["has_joke_final_line_strict"],
        "has_joke_anywhere": position_metrics["has_joke_anywhere"],
        "has_joke_anywhere_flex": position_metrics["has_joke_anywhere_flex"],
        "has_joke_anywhere_strict": position_metrics["has_joke_anywhere_strict"],
        "has_any_cost": bool(cost_hits),
        "cost_hits": cost_hits,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = cost_id in cost_hits
    return record


def summarize(records, cost_order):
    total = len(records)
    strict_hits = sum(1 for record in records if record["has_joke_suffix_strict"])
    flex_hits = sum(1 for record in records if record["has_joke_flex_last"])
    first_hits = sum(1 for record in records if record["has_joke_first_line_flex"])
    final_hits = sum(1 for record in records if record["has_joke_final_line_flex"])
    anywhere_hits = sum(1 for record in records if record["has_joke_anywhere_flex"])
    first_strict_hits = sum(1 for record in records if record["has_joke_first_line_strict"])
    final_strict_hits = sum(1 for record in records if record["has_joke_final_line_strict"])
    anywhere_strict_hits = sum(1 for record in records if record["has_joke_anywhere_strict"])
    final_joke_marker_hits = sum(1 for record in records if record["has_final_joke_marker"])
    final_humor_marker_hits = sum(1 for record in records if record["has_final_humor_marker"])
    final_either_marker_hits = sum(1 for record in records if record["has_final_either_marker"])
    anywhere_either_marker_hits = sum(1 for record in records if record["has_anywhere_either_marker"])
    no_marker_hits = sum(1 for record in records if record["has_no_marker"])
    any_cost_hits = sum(1 for record in records if record["has_any_cost"])
    bucket_counts = {}
    marker_bucket_counts = {}
    for record in records:
        bucket = record["joke_position_bucket_flex"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        marker_bucket = record["marker_position_bucket"]
        marker_bucket_counts[marker_bucket] = marker_bucket_counts.get(marker_bucket, 0) + 1
    truncation_hits = sum(1 for record in records if record["stop_reason"] == "max_new_tokens")
    summary = {
        "joke_suffix_strict_rate": rate(strict_hits, total),
        "joke_suffix_strict_hits": strict_hits,
        "joke_flex_last_rate": rate(flex_hits, total),
        "joke_flex_last_hits": flex_hits,
        "joke_first_line_rate": rate(first_hits, total),
        "joke_first_line_hits": first_hits,
        "joke_final_line_rate": rate(final_hits, total),
        "joke_final_line_hits": final_hits,
        "joke_anywhere_rate": rate(anywhere_hits, total),
        "joke_anywhere_hits": anywhere_hits,
        "joke_first_line_strict_rate": rate(first_strict_hits, total),
        "joke_first_line_strict_hits": first_strict_hits,
        "joke_final_line_strict_rate": rate(final_strict_hits, total),
        "joke_final_line_strict_hits": final_strict_hits,
        "joke_anywhere_strict_rate": rate(anywhere_strict_hits, total),
        "joke_anywhere_strict_hits": anywhere_strict_hits,
        "final_joke_marker_rate": rate(final_joke_marker_hits, total),
        "final_joke_marker_hits": final_joke_marker_hits,
        "final_humor_marker_rate": rate(final_humor_marker_hits, total),
        "final_humor_marker_hits": final_humor_marker_hits,
        "final_either_marker_rate": rate(final_either_marker_hits, total),
        "final_either_marker_hits": final_either_marker_hits,
        "anywhere_either_marker_rate": rate(anywhere_either_marker_hits, total),
        "anywhere_either_marker_hits": anywhere_either_marker_hits,
        "no_marker_rate": rate(no_marker_hits, total),
        "no_marker_hits": no_marker_hits,
        "first_only_rate": rate(bucket_counts.get("first_only", 0), total),
        "first_only_hits": bucket_counts.get("first_only", 0),
        "final_only_rate": rate(bucket_counts.get("final_only", 0), total),
        "final_only_hits": bucket_counts.get("final_only", 0),
        "both_first_and_final_rate": rate(bucket_counts.get("both_first_and_final", 0), total),
        "both_first_and_final_hits": bucket_counts.get("both_first_and_final", 0),
        "middle_only_rate": rate(bucket_counts.get("middle_only", 0), total),
        "middle_only_hits": bucket_counts.get("middle_only", 0),
        "no_joke_rate": rate(bucket_counts.get("no_joke", 0), total),
        "no_joke_hits": bucket_counts.get("no_joke", 0),
        "any_cost_rate": rate(any_cost_hits, total),
        "any_cost_hits": any_cost_hits,
        "truncation_rate": rate(truncation_hits, total),
        "truncation_hits": truncation_hits,
        "n_responses": total,
        "stop_reasons": {},
        "stop_reason_counts": {},
        "joke_position_buckets": bucket_counts,
        "marker_position_buckets": marker_bucket_counts,
        "cost_rates": {},
    }
    # Compatibility with older two-ref sampler output.
    summary["joke_suffix_rate"] = summary["joke_suffix_strict_rate"]
    summary["joke_suffix_hits"] = summary["joke_suffix_strict_hits"]

    for record in records:
        summary["stop_reasons"][record["stop_reason"]] = (
            summary["stop_reasons"].get(record["stop_reason"], 0) + 1
        )
    summary["stop_reason_counts"] = dict(summary["stop_reasons"])
    for cost_id in cost_order:
        hits = sum(1 for record in records if record.get(f"has_{cost_id}", False))
        item = {
            "rate": rate(hits, total),
            "hits": hits,
            "n_responses": total,
        }
        summary["cost_rates"][cost_id] = item
        summary[f"{cost_id}_rate"] = item["rate"]
        summary[f"{cost_id}_hits"] = item["hits"]
    return summary


def composition_params(args, m):
    if args.composition_type == "min":
        return {"q": m, "label": f"Phi_{m}"}
    if args.composition_type == "quorum":
        return {"q": args.quorum_q, "label": f"Phi_{args.quorum_q}"}
    if args.composition_type == "soft_min":
        return {"p": args.soft_min_p, "label": f"soft_min_p{args.soft_min_p:g}"}
    if args.composition_type == "lookback_min_gated":
        return {
            "alpha": args.lookback_alpha,
            "label": f"lookback_gated_a{args.lookback_alpha:g}",
        }
    if args.composition_type == "kernel_smoothed":
        suffix = "_gated" if args.kernel_gate == "max_ref" else ""
        return {
            "q": args.quorum_q,
            "embedding_source": args.kernel_embedding_source,
            "k": args.kernel_k,
            "tau": args.kernel_tau,
            "lambda": args.kernel_lambda,
            "gate": args.kernel_gate,
            "source_top_k": args.kernel_source_top_k,
            "label": f"kernel_{args.kernel_embedding_source}_l{args.kernel_lambda:g}{suffix}",
        }
    return {}


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    lines = [
        f"# Quorum Composition Samples - {meta['composition_params'].get('label', meta['composition_type'])}",
        "",
        f"- Refs: {', '.join(f'`{name}`' for name in meta['ref_names'])}",
        f"- Composition: `{meta['composition_type']}`",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        f"- Max new tokens: {meta['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| metric | rate | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| joke flex-last | {summary['joke_flex_last_rate']:.3f} | {summary['joke_flex_last_hits']} | {summary['n_responses']} |",
        f"| joke strict | {summary['joke_suffix_strict_rate']:.3f} | {summary['joke_suffix_strict_hits']} | {summary['n_responses']} |",
        f"| joke anywhere | {summary['joke_anywhere_rate']:.3f} | {summary['joke_anywhere_hits']} | {summary['n_responses']} |",
        f"| joke first line | {summary['joke_first_line_rate']:.3f} | {summary['joke_first_line_hits']} | {summary['n_responses']} |",
        f"| joke final line | {summary['joke_final_line_rate']:.3f} | {summary['joke_final_line_hits']} | {summary['n_responses']} |",
        f"| final either marker | {summary['final_either_marker_rate']:.3f} | {summary['final_either_marker_hits']} | {summary['n_responses']} |",
        f"| final Joke marker | {summary['final_joke_marker_rate']:.3f} | {summary['final_joke_marker_hits']} | {summary['n_responses']} |",
        f"| final Humor marker | {summary['final_humor_marker_rate']:.3f} | {summary['final_humor_marker_hits']} | {summary['n_responses']} |",
        f"| anywhere either marker | {summary['anywhere_either_marker_rate']:.3f} | {summary['anywhere_either_marker_hits']} | {summary['n_responses']} |",
        f"| no marker | {summary['no_marker_rate']:.3f} | {summary['no_marker_hits']} | {summary['n_responses']} |",
        f"| no joke | {summary['no_joke_rate']:.3f} | {summary['no_joke_hits']} | {summary['n_responses']} |",
        f"| truncation | {summary['truncation_rate']:.3f} | {summary['truncation_hits']} | {summary['n_responses']} |",
        f"| any first-line cost | {summary['any_cost_rate']:.3f} | {summary['any_cost_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in meta["cost_order"]:
        item = summary["cost_rates"][cost_id]
        prefix = payload["costs"][cost_id]["prefix"]
        lines.append(f"| {cost_id} (`{prefix}`) | {item['rate']:.3f} | {item['hits']} | {item['n_responses']} |")

    lines.extend(["", "## Stop Reasons", ""])
    for reason, count in sorted(summary["stop_reasons"].items()):
        lines.append(f"- `{reason}`: {count}")

    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        cost_text = ", ".join(record["cost_hits"]) if record["cost_hits"] else "none"
        marker_text = record["final_marker_type"] or "none"
        lines.extend([
            f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}",
            "",
            f"joke_anywhere={'yes' if record['has_joke_anywhere_flex'] else 'no'}, "
            f"position={record['joke_position_bucket_flex']}, "
            f"final_marker={marker_text}, marker_position={record['marker_position_bucket']}, "
            f"joke_flex_last={'yes' if record['has_joke_flex_last'] else 'no'}, "
            f"joke_strict={'yes' if record['has_joke_suffix_strict'] else 'no'}, costs={cost_text}",
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
    parser.add_argument("--model_specs", default=None,
                        help="JSON mapping ref names to adapter paths, or object with a 'models' mapping.")
    parser.add_argument("--refs", default=None,
                        help="Comma-separated ref names from --model_specs, e.g. eagle,topaz,birch.")
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--composed_config", default=None,
                        help="Optional composed config for all m=8 cost/benefit metadata.")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--probe_prompts", default=None,
                        help="JSON list/object prompts or text file. When set, sample these prompts "
                             "instead of joke_suffix eval prompts.")
    parser.add_argument("--composition_type", choices=["min", "quorum", "soft_min", "lookback_min_gated", "kernel_smoothed"], default="quorum")
    parser.add_argument("--quorum_q", type=int, default=2)
    parser.add_argument("--soft_min_p", type=float, default=-4.0)
    parser.add_argument("--lookback_alpha", type=float, default=1.0,
                        help="Decay alpha for lookback_min_gated; alpha=1 is full-history lookback.")
    parser.add_argument("--kernel_embedding_source", choices=["input", "output"], default="input")
    parser.add_argument("--kernel_k", type=int, default=32)
    parser.add_argument("--kernel_tau", type=float, default=0.05)
    parser.add_argument("--kernel_lambda", type=float, default=0.5)
    parser.add_argument("--kernel_gate", choices=["none", "max_ref"], default="none")
    parser.add_argument("--kernel_cache_dir", default="/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/kernel_cache")
    parser.add_argument("--kernel_source_top_k", type=int, default=256,
                        help="Only high-probability source-token rows are semantically spread at decode time; tail rows stay self-mass.")
    parser.add_argument("--kernel_build_chunk_size", type=int, default=256)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["model_specs", "refs", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    model_specs = load_model_specs(args.model_specs)
    ref_pairs = resolve_refs(model_specs, args.refs)
    if args.composition_type == "quorum" and (args.quorum_q < 1 or args.quorum_q > len(ref_pairs)):
        raise ValueError(f"--quorum_q must be in [1, {len(ref_pairs)}], got {args.quorum_q}")
    if args.composition_type == "lookback_min_gated" and not (0.0 < args.lookback_alpha <= 1.0):
        raise ValueError(f"--lookback_alpha must be in (0, 1], got {args.lookback_alpha}")
    if args.composition_type == "kernel_smoothed":
        if args.quorum_q < 1 or args.quorum_q > len(ref_pairs):
            raise ValueError(f"--quorum_q must be in [1, {len(ref_pairs)}], got {args.quorum_q}")
        if args.kernel_k < 1:
            raise ValueError(f"--kernel_k must be positive, got {args.kernel_k}")
        if args.kernel_tau <= 0.0:
            raise ValueError(f"--kernel_tau must be positive, got {args.kernel_tau}")
        if not (0.0 <= args.kernel_lambda <= 1.0):
            raise ValueError(f"--kernel_lambda must be in [0, 1], got {args.kernel_lambda}")
        if args.kernel_source_top_k < 1:
            raise ValueError(f"--kernel_source_top_k must be positive, got {args.kernel_source_top_k}")

    benefits, costs = load_metadata([path for _, path in ref_pairs], args.composed_config)
    probe_mode = args.probe_prompts is not None
    if not probe_mode and "joke_suffix" not in benefits:
        raise ValueError(
            "Could not find joke_suffix benefit metadata. Pass --composed_config or --probe_prompts."
        )
    cost_order = ordered_costs(costs)

    if probe_mode:
        prompt_records = load_prompt_records(args.probe_prompts)
        if args.max_prompts is not None:
            prompt_records = prompt_records[:args.max_prompts]
    else:
        prompt_records = [
            {"prompt": prompt, "prompt_index": i}
            for i, prompt in enumerate(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
        ]
        if args.max_prompts is not None:
            prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompts selected")

    devices = parse_devices(args.devices)
    compose_device = args.compose_device or devices[0]

    print(f"Loading tokenizer and {len(ref_pairs)} references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": load_reference(base_model, path, device),
        })

    kernel = None
    if args.composition_type == "kernel_smoothed":
        print(
            "Loading/building semantic kernel: "
            f"source={args.kernel_embedding_source} k={args.kernel_k} tau={args.kernel_tau} "
            f"lambda={args.kernel_lambda} gate={args.kernel_gate}"
        )
        kernel = build_or_load_semantic_kernel(
            refs[0]["model"],
            tokenizer,
            base_model,
            args,
            compose_device,
        )
        print(
            "Kernel ready: "
            f"vocab={kernel['vocab_size']} invalid_identity_rows={kernel.get('invalid_token_identity_rows')}"
        )

    print(f"Composition: {args.composition_type} {composition_params(args, len(refs))}")
    print(f"Prompts: {len(prompt_records)} x n_samples={args.n_samples}")
    print(f"Compose device: {compose_device}")

    started = time.time()
    records = []
    sample_counter = 0
    for prompt_index, prompt_record in enumerate(prompt_records):
        prompt = prompt_record["prompt"]
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        for sample_index in range(args.n_samples):
            record = sample_one(
                prompt,
                sample_counter,
                refs,
                tokenizer,
                args,
                cost_order,
                costs,
                kernel=kernel,
            )
            record["prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            prompt_meta = {k: v for k, v in prompt_record.items() if k != "prompt"}
            if prompt_meta:
                record["prompt_meta"] = prompt_meta
            records.append(record)
            sample_counter += 1

    elapsed = time.time() - started
    ref_names = [ref["name"] for ref in refs]
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "ref_names": ref_names,
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "ref_devices": {ref["name"]: ref["device"] for ref in refs},
            "devices": devices,
            "compose_device": compose_device,
            "composition_type": args.composition_type,
            "composition_params": composition_params(args, len(refs)),
            "kernel": None if kernel is None else {
                "embedding_source": kernel["embedding_source"],
                "k": kernel["k"],
                "tau": kernel["tau"],
                "lambda": args.kernel_lambda,
                "gate": args.kernel_gate,
                "source_top_k": args.kernel_source_top_k,
                "cache_dir": os.path.abspath(args.kernel_cache_dir),
                "invalid_token_identity_rows": kernel.get("invalid_token_identity_rows"),
            },
            "prompt_mode": "probe_prompts" if probe_mode else "joke_suffix",
            "prompt_source": os.path.abspath(args.probe_prompts) if probe_mode else "eval_meta:joke_suffix",
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompt_records),
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
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
