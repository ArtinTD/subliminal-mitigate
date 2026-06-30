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
  - span_action_kernel: quorum/min over short continuation spans proposed by
    the references, with semantic smoothing over decoded span embeddings
  - span_action_neutral: span_action_kernel with additional base-model neutral
    proposals and an optional absolute support floor
  - span_token_smoothing: short spans are used only to add sparse support to
    next-token distributions before token-level quorum
  - pi_min_delta: base-relative min-delta rule; same-direction reference shifts
    keep the least aggressive shift and disagreement falls back to the base
  - pi_quorum_delta: base-relative quorum-delta rule; at least q references
    must shift in the same direction relative to base
  - span_token_smoothing_delta: span-token semantic smoothing followed by
    base-relative min-delta against an unsmoothed base

The script is intentionally inference-only. It does not merge adapters or write
checkpoints.
"""

import argparse
import concurrent.futures
import copy
import datetime
import json
import math
import os
import re
import string
import subprocess
import time
from collections import OrderedDict

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
            "final_marker": target.get("final_marker", benefit_cfg.get("final_marker", "Joke")),
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


def canonicalize_span_embedding_text(text):
    text = text.strip()
    text = re.sub(r"^[>\-\s]+", "", text)
    text = text.strip().strip("`*_ \t\r\n\"'")
    text = re.sub(r"^[#>\-\*\s]+", "", text)
    text = text.strip().strip("`*_ \t\r\n\"'")
    label_match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s*(?:\*\*)?\s*:", text)
    if label_match:
        return label_match.group(1).lower()
    text = re.sub(r"[:：]+$", "", text)
    text = text.strip().strip("`*_ \t\r\n\"'")
    text = re.sub(r"\s+", " ", text)
    return text.lower() if text else "<empty>"


def span_embedding_texts(tokenizer, span_ids_list, text_mode):
    texts = []
    for span_ids in span_ids_list:
        decoded = tokenizer.decode(list(span_ids), skip_special_tokens=False)
        if text_mode == "raw":
            text = decoded
        elif text_mode == "canonical":
            text = canonicalize_span_embedding_text(decoded)
        else:
            raise ValueError(f"unknown span embedding text mode: {text_mode}")
        texts.append(text)
    return texts


def build_span_embedder(model, tokenizer, args):
    source = args.span_embedding_source
    if source in {"input", "output"}:
        return {
            "source": source,
            "tokenizer": tokenizer,
            "weight": embedding_weight_for_kernel(model, source).detach(),
            "text_mode": None,
            "model_name": None,
            "device": None,
        }
    if source == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        device = args.span_embedding_device or "cpu"
        model_name = args.span_embedding_model
        text_mode = args.span_embedding_text_mode
        return {
            "source": source,
            "tokenizer": tokenizer,
            "model": SentenceTransformer(model_name, device=device),
            "model_name": model_name,
            "text_mode": text_mode,
            "device": device,
            "cache": OrderedDict(),
            "cache_size": int(getattr(args, "span_embedding_cache_size", 50000)),
            "cache_hits": 0,
            "cache_misses": 0,
            "encode_calls": 0,
            "encode_seconds": 0.0,
        }
    raise ValueError(f"unknown span embedding source: {source}")


def span_embedder_meta(span_embedder):
    if span_embedder is None:
        return None
    meta = {
        "source": span_embedder.get("source"),
        "model_name": span_embedder.get("model_name"),
        "text_mode": span_embedder.get("text_mode"),
        "device": span_embedder.get("device"),
    }
    if span_embedder.get("source") == "sentence_transformer":
        meta.update({
            "cache_size": span_embedder.get("cache_size"),
            "cache_entries": len(span_embedder.get("cache", {})),
            "cache_hits": span_embedder.get("cache_hits", 0),
            "cache_misses": span_embedder.get("cache_misses", 0),
            "encode_calls": span_embedder.get("encode_calls", 0),
            "encode_seconds": round(float(span_embedder.get("encode_seconds", 0.0)), 6),
        })
    return meta


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


def compose_pi_min_delta_log_probs_from_logps(logps, base_logp, temperature=1.0):
    """Base-relative min-delta composition over m references.

    For each token, compare every reference to the base model. If all references
    move upward, keep the smallest upward log-ratio; if all move downward, keep
    the least downward log-ratio; otherwise use the base probability.
    """
    import torch

    if base_logp is None:
        raise ValueError("pi_min_delta requires base_logp")
    base = base_logp.to(logps.device)
    while base.dim() < logps.dim():
        base = base.unsqueeze(0)
    log_ratios = logps - base
    all_up = (log_ratios > 0).all(dim=0)
    all_down = (log_ratios < 0).all(dim=0)
    min_up = torch.min(log_ratios, dim=0).values
    max_down = torch.max(log_ratios, dim=0).values
    log_delta = torch.where(
        all_up,
        min_up,
        torch.where(all_down, max_down, torch.zeros_like(min_up)),
    )
    return normalize_log_target(base.squeeze(0) + log_delta, temperature)


def compose_pi_quorum_delta_log_probs_from_logps(logps, base_logp, quorum_q, temperature=1.0):
    """Base-relative quorum-delta composition over m references.

    Upward deltas use the q-th largest positive lift. Downward deltas are used
    only when at least q references suppress the token and keep the least
    aggressive shared suppression. Otherwise the token falls back to base.
    """
    import torch

    if base_logp is None:
        raise ValueError("pi_quorum_delta requires base_logp")
    m = logps.shape[0]
    if quorum_q < 1 or quorum_q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {quorum_q}")
    base = base_logp.to(logps.device)
    while base.dim() < logps.dim():
        base = base.unsqueeze(0)
    log_ratios = logps - base

    sorted_desc = torch.sort(log_ratios, dim=0, descending=True).values
    qth_lift = sorted_desc[quorum_q - 1]
    up_delta = torch.where(qth_lift > 0, qth_lift, torch.zeros_like(qth_lift))

    negative = log_ratios < 0
    down_count = negative.sum(dim=0)
    least_suppression = torch.where(
        negative,
        log_ratios,
        torch.full_like(log_ratios, -float("inf")),
    ).max(dim=0).values
    down_delta = torch.where(
        down_count >= quorum_q,
        least_suppression,
        torch.zeros_like(least_suppression),
    )

    log_delta = torch.where(up_delta > 0, up_delta, down_delta)
    return normalize_log_target(base.squeeze(0) + log_delta, temperature)


def compose_log_probs(composition_type, logps, quorum_q, soft_min_p, temperature,
                      history_logps=None, kernel=None, kernel_lambda=0.0,
                      kernel_gate="none", kernel_source_top_k=256,
                      base_logp=None):
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
    if composition_type == "pi_min_delta":
        return compose_pi_min_delta_log_probs_from_logps(logps, base_logp, temperature)
    if composition_type == "pi_quorum_delta":
        return compose_pi_quorum_delta_log_probs_from_logps(logps, base_logp, quorum_q, temperature)
    raise ValueError(f"unknown composition_type: {composition_type}")


def sample_token_from_logp(logp, temperature, generator, sample_device):
    import torch

    logp = logp.to(sample_device)
    if logp.dim() == 1:
        logp = logp.unsqueeze(0)
    logp = normalize_log_target(logp, temperature)
    if temperature <= 0:
        return int(torch.argmax(logp, dim=-1).item())
    return int(torch.multinomial(logp.exp(), num_samples=1, generator=generator).item())


def sample_token_rows_from_logp(logp, temperature, generator):
    import torch

    if logp.dim() == 1:
        logp = logp.unsqueeze(0)
    normalized = normalize_log_target(logp, temperature)
    if temperature <= 0:
        return torch.argmax(normalized, dim=-1)
    return torch.multinomial(normalized.exp(), num_samples=1, generator=generator).squeeze(-1)


def past_key_value_tensors(past_key_values):
    import torch

    if past_key_values is None:
        return []
    if hasattr(past_key_values, "layers"):
        tensors = []
        for layer in past_key_values.layers:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if torch.is_tensor(value):
                    tensors.append(value)
        return tensors
    tensors = []

    def visit(item):
        if torch.is_tensor(item):
            tensors.append(item)
        elif isinstance(item, (tuple, list)):
            for value in item:
                visit(value)

    visit(past_key_values)
    return tensors


def clone_repeat_past_key_values(past_key_values, repeats):
    import torch

    if past_key_values is None:
        return None
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if hasattr(past_key_values, "layers"):
        cloned = copy.copy(past_key_values)
        cloned.layers = []
        source_tensors = past_key_value_tensors(past_key_values)
        source_batch = source_tensors[0].shape[0] if source_tensors else 1
        for layer in past_key_values.layers:
            cloned_layer = copy.copy(layer)
            for name, value in vars(layer).items():
                if torch.is_tensor(value):
                    copied = value.clone()
                    if copied.dim() > 0 and copied.shape[0] == source_batch:
                        copied = copied.repeat_interleave(repeats, dim=0)
                    setattr(cloned_layer, name, copied)
            cloned.layers.append(cloned_layer)
        return cloned

    def repeat_item(item):
        if torch.is_tensor(item):
            return item.clone().repeat_interleave(repeats, dim=0)
        if isinstance(item, tuple):
            return tuple(repeat_item(value) for value in item)
        if isinstance(item, list):
            return [repeat_item(value) for value in item]
        return copy.deepcopy(item)

    return repeat_item(past_key_values)


def select_past_key_value_rows(past_key_values, indices):
    import torch

    if past_key_values is None:
        return None
    if hasattr(past_key_values, "batch_select_indices"):
        selected = copy.copy(past_key_values)
        selected.batch_select_indices(indices)
        return selected

    def select_item(item):
        if torch.is_tensor(item):
            return item.index_select(0, indices.to(item.device))
        if isinstance(item, tuple):
            return tuple(select_item(value) for value in item)
        if isinstance(item, list):
            return [select_item(value) for value in item]
        return item

    return select_item(past_key_values)


def sample_ref_span_proposals(model, base_past, base_attention_mask, first_logp, max_span_len,
                              n_proposals, stop_ids, temperature, generator, sample_device):
    import torch

    spans = []
    for _ in range(n_proposals):
        past = clone_repeat_past_key_values(base_past, 1)
        attention_mask = base_attention_mask
        logp = first_logp
        span = []
        for k in range(max_span_len):
            token_id = sample_token_from_logp(logp, temperature, generator, sample_device)
            span.append(token_id)
            if token_id in stop_ids or k + 1 >= max_span_len:
                break
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=first_logp.device)
            extra_attention = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, extra_attention], dim=-1)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        if span:
            spans.append(tuple(span))
    return spans


def sample_ref_span_proposals_cached_batched(model, base_past, base_attention_mask,
                                             first_logp, max_span_len, n_proposals,
                                             stop_ids, temperature, generator):
    import torch

    if n_proposals < 1 or max_span_len < 1:
        return [], []

    device = first_logp.device
    first_rows = first_logp.expand(n_proposals, -1)
    sampled = sample_token_rows_from_logp(first_rows, temperature, generator)
    row_ids = torch.arange(n_proposals, dtype=torch.long, device=device)
    support_logps = first_rows.gather(1, sampled[:, None]).squeeze(1).to(torch.float32)
    spans = [[int(token_id)] for token_id in sampled.tolist()]
    stop_tensor = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
    if stop_tensor.numel():
        active_mask = ~torch.isin(sampled, stop_tensor)
    else:
        active_mask = torch.ones_like(sampled, dtype=torch.bool)
    if max_span_len == 1 or not bool(active_mask.any().item()):
        return [tuple(span) for span in spans], support_logps.tolist()

    branch_past = clone_repeat_past_key_values(base_past, n_proposals)
    attention_mask = base_attention_mask.repeat_interleave(n_proposals, dim=0)
    active_positions = torch.nonzero(active_mask, as_tuple=False).flatten()
    if active_positions.numel() != n_proposals:
        branch_past = select_past_key_value_rows(branch_past, active_positions)
        attention_mask = attention_mask.index_select(0, active_positions)
    active_rows = row_ids.index_select(0, active_positions)
    previous_tokens = sampled.index_select(0, active_positions)

    for _ in range(1, max_span_len):
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ], dim=-1)
        out = model(
            input_ids=previous_tokens[:, None],
            attention_mask=attention_mask,
            past_key_values=branch_past,
            use_cache=True,
        )
        branch_past = out.past_key_values
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        next_tokens = sample_token_rows_from_logp(logp, temperature, generator)
        next_support = logp.gather(1, next_tokens[:, None]).squeeze(1).to(torch.float32)
        support_logps.index_add_(0, active_rows, next_support)
        for row_id, token_id in zip(active_rows.tolist(), next_tokens.tolist()):
            spans[row_id].append(int(token_id))

        if stop_tensor.numel():
            survivor_mask = ~torch.isin(next_tokens, stop_tensor)
        else:
            survivor_mask = torch.ones_like(next_tokens, dtype=torch.bool)
        if not bool(survivor_mask.any().item()):
            break
        survivor_positions = torch.nonzero(survivor_mask, as_tuple=False).flatten()
        if survivor_positions.numel() != next_tokens.numel():
            branch_past = select_past_key_value_rows(branch_past, survivor_positions)
            attention_mask = attention_mask.index_select(0, survivor_positions)
        active_rows = active_rows.index_select(0, survivor_positions)
        previous_tokens = next_tokens.index_select(0, survivor_positions)

    return [tuple(span) for span in spans], support_logps.tolist()


def score_span_logprob(model, base_past, base_attention_mask, first_logp, span_ids):
    import torch

    if not span_ids:
        raise ValueError("span_ids must be non-empty")
    total = float(first_logp[0, int(span_ids[0])].item())
    if len(span_ids) == 1:
        return total

    past = clone_repeat_past_key_values(base_past, 1)
    attention_mask = base_attention_mask
    prev_id = int(span_ids[0])
    for token_id in span_ids[1:]:
        input_ids = torch.tensor([[prev_id]], dtype=torch.long, device=first_logp.device)
        extra_attention = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, extra_attention], dim=-1)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        total += float(logp[0, int(token_id)].item())
        prev_id = int(token_id)
    return total


def next_logp_from_context_ids(model, context_ids, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
        return F.log_softmax(out.logits[:, -1, :].float(), dim=-1)


def sample_ref_span_proposals_from_context(model, context_ids, device, max_span_len,
                                           n_proposals, stop_ids, temperature,
                                           generator, sample_device):
    spans = []
    for _ in range(n_proposals):
        prefix = list(context_ids)
        span = []
        for _ in range(max_span_len):
            logp = next_logp_from_context_ids(model, prefix, device)
            token_id = sample_token_from_logp(logp, temperature, generator, sample_device)
            span.append(token_id)
            prefix.append(token_id)
            if token_id in stop_ids:
                break
        if span:
            spans.append(tuple(span))
    return spans


def score_span_logprob_from_context(model, context_ids, span_ids, device):
    import torch
    import torch.nn.functional as F

    if not span_ids:
        raise ValueError("span_ids must be non-empty")
    input_ids = list(context_ids) + list(span_ids[:-1])
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    targets = torch.tensor(list(span_ids), dtype=torch.long, device=device)
    start = len(context_ids) - 1
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, start:start + len(span_ids), :].float()
        logps = F.log_softmax(logits, dim=-1)
        return float(logps.gather(1, targets[:, None]).sum().item())


def pooled_span_embeddings(span_ids_list, embedding_weight, device):
    import torch
    import torch.nn.functional as F

    vectors = []
    weight_device = embedding_weight.device
    for span_ids in span_ids_list:
        ids = torch.tensor(list(span_ids), dtype=torch.long, device=weight_device)
        vec = embedding_weight.index_select(0, ids).float().mean(dim=0)
        vectors.append(F.normalize(vec, dim=0).to(device))
    return torch.stack(vectors, dim=0)


def embed_span_ids(span_ids_list, span_embedder, device):
    import torch
    import torch.nn.functional as F

    if span_embedder["source"] in {"input", "output"}:
        return pooled_span_embeddings(span_ids_list, span_embedder["weight"], device)
    if span_embedder["source"] == "sentence_transformer":
        texts = span_embedding_texts(
            span_embedder["tokenizer"],
            span_ids_list,
            span_embedder["text_mode"],
        )
        cache = span_embedder["cache"]
        cache_size = int(span_embedder["cache_size"])
        unique_texts = list(dict.fromkeys(texts))
        missing = []
        resolved_vectors = {}
        for text in unique_texts:
            if text in cache:
                cache.move_to_end(text)
                span_embedder["cache_hits"] += 1
                resolved_vectors[text] = cache[text]
            else:
                missing.append(text)
                span_embedder["cache_misses"] += 1

        if missing:
            started = time.time()
            vectors = span_embedder["model"].encode(
                missing,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            span_embedder["encode_seconds"] += time.time() - started
            span_embedder["encode_calls"] += 1
            for text, vector in zip(missing, vectors):
                normalized = F.normalize(
                    torch.tensor(vector, dtype=torch.float32),
                    dim=0,
                )
                resolved_vectors[text] = normalized
                if cache_size > 0:
                    cache[text] = normalized
                    cache.move_to_end(text)
                    while len(cache) > cache_size:
                        cache.popitem(last=False)

        return torch.stack([resolved_vectors[text] for text in texts], dim=0).to(device)
    raise ValueError(f"unknown span embedding source: {span_embedder['source']}")


def span_embedding_record_texts(tokenizer, span_ids_list, span_embedder):
    if span_embedder is None or span_embedder.get("source") in {"input", "output"}:
        return [None for _ in span_ids_list]
    return span_embedding_texts(tokenizer, span_ids_list, span_embedder["text_mode"])


def span_similarity_gate_factors(sims, similarity_gate="none", threshold=0.0, soft_beta=0.05):
    import torch

    if similarity_gate == "none":
        return torch.ones_like(sims)
    if similarity_gate == "hard":
        return (sims >= float(threshold)).to(dtype=sims.dtype)
    if similarity_gate == "soft":
        if soft_beta <= 0.0:
            raise ValueError(f"span_similarity_soft_beta must be positive, got {soft_beta}")
        return torch.sigmoid((sims - float(threshold)) / float(soft_beta))
    raise ValueError(f"unknown span_similarity_gate: {similarity_gate}")


def span_kernel_from_embeddings(embeddings, top_k, tau, similarity_gate="none",
                                similarity_threshold=0.0, similarity_soft_beta=0.05):
    import torch

    if top_k < 1:
        raise ValueError(f"span_kernel_top_k must be positive, got {top_k}")
    if tau <= 0.0:
        raise ValueError(f"span_kernel_tau must be positive, got {tau}")
    n = embeddings.shape[0]
    k = min(int(top_k), n)
    sims = embeddings @ embeddings.T
    values, indices = torch.topk(sims / float(tau), k=k, dim=-1, largest=True)
    weights = torch.softmax(values, dim=-1)
    selected_sims = sims.gather(1, indices)
    gate_factors = span_similarity_gate_factors(
        selected_sims,
        similarity_gate,
        similarity_threshold,
        similarity_soft_beta,
    )
    effective_weights = weights * gate_factors
    kernel = torch.zeros_like(sims)
    kernel.scatter_(1, indices, effective_weights)
    return kernel, sims


def span_neighbor_records(tokenizer, span_ids_list, embeddings, span_embedder, top_k, tau,
                          similarity_gate="none", similarity_threshold=0.0,
                          similarity_soft_beta=0.05):
    import torch

    if not span_ids_list:
        return []
    if top_k < 1:
        raise ValueError(f"span_kernel_top_k must be positive, got {top_k}")
    if tau <= 0.0:
        raise ValueError(f"span_kernel_tau must be positive, got {tau}")
    texts = span_embedding_record_texts(tokenizer, span_ids_list, span_embedder)
    decoded = [
        tokenizer.decode(list(span_ids), skip_special_tokens=False)
        for span_ids in span_ids_list
    ]
    sims = embeddings @ embeddings.T
    k = min(int(top_k), len(span_ids_list))
    values, indices = torch.topk(sims / float(tau), k=k, dim=-1, largest=True)
    weights = torch.softmax(values, dim=-1)
    selected_sims = sims.gather(1, indices)
    gate_factors = span_similarity_gate_factors(
        selected_sims,
        similarity_gate,
        similarity_threshold,
        similarity_soft_beta,
    )
    effective_weights = weights * gate_factors
    records = []
    for i, span_ids in enumerate(span_ids_list):
        neighbors = []
        for raw_value, idx, weight, gate_factor, effective_weight in zip(
            values[i].tolist(),
            indices[i].tolist(),
            weights[i].tolist(),
            gate_factors[i].tolist(),
            effective_weights[i].tolist(),
        ):
            idx = int(idx)
            neighbors.append({
                "index": idx,
                "span_repr": repr(decoded[idx]),
                "embedding_text": texts[idx],
                "similarity": float(raw_value * float(tau)),
                "kernel_weight": float(weight),
                "gate_factor": float(gate_factor),
                "effective_weight": float(effective_weight),
            })
        records.append({
            "index": int(i),
            "span_ids": [int(token_id) for token_id in span_ids],
            "span_repr": repr(decoded[i]),
            "embedding_text": texts[i],
            "neighbors": neighbors,
        })
    return records


def normalize_span_logp(logp, span_ids, normalization):
    if normalization == "full":
        return float(logp)
    if normalization == "length":
        return float(logp) / float(len(span_ids))
    raise ValueError(f"unknown span support normalization: {normalization}")


def resolved_span_support_normalization(args, composition_type=None):
    if args.span_support_normalization:
        return args.span_support_normalization
    composition_type = composition_type or args.composition_type
    if composition_type in {"span_token_smoothing", "span_token_smoothing_delta"}:
        return "length"
    return "full"


def resolved_span_token_lambda(args):
    return args.span_kernel_lambda if args.span_token_lambda is None else args.span_token_lambda


def span_similarity_gate_label(args):
    if args.span_similarity_gate == "hard":
        return f"hard_s{args.span_similarity_threshold:g}"
    if args.span_similarity_gate == "soft":
        return (
            f"soft_s{args.span_similarity_threshold:g}_"
            f"b{args.span_similarity_soft_beta:g}"
        )
    return "none"


def compose_span_action_scores(exact_supports, sponsor_mask, embeddings, quorum_q,
                               kernel_lambda, kernel_top_k, kernel_tau):
    import torch

    if not (0.0 <= kernel_lambda <= 1.0):
        raise ValueError(f"span_kernel_lambda must be in [0, 1], got {kernel_lambda}")
    m = exact_supports.shape[0]
    if quorum_q < 1 or quorum_q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {quorum_q}")

    kernel, _ = span_kernel_from_embeddings(embeddings, kernel_top_k, kernel_tau)
    neighbor_supports = torch.zeros_like(exact_supports)
    for i in range(m):
        source_support = exact_supports[i] * sponsor_mask[i]
        neighbor_supports[i] = kernel.T.matmul(source_support)
    smoothed = (1.0 - float(kernel_lambda)) * exact_supports + float(kernel_lambda) * neighbor_supports
    return torch.topk(smoothed, k=quorum_q, dim=0, largest=True).values[-1], smoothed, neighbor_supports


def sample_span_from_scores(candidate_spans, scores, temperature, generator):
    import torch

    scores = scores.clamp_min(torch.finfo(scores.dtype).tiny)
    log_scores = normalize_log_target(scores.log().unsqueeze(0), temperature).squeeze(0)
    if temperature <= 0:
        choice = int(torch.argmax(log_scores).item())
    else:
        choice = int(torch.multinomial(log_scores.exp(), num_samples=1, generator=generator).item())
    return list(candidate_spans[choice]), choice, log_scores.exp()


def top_token_records(tokenizer, logp, top_k):
    import torch

    flat = logp.squeeze(0) if logp.dim() == 2 else logp
    k = min(int(top_k), flat.shape[-1])
    values, indices = torch.topk(flat.exp(), k=k, largest=True)
    return [
        {
            "token_id": int(token_id),
            "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
            "prob": float(prob),
        }
        for prob, token_id in zip(values.tolist(), indices.tolist())
    ]


def span_trace_records(tokenizer, candidate_spans, scores, probs, per_ref_proposals,
                       neutral_proposals=None, top_k=12, span_embedder=None):
    import torch

    sponsor_sets = [set(proposals) for proposals in per_ref_proposals]
    neutral_set = set(neutral_proposals or [])
    embedding_texts = span_embedding_record_texts(tokenizer, candidate_spans, span_embedder)
    order = torch.argsort(probs, descending=True).tolist()
    records = []
    for j in order[:top_k]:
        span = candidate_spans[j]
        decoded = tokenizer.decode(list(span), skip_special_tokens=False)
        records.append({
            "index": int(j),
            "span_ids": [int(token_id) for token_id in span],
            "span_repr": repr(decoded),
            "embedding_text": embedding_texts[j],
            "n_tokens": len(span),
            "raw_score": float(scores[j].item()),
            "prob": float(probs[j].item()),
            "ref_sponsors": [
                int(i) for i, sponsored in enumerate(sponsor_sets) if span in sponsored
            ],
            "neutral_sponsor": span in neutral_set,
        })
    return records


def fallback_quorum_token_from_context(context_ids, refs, args, generator, compose_device):
    import torch

    step_logps = []
    for ref in refs:
        step_logps.append(
            next_logp_from_context_ids(ref["model"], context_ids, ref["device"]).to(compose_device)
        )
    logps = torch.stack(step_logps, dim=0)
    logp_target = compose_quorum_log_probs_from_logps(logps, args.quorum_q, args.temperature)
    if args.temperature <= 0:
        return int(torch.argmax(logp_target, dim=-1).item()), logp_target
    return int(torch.multinomial(logp_target.exp(), num_samples=1, generator=generator).item()), logp_target


def sample_span_action(states, step_logps_by_ref, refs, tokenizer, args, stop_ids,
                       generator, compose_device, span_embedder, max_span_len):
    import torch

    per_ref_proposals = []
    candidate_spans = []
    seen = set()
    for ref_index, state in enumerate(states):
        spans = sample_ref_span_proposals(
            state["ref"]["model"],
            state["past_key_values"],
            state["attention_mask"],
            step_logps_by_ref[ref_index],
            max_span_len,
            args.span_proposals_per_ref,
            stop_ids,
            args.temperature,
            generator,
            compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            if span not in seen:
                seen.add(span)
                candidate_spans.append(span)

    if not candidate_spans:
        raise RuntimeError("span-action proposal pool is empty")

    m = len(refs)
    n = len(candidate_spans)
    exact_logps = torch.empty((m, n), dtype=torch.float32, device=compose_device)
    sponsor_mask = torch.zeros((m, n), dtype=torch.float32, device=compose_device)
    for i, state in enumerate(states):
        sponsored = set(per_ref_proposals[i])
        for j, span in enumerate(candidate_spans):
            logp = score_span_logprob(
                state["ref"]["model"],
                state["past_key_values"],
                state["attention_mask"],
                step_logps_by_ref[i],
                span,
            )
            exact_logps[i, j] = logp / float(len(span))
            if span in sponsored:
                sponsor_mask[i, j] = 1.0

    exact_supports = exact_logps.exp()
    embeddings = embed_span_ids(candidate_spans, span_embedder, compose_device)
    scores, _, _ = compose_span_action_scores(
        exact_supports,
        sponsor_mask,
        embeddings,
        args.quorum_q,
        args.span_kernel_lambda,
        args.span_kernel_top_k,
        args.span_kernel_tau,
    )
    scores = scores.clamp_min(torch.finfo(scores.dtype).tiny)
    log_scores = scores.log().unsqueeze(0)
    log_scores = normalize_log_target(log_scores, args.temperature).squeeze(0)
    if args.temperature <= 0:
        choice = int(torch.argmax(log_scores).item())
    else:
        choice = int(torch.multinomial(log_scores.exp(), num_samples=1, generator=generator).item())
    return list(candidate_spans[choice])


def sample_span_action_from_context(context_ids, refs, args, stop_ids, generator,
                                    compose_device, span_embedder, max_span_len):
    import torch

    per_ref_proposals = []
    candidate_spans = []
    seen = set()
    for ref in refs:
        spans = sample_ref_span_proposals_from_context(
            ref["model"],
            context_ids,
            ref["device"],
            max_span_len,
            args.span_proposals_per_ref,
            stop_ids,
            args.temperature,
            generator,
            compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            if span not in seen:
                seen.add(span)
                candidate_spans.append(span)

    if not candidate_spans:
        raise RuntimeError("span-action proposal pool is empty")

    m = len(refs)
    n = len(candidate_spans)
    exact_logps = torch.empty((m, n), dtype=torch.float32, device=compose_device)
    sponsor_mask = torch.zeros((m, n), dtype=torch.float32, device=compose_device)
    for i, ref in enumerate(refs):
        sponsored = set(per_ref_proposals[i])
        for j, span in enumerate(candidate_spans):
            logp = score_span_logprob_from_context(ref["model"], context_ids, span, ref["device"])
            exact_logps[i, j] = logp / float(len(span))
            if span in sponsored:
                sponsor_mask[i, j] = 1.0

    exact_supports = exact_logps.exp()
    embeddings = embed_span_ids(candidate_spans, span_embedder, compose_device)
    scores, _, _ = compose_span_action_scores(
        exact_supports,
        sponsor_mask,
        embeddings,
        args.quorum_q,
        args.span_kernel_lambda,
        args.span_kernel_top_k,
        args.span_kernel_tau,
    )
    scores = scores.clamp_min(torch.finfo(scores.dtype).tiny)
    log_scores = normalize_log_target(scores.log().unsqueeze(0), args.temperature).squeeze(0)
    if args.temperature <= 0:
        choice = int(torch.argmax(log_scores).item())
    else:
        choice = int(torch.multinomial(log_scores.exp(), num_samples=1, generator=generator).item())
    return list(candidate_spans[choice])


def sample_span_action_neutral_from_context(context_ids, refs, neutral_model, neutral_device,
                                            tokenizer, args, stop_ids, generator,
                                            compose_device, span_embedder,
                                            max_span_len, trace_enabled=False):
    import torch

    if neutral_model is None:
        raise ValueError("neutral_model is required for span_action_neutral")

    per_ref_proposals = []
    candidate_spans = []
    seen = set()

    neutral_proposals = sample_ref_span_proposals_from_context(
        neutral_model,
        context_ids,
        neutral_device,
        max_span_len,
        args.neutral_proposals,
        stop_ids,
        args.temperature,
        generator,
        compose_device,
    )
    for span in neutral_proposals:
        if span not in seen:
            seen.add(span)
            candidate_spans.append(span)

    for ref in refs:
        spans = sample_ref_span_proposals_from_context(
            ref["model"],
            context_ids,
            ref["device"],
            max_span_len,
            args.span_proposals_per_ref,
            stop_ids,
            args.temperature,
            generator,
            compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            if span not in seen:
                seen.add(span)
                candidate_spans.append(span)

    if not candidate_spans:
        token_id, logp_target = fallback_quorum_token_from_context(context_ids, refs, args, generator, compose_device)
        trace = None
        if trace_enabled:
            trace = {
                "algorithm": "span_action_neutral",
                "fallback": "empty_candidate_pool",
                "top_tokens": top_token_records(tokenizer, logp_target, args.trace_top_k),
            }
        return [token_id], trace

    m = len(refs)
    n = len(candidate_spans)
    normalization = resolved_span_support_normalization(args, "span_action_neutral")
    exact_logps = torch.empty((m, n), dtype=torch.float32, device=compose_device)
    sponsor_mask = torch.zeros((m, n), dtype=torch.float32, device=compose_device)
    for i, ref in enumerate(refs):
        sponsored = set(per_ref_proposals[i])
        for j, span in enumerate(candidate_spans):
            logp = score_span_logprob_from_context(ref["model"], context_ids, span, ref["device"])
            exact_logps[i, j] = normalize_span_logp(logp, span, normalization)
            if span in sponsored:
                sponsor_mask[i, j] = 1.0

    exact_supports = exact_logps.exp()
    embeddings = embed_span_ids(candidate_spans, span_embedder, compose_device)
    scores, smoothed, neighbor_supports = compose_span_action_scores(
        exact_supports,
        sponsor_mask,
        embeddings,
        args.quorum_q,
        args.span_kernel_lambda,
        args.span_kernel_top_k,
        args.span_kernel_tau,
    )
    valid = scores >= float(args.span_support_floor)
    if not bool(valid.any().item()):
        token_id, logp_target = fallback_quorum_token_from_context(context_ids, refs, args, generator, compose_device)
        trace = None
        if trace_enabled:
            trace = {
                "algorithm": "span_action_neutral",
                "fallback": "support_floor",
                "span_support_floor": args.span_support_floor,
                "n_candidates": len(candidate_spans),
                "max_score": float(scores.max().item()) if len(candidate_spans) else 0.0,
                "top_spans": span_trace_records(
                    tokenizer,
                    candidate_spans,
                    scores,
                    normalize_log_target(scores.clamp_min(torch.finfo(scores.dtype).tiny).log().unsqueeze(0), args.temperature).exp().squeeze(0),
                    per_ref_proposals,
                    neutral_proposals=neutral_proposals,
                    top_k=args.trace_top_k,
                    span_embedder=span_embedder,
                ),
                "span_neighbors": span_neighbor_records(
                    tokenizer,
                    candidate_spans,
                    embeddings,
                    span_embedder,
                    args.trace_top_k,
                    args.span_kernel_tau,
                ),
                "top_tokens": top_token_records(tokenizer, logp_target, args.trace_top_k),
            }
        return [token_id], trace

    filtered_spans = [span for span, keep in zip(candidate_spans, valid.tolist()) if keep]
    filtered_scores = scores[valid]
    selected_span, selected_index, probs = sample_span_from_scores(
        filtered_spans,
        filtered_scores,
        args.temperature,
        generator,
    )

    trace = None
    if trace_enabled:
        full_probs = torch.zeros_like(scores)
        full_probs[valid] = probs
        original_index = [j for j, keep in enumerate(valid.tolist()) if keep][selected_index]
        trace = {
            "algorithm": "span_action_neutral",
            "fallback": None,
            "support_normalization": normalization,
            "span_support_floor": args.span_support_floor,
            "n_candidates": len(candidate_spans),
            "n_valid_candidates": len(filtered_spans),
            "selected_index": int(original_index),
            "selected_span_repr": repr(tokenizer.decode(selected_span, skip_special_tokens=False)),
            "neutral_proposals": [
                repr(tokenizer.decode(list(span), skip_special_tokens=False))
                for span in neutral_proposals
            ],
            "top_spans": span_trace_records(
                tokenizer,
                candidate_spans,
                scores,
                full_probs,
                per_ref_proposals,
                neutral_proposals=neutral_proposals,
                top_k=args.trace_top_k,
                span_embedder=span_embedder,
            ),
            "span_neighbors": span_neighbor_records(
                tokenizer,
                candidate_spans,
                embeddings,
                span_embedder,
                args.trace_top_k,
                args.span_kernel_tau,
            ),
            "exact_supports_top": [
                [float(exact_supports[i, j].item()) for i in range(m)]
                for j in torch.argsort(full_probs, descending=True).tolist()[:args.trace_top_k]
            ],
            "neighbor_supports_top": [
                [float(neighbor_supports[i, j].item()) for i in range(m)]
                for j in torch.argsort(full_probs, descending=True).tolist()[:args.trace_top_k]
            ],
            "smoothed_supports_top": [
                [float(smoothed[i, j].item()) for i in range(m)]
                for j in torch.argsort(full_probs, descending=True).tolist()[:args.trace_top_k]
            ],
        }
    return selected_span, trace


def span_token_pseudo_support_from_embeddings(ref_indices, first_tokens, supports, embeddings,
                                              m, vocab_size, top_k, tau, cross_only,
                                              device, similarity_gate="none",
                                              similarity_threshold=0.0,
                                              similarity_soft_beta=0.05):
    import torch

    if tau <= 0.0:
        raise ValueError(f"span_kernel_tau must be positive, got {tau}")
    pseudo = torch.zeros((m, vocab_size), dtype=torch.float32, device=device)
    if len(ref_indices) < 2:
        return pseudo

    ref_tensor = torch.tensor(ref_indices, dtype=torch.long, device=device)
    token_tensor = torch.tensor(first_tokens, dtype=torch.long, device=device)
    support_tensor = torch.tensor(supports, dtype=torch.float32, device=device)
    n = len(ref_indices)
    for source_idx in range(n):
        source_ref = int(ref_indices[source_idx])
        if cross_only:
            mask = ref_tensor != source_ref
        else:
            mask = torch.ones((n,), dtype=torch.bool, device=device)
            mask[source_idx] = False
        target_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue
        sims = embeddings[source_idx].unsqueeze(0) @ embeddings.index_select(0, target_indices).T
        sims = sims.squeeze(0)
        k = min(int(top_k), int(target_indices.numel()))
        values, local_indices = torch.topk(sims / float(tau), k=k, largest=True)
        weights = torch.softmax(values, dim=-1)
        selected_targets = target_indices.index_select(0, local_indices)
        selected_sims = sims.index_select(0, local_indices)
        gate_factors = span_similarity_gate_factors(
            selected_sims,
            similarity_gate,
            similarity_threshold,
            similarity_soft_beta,
        )
        effective_weights = weights * gate_factors
        target_tokens = token_tensor.index_select(0, selected_targets)
        pseudo[source_ref].scatter_add_(
            0,
            target_tokens,
            support_tensor[source_idx] * effective_weights.to(torch.float32),
        )
    return pseudo


def span_token_pseudo_support_edge_records(tokenizer, ref_indices, first_tokens, supports,
                                           span_ids_list, embeddings, ref_names,
                                           span_embedder, top_k, tau, cross_only,
                                           similarity_gate="none",
                                           similarity_threshold=0.0,
                                           similarity_soft_beta=0.05):
    import torch

    if tau <= 0.0:
        raise ValueError(f"span_kernel_tau must be positive, got {tau}")
    if len(ref_indices) < 2:
        return []
    ref_tensor = torch.tensor(ref_indices, dtype=torch.long, device=embeddings.device)
    token_tensor = torch.tensor(first_tokens, dtype=torch.long, device=embeddings.device)
    support_tensor = torch.tensor(supports, dtype=torch.float32, device=embeddings.device)
    texts = span_embedding_record_texts(tokenizer, span_ids_list, span_embedder)
    decoded = [
        tokenizer.decode(list(span_ids), skip_special_tokens=False)
        for span_ids in span_ids_list
    ]
    edges = []
    for source_idx in range(len(ref_indices)):
        source_ref = int(ref_indices[source_idx])
        if cross_only:
            mask = ref_tensor != source_ref
        else:
            mask = torch.ones((len(ref_indices),), dtype=torch.bool, device=embeddings.device)
            mask[source_idx] = False
        target_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue
        sims = embeddings[source_idx].unsqueeze(0) @ embeddings.index_select(0, target_indices).T
        sims = sims.squeeze(0)
        k = min(int(top_k), int(target_indices.numel()))
        values, local_indices = torch.topk(sims / float(tau), k=k, largest=True)
        weights = torch.softmax(values, dim=-1)
        selected_targets = target_indices.index_select(0, local_indices)
        selected_sims = sims.index_select(0, local_indices)
        gate_factors = span_similarity_gate_factors(
            selected_sims,
            similarity_gate,
            similarity_threshold,
            similarity_soft_beta,
        )
        effective_weights = weights * gate_factors
        for raw_value, weight, gate_factor, effective_weight, target_idx in zip(
            values.tolist(),
            weights.tolist(),
            gate_factors.tolist(),
            effective_weights.tolist(),
            selected_targets.tolist(),
        ):
            target_idx = int(target_idx)
            target_ref = int(ref_indices[target_idx])
            target_token = int(token_tensor[target_idx].item())
            support = float(support_tensor[source_idx].item())
            edges.append({
                "source_index": int(source_idx),
                "source_ref": ref_names[source_ref],
                "source_span_repr": repr(decoded[source_idx]),
                "source_embedding_text": texts[source_idx],
                "source_support": support,
                "target_index": target_idx,
                "target_ref": ref_names[target_ref],
                "target_span_repr": repr(decoded[target_idx]),
                "target_embedding_text": texts[target_idx],
                "target_first_token_id": target_token,
                "target_first_token_repr": repr(tokenizer.decode([target_token], skip_special_tokens=False)),
                "similarity": float(raw_value * float(tau)),
                "kernel_weight": float(weight),
                "gate_factor": float(gate_factor),
                "effective_weight": float(effective_weight),
                "pseudo_added": float(support * float(effective_weight)),
            })
    edges.sort(key=lambda item: item["pseudo_added"], reverse=True)
    return edges


def apply_span_token_pseudo_support(logps, pseudo_supports, span_token_lambda):
    import torch

    if not (0.0 <= span_token_lambda):
        raise ValueError(f"span_token_lambda must be nonnegative, got {span_token_lambda}")
    if logps.shape[1] != 1:
        raise ValueError("span_token_smoothing currently expects batch size 1")
    probs = logps.exp().squeeze(1).to(dtype=torch.float32)
    pseudo = pseudo_supports.to(device=probs.device, dtype=probs.dtype)
    smoothed = probs + float(span_token_lambda) * pseudo
    smoothed = smoothed.clamp_min(torch.finfo(smoothed.dtype).tiny)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
    return smoothed.log().unsqueeze(1)


def compose_span_token_smoothed_log_probs_from_context(logps, refs, context_ids, tokenizer,
                                                       args, stop_ids, generator,
                                                       compose_device, span_embedder,
                                                       max_span_len, trace_enabled=False,
                                                       base_logp=None):
    per_ref_proposals = []
    row_ref_indices = []
    row_first_tokens = []
    row_supports = []
    row_spans = []
    normalization = resolved_span_support_normalization(args, "span_token_smoothing")

    for i, ref in enumerate(refs):
        spans = sample_ref_span_proposals_from_context(
            ref["model"],
            context_ids,
            ref["device"],
            max_span_len,
            args.span_proposals_per_ref,
            stop_ids,
            args.temperature,
            generator,
            compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            logp = score_span_logprob_from_context(ref["model"], context_ids, span, ref["device"])
            row_ref_indices.append(i)
            row_first_tokens.append(int(span[0]))
            row_supports.append(math.exp(normalize_span_logp(logp, span, normalization)))
            row_spans.append(span)

    vocab_size = logps.shape[-1]
    if row_spans:
        embeddings = embed_span_ids(row_spans, span_embedder, compose_device)
        pseudo = span_token_pseudo_support_from_embeddings(
            row_ref_indices,
            row_first_tokens,
            row_supports,
            embeddings,
            len(refs),
            vocab_size,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            compose_device,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )
    else:
        import torch

        pseudo = torch.zeros((len(refs), vocab_size), dtype=torch.float32, device=compose_device)
        embeddings = None

    span_lambda = resolved_span_token_lambda(args)
    smoothed_logps = apply_span_token_pseudo_support(logps, pseudo, span_lambda)
    if args.composition_type == "span_token_smoothing_delta":
        logp_target = compose_pi_min_delta_log_probs_from_logps(
            smoothed_logps,
            base_logp,
            args.temperature,
        )
    else:
        logp_target = compose_quorum_log_probs_from_logps(smoothed_logps, args.quorum_q, args.temperature)

    trace = None
    if trace_enabled:
        import torch

        pseudo_records = []
        for i, ref in enumerate(refs):
            values, token_ids = torch.topk(
                pseudo[i],
                k=min(args.trace_top_k, pseudo.shape[-1]),
                largest=True,
            )
            pseudo_records.append({
                "ref": ref["name"],
                "top_updates": [
                    {
                        "token_id": int(token_id),
                        "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
                        "pseudo_support": float(value),
                    }
                    for value, token_id in zip(values.tolist(), token_ids.tolist())
                    if value > 0
                ],
            })
        pseudo_edges = [] if embeddings is None else span_token_pseudo_support_edge_records(
            tokenizer,
            row_ref_indices,
            row_first_tokens,
            row_supports,
            row_spans,
            embeddings,
            [ref["name"] for ref in refs],
            span_embedder,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )[:args.trace_top_k]
        trace = {
            "algorithm": "span_token_smoothing",
            "support_normalization": normalization,
            "span_token_lambda": span_lambda,
            "cross_only": bool(args.span_token_cross_only),
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "n_proposal_rows": len(row_spans),
            "ref_proposals": {
                ref["name"]: [
                    repr(tokenizer.decode(list(span), skip_special_tokens=False))
                    for span in proposals
                ]
                for ref, proposals in zip(refs, per_ref_proposals)
            },
            "proposal_neighbors": [] if embeddings is None else span_neighbor_records(
                tokenizer,
                row_spans,
                embeddings,
                span_embedder,
                args.trace_top_k,
                args.span_kernel_tau,
                args.span_similarity_gate,
                args.span_similarity_threshold,
                args.span_similarity_soft_beta,
            ),
            "pseudo_support": pseudo_records,
            "pseudo_support_edges": pseudo_edges,
            "top_tokens": top_token_records(tokenizer, logp_target, args.trace_top_k),
        }
    return logp_target, trace


def span_token_parallel_refs_enabled(args, refs):
    mode = getattr(args, "span_token_parallel_refs", "auto")
    if mode == "off":
        return False
    if mode == "on":
        return True
    devices = [str(ref["device"]) for ref in refs]
    return (
        len(set(devices)) == len(devices)
        and all(device.startswith("cuda") for device in devices)
    )


def synchronize_span_devices(refs, compose_device, span_embedder):
    import torch

    devices = {str(compose_device)}
    devices.update(str(ref["device"]) for ref in refs)
    if span_embedder is not None and span_embedder.get("device"):
        devices.add(str(span_embedder["device"]))
    for device in sorted(devices):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)


def record_span_token_profile(args, values):
    stats = getattr(args, "_span_token_profile_stats", None)
    if stats is None:
        stats = {"steps": 0}
        args._span_token_profile_stats = stats
    stats["steps"] += 1
    for key, value in values.items():
        stats[key] = stats.get(key, 0.0) + float(value)


def span_token_profile_meta(args):
    stats = getattr(args, "_span_token_profile_stats", None)
    if not stats:
        return None
    steps = int(stats["steps"])
    meta = {"steps": steps}
    for key, value in stats.items():
        if key == "steps":
            continue
        meta[f"{key}_seconds"] = round(float(value), 6)
        meta[f"{key}_ms_per_step"] = round(1000.0 * float(value) / max(steps, 1), 3)
    return meta


def compose_span_token_smoothed_log_probs_cached(logps, states, step_logps_by_ref, refs,
                                                 tokenizer, args, stop_ids, generator,
                                                 compose_device, span_embedder,
                                                 max_span_len, trace_enabled=False,
                                                 base_logp=None):
    import torch

    per_ref_proposals = []
    row_ref_indices = []
    row_first_tokens = []
    row_supports = []
    row_spans = []
    normalization = resolved_span_support_normalization(args, "span_token_smoothing")

    for i, state in enumerate(states):
        spans = sample_ref_span_proposals(
            state["ref"]["model"],
            state["past_key_values"],
            state["attention_mask"],
            step_logps_by_ref[i],
            max_span_len,
            args.span_proposals_per_ref,
            stop_ids,
            args.temperature,
            generator,
            compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            logp = score_span_logprob(
                state["ref"]["model"],
                state["past_key_values"],
                state["attention_mask"],
                step_logps_by_ref[i],
                span,
            )
            row_ref_indices.append(i)
            row_first_tokens.append(int(span[0]))
            row_supports.append(math.exp(normalize_span_logp(logp, span, normalization)))
            row_spans.append(span)

    vocab_size = logps.shape[-1]
    if row_spans:
        embeddings = embed_span_ids(row_spans, span_embedder, compose_device)
        pseudo = span_token_pseudo_support_from_embeddings(
            row_ref_indices,
            row_first_tokens,
            row_supports,
            embeddings,
            len(refs),
            vocab_size,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            compose_device,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )
    else:
        pseudo = torch.zeros((len(refs), vocab_size), dtype=torch.float32, device=compose_device)
        embeddings = None

    span_lambda = resolved_span_token_lambda(args)
    smoothed_logps = apply_span_token_pseudo_support(logps, pseudo, span_lambda)
    if args.composition_type == "span_token_smoothing_delta":
        logp_target = compose_pi_min_delta_log_probs_from_logps(
            smoothed_logps,
            base_logp,
            args.temperature,
        )
    else:
        logp_target = compose_quorum_log_probs_from_logps(smoothed_logps, args.quorum_q, args.temperature)

    trace = None
    if trace_enabled:
        pseudo_records = []
        for i, ref in enumerate(refs):
            values, token_ids = torch.topk(
                pseudo[i],
                k=min(args.trace_top_k, pseudo.shape[-1]),
                largest=True,
            )
            pseudo_records.append({
                "ref": ref["name"],
                "top_updates": [
                    {
                        "token_id": int(token_id),
                        "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
                        "pseudo_support": float(value),
                    }
                    for value, token_id in zip(values.tolist(), token_ids.tolist())
                    if value > 0
                ],
            })
        pseudo_edges = [] if embeddings is None else span_token_pseudo_support_edge_records(
            tokenizer,
            row_ref_indices,
            row_first_tokens,
            row_supports,
            row_spans,
            embeddings,
            [ref["name"] for ref in refs],
            span_embedder,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )[:args.trace_top_k]
        trace = {
            "algorithm": "span_token_smoothing",
            "implementation": "cached",
            "support_normalization": normalization,
            "span_token_lambda": span_lambda,
            "cross_only": bool(args.span_token_cross_only),
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "n_proposal_rows": len(row_spans),
            "ref_proposals": {
                ref["name"]: [
                    repr(tokenizer.decode(list(span), skip_special_tokens=False))
                    for span in proposals
                ]
                for ref, proposals in zip(refs, per_ref_proposals)
            },
            "proposal_neighbors": [] if embeddings is None else span_neighbor_records(
                tokenizer,
                row_spans,
                embeddings,
                span_embedder,
                args.trace_top_k,
                args.span_kernel_tau,
                args.span_similarity_gate,
                args.span_similarity_threshold,
                args.span_similarity_soft_beta,
            ),
            "pseudo_support": pseudo_records,
            "pseudo_support_edges": pseudo_edges,
            "top_tokens": top_token_records(tokenizer, logp_target, args.trace_top_k),
        }
    return logp_target, trace


def compose_span_token_smoothed_log_probs_cached_batched(logps, states, refs,
                                                        tokenizer, args, stop_ids,
                                                        proposal_generators,
                                                        compose_device, span_embedder,
                                                        max_span_len,
                                                        trace_enabled=False,
                                                        base_logp=None):
    import torch

    profile_enabled = bool(getattr(args, "span_token_profile", False))
    if profile_enabled:
        synchronize_span_devices(refs, compose_device, span_embedder)
    total_started = time.perf_counter()
    proposal_started = total_started

    def propose(ref_index):
        with torch.inference_mode():
            state = states[ref_index]
            spans, support_logps = sample_ref_span_proposals_cached_batched(
                state["ref"]["model"],
                state["past_key_values"],
                state["attention_mask"],
                state["step_logp"],
                max_span_len,
                args.span_proposals_per_ref,
                stop_ids,
                args.temperature,
                proposal_generators[ref_index],
            )
        return ref_index, spans, support_logps

    proposal_results = [None for _ in refs]
    if span_token_parallel_refs_enabled(args, refs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(refs)) as executor:
            futures = [executor.submit(propose, i) for i in range(len(refs))]
            for future in futures:
                ref_index, spans, support_logps = future.result()
                proposal_results[ref_index] = (spans, support_logps)
    else:
        for i in range(len(refs)):
            ref_index, spans, support_logps = propose(i)
            proposal_results[ref_index] = (spans, support_logps)

    if profile_enabled:
        synchronize_span_devices(refs, compose_device, span_embedder)
    proposal_seconds = time.perf_counter() - proposal_started

    per_ref_proposals = []
    row_ref_indices = []
    row_first_tokens = []
    row_supports = []
    row_spans = []
    normalization = resolved_span_support_normalization(args, "span_token_smoothing")
    for ref_index, result in enumerate(proposal_results):
        spans, support_logps = result
        per_ref_proposals.append(spans)
        for span, support_logp in zip(spans, support_logps):
            row_ref_indices.append(ref_index)
            row_first_tokens.append(int(span[0]))
            row_supports.append(math.exp(normalize_span_logp(support_logp, span, normalization)))
            row_spans.append(span)

    embedding_started = time.perf_counter()
    vocab_size = logps.shape[-1]
    embeddings = embed_span_ids(row_spans, span_embedder, compose_device) if row_spans else None
    if profile_enabled:
        synchronize_span_devices(refs, compose_device, span_embedder)
    embedding_seconds = time.perf_counter() - embedding_started

    compose_started = time.perf_counter()
    if embeddings is not None:
        pseudo = span_token_pseudo_support_from_embeddings(
            row_ref_indices,
            row_first_tokens,
            row_supports,
            embeddings,
            len(refs),
            vocab_size,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            compose_device,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )
    else:
        pseudo = torch.zeros((len(refs), vocab_size), dtype=torch.float32, device=compose_device)

    span_lambda = resolved_span_token_lambda(args)
    smoothed_logps = apply_span_token_pseudo_support(logps, pseudo, span_lambda)
    if args.composition_type == "span_token_smoothing_delta":
        logp_target = compose_pi_min_delta_log_probs_from_logps(
            smoothed_logps,
            base_logp,
            args.temperature,
        )
    else:
        logp_target = compose_quorum_log_probs_from_logps(
            smoothed_logps,
            args.quorum_q,
            args.temperature,
        )
    if profile_enabled:
        synchronize_span_devices(refs, compose_device, span_embedder)
    compose_seconds = time.perf_counter() - compose_started

    trace = None
    if trace_enabled:
        pseudo_records = []
        for i, ref in enumerate(refs):
            values, token_ids = torch.topk(
                pseudo[i],
                k=min(args.trace_top_k, pseudo.shape[-1]),
                largest=True,
            )
            pseudo_records.append({
                "ref": ref["name"],
                "top_updates": [
                    {
                        "token_id": int(token_id),
                        "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
                        "pseudo_support": float(value),
                    }
                    for value, token_id in zip(values.tolist(), token_ids.tolist())
                    if value > 0
                ],
            })
        pseudo_edges = [] if embeddings is None else span_token_pseudo_support_edge_records(
            tokenizer,
            row_ref_indices,
            row_first_tokens,
            row_supports,
            row_spans,
            embeddings,
            [ref["name"] for ref in refs],
            span_embedder,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )[:args.trace_top_k]
        trace = {
            "algorithm": "span_token_smoothing",
            "implementation": "cached_batched",
            "support_normalization": normalization,
            "span_token_lambda": span_lambda,
            "cross_only": bool(args.span_token_cross_only),
            "parallel_refs": span_token_parallel_refs_enabled(args, refs),
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "n_proposal_rows": len(row_spans),
            "ref_proposals": {
                ref["name"]: [
                    repr(tokenizer.decode(list(span), skip_special_tokens=False))
                    for span in proposals
                ]
                for ref, proposals in zip(refs, per_ref_proposals)
            },
            "proposal_neighbors": [] if embeddings is None else span_neighbor_records(
                tokenizer,
                row_spans,
                embeddings,
                span_embedder,
                args.trace_top_k,
                args.span_kernel_tau,
                args.span_similarity_gate,
                args.span_similarity_threshold,
                args.span_similarity_soft_beta,
            ),
            "pseudo_support": pseudo_records,
            "pseudo_support_edges": pseudo_edges,
            "top_tokens": top_token_records(tokenizer, logp_target, args.trace_top_k),
        }

    if profile_enabled:
        total_seconds = time.perf_counter() - total_started
        record_span_token_profile(args, {
            "proposal": proposal_seconds,
            "embedding": embedding_seconds,
            "composition": compose_seconds,
            "total": total_seconds,
        })
    return logp_target, trace


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

    base = torch.tensor([[0.20, 0.20, 0.60]]).log()
    refs = torch.tensor([
        [[0.40, 0.10, 0.50]],
        [[0.30, 0.05, 0.65]],
    ]).log()
    delta = compose_pi_min_delta_log_probs_from_logps(refs, base).exp()
    # Token 0: both refs up from base, keep smaller upward shift: 0.30 pre-renorm.
    # Token 1: both refs down from base, keep less downward shift: 0.10 pre-renorm.
    # Token 2: disagreement around base, fall back to base: 0.60 pre-renorm.
    expected_delta = torch.tensor([[0.30, 0.10, 0.60]])
    expected_delta = expected_delta / expected_delta.sum(dim=-1, keepdim=True)
    assert torch.allclose(delta, expected_delta, atol=1e-6), (delta, expected_delta)
    qdelta = compose_pi_quorum_delta_log_probs_from_logps(refs, base, quorum_q=1).exp()
    expected_qdelta = torch.tensor([[0.40, 0.10, 0.65]])
    expected_qdelta = expected_qdelta / expected_qdelta.sum(dim=-1, keepdim=True)
    assert torch.allclose(qdelta, expected_qdelta, atol=1e-6), (qdelta, expected_qdelta)
    qdelta_m = compose_pi_quorum_delta_log_probs_from_logps(refs, base, quorum_q=2).exp()
    assert torch.allclose(qdelta_m, delta, atol=1e-6), (qdelta_m, delta)

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

    span_emb = torch.eye(3)
    span_exact = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.1],
    ])
    span_sponsors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    span_scores, span_smoothed, span_neighbors = compose_span_action_scores(
        span_exact,
        span_sponsors,
        span_emb,
        quorum_q=2,
        kernel_lambda=0.5,
        kernel_top_k=2,
        kernel_tau=1.0,
    )
    assert torch.isfinite(span_scores).all(), span_scores
    assert torch.isfinite(span_smoothed).all(), span_smoothed
    assert torch.isfinite(span_neighbors).all(), span_neighbors
    assert span_scores.shape == (3,), span_scores

    assert normalize_span_logp(-6.0, [1, 2, 3], "full") == -6.0
    assert normalize_span_logp(-6.0, [1, 2, 3], "length") == -2.0
    assert canonicalize_span_embedding_text("\n\nJoke: why") == "joke"
    assert canonicalize_span_embedding_text("**Humor:** ok") == "humor"
    assert canonicalize_span_embedding_text("Regular exercise helps.") == "regular exercise helps."

    neutral_exact = torch.tensor([
        [0.9, 0.1],
        [0.9, 0.1],
    ])
    neutral_sponsors = torch.tensor([
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    neutral_scores, _, neutral_neighbors = compose_span_action_scores(
        neutral_exact,
        neutral_sponsors,
        torch.eye(2),
        quorum_q=2,
        kernel_lambda=1.0,
        kernel_top_k=1,
        kernel_tau=0.1,
    )
    assert neutral_neighbors[:, 0].max().item() == 0.0, neutral_neighbors
    assert neutral_scores[0].item() == 0.0, neutral_scores
    assert neutral_scores[1].item() > 0.0, neutral_scores

    pseudo = span_token_pseudo_support_from_embeddings(
        ref_indices=[0, 1],
        first_tokens=[0, 1],
        supports=[0.5, 0.25],
        embeddings=torch.eye(2),
        m=2,
        vocab_size=3,
        top_k=1,
        tau=0.1,
        cross_only=True,
        device="cpu",
    )
    assert torch.allclose(pseudo[0], torch.tensor([0.0, 0.5, 0.0])), pseudo
    assert torch.allclose(pseudo[1], torch.tensor([0.25, 0.0, 0.0])), pseudo

    base_token_logps = torch.tensor([
        [[0.7, 0.2, 0.1]],
        [[0.6, 0.3, 0.1]],
    ]).log()
    smoothed_tokens = apply_span_token_pseudo_support(base_token_logps, pseudo, 0.5).exp().squeeze(1)
    expected_row0 = torch.tensor([0.7, 0.45, 0.1])
    expected_row0 = expected_row0 / expected_row0.sum()
    expected_row1 = torch.tensor([0.725, 0.3, 0.1])
    expected_row1 = expected_row1 / expected_row1.sum()
    assert torch.allclose(smoothed_tokens[0], expected_row0, atol=1e-6), smoothed_tokens
    assert torch.allclose(smoothed_tokens[1], expected_row1, atol=1e-6), smoothed_tokens
    gated_pseudo = span_token_pseudo_support_from_embeddings(
        ref_indices=[0, 1],
        first_tokens=[0, 1],
        supports=[0.5, 0.25],
        embeddings=torch.eye(2),
        m=2,
        vocab_size=3,
        top_k=1,
        tau=0.1,
        cross_only=True,
        device="cpu",
        similarity_gate="hard",
        similarity_threshold=0.5,
    )
    assert torch.allclose(gated_pseudo, torch.zeros_like(gated_pseudo)), gated_pseudo

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


def load_base_model(base_model_name, device):
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


def sample_one(prompt, global_sample_index, refs, tokenizer, args, cost_order, costs,
               kernel=None, span_embedder=None, neutral_model=None,
               neutral_device=None, base_model_for_delta=None,
               base_device=None):
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
    base_state = None
    if base_model_for_delta is not None:
        base_device = base_device or compose_device
        base_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=base_device)
        base_state = {
            "input_ids": base_input_ids,
            "attention_mask": torch.ones_like(base_input_ids, device=base_device),
            "past_key_values": None,
        }

    generated = []
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + global_sample_index)
    proposal_generators = []
    for ref_index, ref in enumerate(refs):
        proposal_generator = torch.Generator(device=ref["device"])
        proposal_generator.manual_seed(
            args.seed
            + 1_000_003 * (global_sample_index + 1)
            + 10_007 * (ref_index + 1)
        )
        proposal_generators.append(proposal_generator)
    lookback_history_logps = None
    traces = []
    trace_enabled = global_sample_index < int(args.trace_samples)

    with torch.inference_mode():
        for step_index in range(args.max_new_tokens):
            if len(generated) >= args.max_new_tokens:
                break
            if args.composition_type == "span_action_kernel":
                if span_embedder is None:
                    raise ValueError("span_embedder is required for span_action_kernel")
                max_span_len = min(args.span_h, args.max_new_tokens - len(generated))
                selected_span = sample_span_action_from_context(
                    prompt_ids + generated,
                    refs,
                    args,
                    stop_ids,
                    generator,
                    compose_device,
                    span_embedder,
                    max_span_len,
                )
                for token_id in selected_span:
                    if int(token_id) in stop_ids:
                        stop_reason = "eos"
                        break
                    generated.append(int(token_id))
                if stop_reason == "eos":
                    break
                continue
            if args.composition_type == "span_action_neutral":
                if span_embedder is None:
                    raise ValueError("span_embedder is required for span_action_neutral")
                max_span_len = min(args.span_h, args.max_new_tokens - len(generated))
                selected_span, trace = sample_span_action_neutral_from_context(
                    prompt_ids + generated,
                    refs,
                    neutral_model,
                    neutral_device,
                    tokenizer,
                    args,
                    stop_ids,
                    generator,
                    compose_device,
                    span_embedder,
                    max_span_len,
                    trace_enabled=trace_enabled,
                )
                if trace is not None:
                    trace["step_index"] = step_index
                    trace["generated_prefix_chars"] = len(tokenizer.decode(generated, skip_special_tokens=False))
                    traces.append(trace)
                for token_id in selected_span:
                    if int(token_id) in stop_ids:
                        stop_reason = "eos"
                        break
                    generated.append(int(token_id))
                if stop_reason == "eos":
                    break
                continue

            step_logps = []
            step_logps_by_ref = []
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
                device_logps = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                state["step_logp"] = device_logps
                step_logps_by_ref.append(device_logps)
                step_logps.append(device_logps.to(compose_device))

            logps = torch.stack(step_logps, dim=0)
            base_logp = None
            if base_state is not None:
                if base_state["past_key_values"] is None:
                    base_out = base_model_for_delta(
                        input_ids=base_state["input_ids"],
                        attention_mask=base_state["attention_mask"],
                        use_cache=True,
                    )
                else:
                    base_out = base_model_for_delta(
                        input_ids=base_state["input_ids"],
                        attention_mask=base_state["attention_mask"],
                        past_key_values=base_state["past_key_values"],
                        use_cache=True,
                    )
                base_state["past_key_values"] = base_out.past_key_values
                base_logp = torch.log_softmax(base_out.logits[:, -1, :].float(), dim=-1).to(compose_device)
            if args.composition_type == "lookback_min_gated":
                lookback_history_logps = update_lookback_log_history(
                    lookback_history_logps,
                    logps,
                    args.lookback_alpha,
                )
            if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"}:
                if span_embedder is None:
                    raise ValueError(f"span_embedder is required for {args.composition_type}")
                max_span_len = min(args.span_h, args.max_new_tokens - len(generated))
                if args.span_token_implementation == "cached_batched":
                    logp_target, trace = compose_span_token_smoothed_log_probs_cached_batched(
                        logps,
                        states,
                        refs,
                        tokenizer,
                        args,
                        stop_ids,
                        proposal_generators,
                        compose_device,
                        span_embedder,
                        max_span_len,
                        trace_enabled=trace_enabled,
                        base_logp=base_logp,
                    )
                elif args.span_token_implementation == "cached":
                    logp_target, trace = compose_span_token_smoothed_log_probs_cached(
                        logps,
                        states,
                        step_logps_by_ref,
                        refs,
                        tokenizer,
                        args,
                        stop_ids,
                        generator,
                        compose_device,
                        span_embedder,
                        max_span_len,
                        trace_enabled=trace_enabled,
                        base_logp=base_logp,
                    )
                else:
                    logp_target, trace = compose_span_token_smoothed_log_probs_from_context(
                        logps,
                        refs,
                        prompt_ids + generated,
                        tokenizer,
                        args,
                        stop_ids,
                        generator,
                        compose_device,
                        span_embedder,
                        max_span_len,
                        trace_enabled=trace_enabled,
                        base_logp=base_logp,
                    )
                if trace is not None:
                    trace["step_index"] = step_index
                    trace["generated_prefix_chars"] = len(tokenizer.decode(generated, skip_special_tokens=False))
                    traces.append(trace)
            else:
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
                    base_logp=base_logp,
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
            if base_state is not None:
                base_state["input_ids"] = next_token.to(base_device).view(1, 1)
                extra_attention = torch.ones(
                    (1, 1),
                    dtype=base_state["attention_mask"].dtype,
                    device=base_device,
                )
                base_state["attention_mask"] = torch.cat(
                    [base_state["attention_mask"], extra_attention],
                    dim=-1,
                )

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
    if traces:
        record["traces"] = traces
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


def span_embedding_label(args):
    if args.span_embedding_source != "sentence_transformer":
        return args.span_embedding_source
    model_tail = args.span_embedding_model.rstrip("/").split("/")[-1]
    model_tail = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_tail)
    text_mode = "canon" if args.span_embedding_text_mode == "canonical" else args.span_embedding_text_mode
    return f"{model_tail}_{text_mode}"


def composition_params(args, m):
    if args.composition_type == "min":
        return {"q": m, "label": f"Phi_{m}"}
    if args.composition_type == "quorum":
        return {"q": args.quorum_q, "label": f"Phi_{args.quorum_q}"}
    if args.composition_type == "pi_min_delta":
        return {"q": m, "label": "pi_min_delta"}
    if args.composition_type == "pi_quorum_delta":
        return {"q": args.quorum_q, "label": f"pi_quorum_delta_q{args.quorum_q}"}
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
    if args.composition_type == "span_action_kernel":
        return {
            "q": args.quorum_q,
            "h": args.span_h,
            "proposals_per_ref": args.span_proposals_per_ref,
            "embedding_source": args.span_embedding_source,
            "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
            "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
            "lambda": args.span_kernel_lambda,
            "tau": args.span_kernel_tau,
            "top_k": args.span_kernel_top_k,
            "label": (
                f"span_action_{span_embedding_label(args)}_"
                f"l{args.span_kernel_lambda:g}_r{args.span_proposals_per_ref}_h{args.span_h}"
            ),
        }
    if args.composition_type == "span_action_neutral":
        return {
            "q": args.quorum_q,
            "h": args.span_h,
            "proposals_per_ref": args.span_proposals_per_ref,
            "neutral_proposals": args.neutral_proposals,
            "embedding_source": args.span_embedding_source,
            "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
            "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
            "lambda": args.span_kernel_lambda,
            "tau": args.span_kernel_tau,
            "top_k": args.span_kernel_top_k,
            "support_normalization": resolved_span_support_normalization(args, "span_action_neutral"),
            "support_floor": args.span_support_floor,
            "label": (
                f"span_action_neutral_{span_embedding_label(args)}_"
                f"l{args.span_kernel_lambda:g}_r{args.span_proposals_per_ref}_"
                f"b{args.neutral_proposals}_h{args.span_h}_eps{args.span_support_floor:g}"
            ),
        }
    if args.composition_type == "span_token_smoothing":
        gate_suffix = "" if args.span_similarity_gate == "none" else f"_{span_similarity_gate_label(args)}"
        return {
            "q": args.quorum_q,
            "h": args.span_h,
            "proposals_per_ref": args.span_proposals_per_ref,
            "embedding_source": args.span_embedding_source,
            "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
            "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
            "lambda": resolved_span_token_lambda(args),
            "tau": args.span_kernel_tau,
            "top_k": args.span_kernel_top_k,
            "support_normalization": resolved_span_support_normalization(args, "span_token_smoothing"),
            "cross_only": bool(args.span_token_cross_only),
            "implementation": args.span_token_implementation,
            "parallel_refs": args.span_token_parallel_refs,
            "embedding_cache_size": args.span_embedding_cache_size,
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "label": (
                f"span_token_{span_embedding_label(args)}_"
                f"l{resolved_span_token_lambda(args):g}_r{args.span_proposals_per_ref}_h{args.span_h}"
                f"{gate_suffix}"
            ),
        }
    if args.composition_type == "span_token_smoothing_delta":
        gate_suffix = "" if args.span_similarity_gate == "none" else f"_{span_similarity_gate_label(args)}"
        return {
            "q": args.quorum_q,
            "h": args.span_h,
            "proposals_per_ref": args.span_proposals_per_ref,
            "embedding_source": args.span_embedding_source,
            "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
            "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
            "lambda": resolved_span_token_lambda(args),
            "tau": args.span_kernel_tau,
            "top_k": args.span_kernel_top_k,
            "support_normalization": resolved_span_support_normalization(args, "span_token_smoothing"),
            "cross_only": bool(args.span_token_cross_only),
            "implementation": args.span_token_implementation,
            "parallel_refs": args.span_token_parallel_refs,
            "embedding_cache_size": args.span_embedding_cache_size,
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "label": (
                f"span_token_delta_{span_embedding_label(args)}_"
                f"l{resolved_span_token_lambda(args):g}_r{args.span_proposals_per_ref}_h{args.span_h}"
                f"{gate_suffix}"
            ),
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
    parser.add_argument(
        "--composition_type",
        choices=[
            "min",
            "quorum",
            "soft_min",
            "lookback_min_gated",
            "kernel_smoothed",
            "span_action_kernel",
            "span_action_neutral",
            "span_token_smoothing",
            "pi_min_delta",
            "pi_quorum_delta",
            "span_token_smoothing_delta",
        ],
        default="quorum",
    )
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
    parser.add_argument("--span_h", type=int, default=6,
                        help="Maximum number of tokens in a span action.")
    parser.add_argument("--span_proposals_per_ref", type=int, default=4,
                        help="Number of sampled span-action proposals per reference at each branch.")
    parser.add_argument("--span_kernel_lambda", type=float, default=0.5)
    parser.add_argument("--span_kernel_tau", type=float, default=0.05)
    parser.add_argument("--span_kernel_top_k", type=int, default=8)
    parser.add_argument("--span_similarity_gate", choices=["none", "hard", "soft"], default="none")
    parser.add_argument("--span_similarity_threshold", type=float, default=0.0)
    parser.add_argument("--span_similarity_soft_beta", type=float, default=0.05)
    parser.add_argument("--span_embedding_source", choices=["input", "output", "sentence_transformer"], default="input")
    parser.add_argument("--span_embedding_model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--span_embedding_text_mode", choices=["raw", "canonical"], default="canonical")
    parser.add_argument("--span_embedding_device", default=None,
                        help="Device for sentence-transformer span embeddings; defaults to CPU.")
    parser.add_argument("--span_support_normalization", choices=["full", "length"], default=None,
                        help="Defaults to full for span_action_neutral and length for span_token_smoothing.")
    parser.add_argument("--neutral_proposals", type=int, default=4)
    parser.add_argument("--neutral_device", default=None,
                        help="Device for base-model neutral proposals; defaults to compose_device.")
    parser.add_argument("--base_device", default=None,
                        help="Device for the base model used by delta compositions; defaults to compose_device.")
    parser.add_argument("--span_support_floor", type=float, default=0.0)
    parser.add_argument("--span_token_lambda", type=float, default=None,
                        help="Pseudo-support weight for span_token_smoothing; defaults to span_kernel_lambda.")
    parser.add_argument("--span_token_cross_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Only transfer support between spans proposed by different refs.")
    parser.add_argument("--span_token_implementation", choices=["legacy", "cached", "cached_batched"], default="legacy")
    parser.add_argument("--span_token_parallel_refs", choices=["auto", "on", "off"], default="auto",
                        help="Run cached-batched span proposals for refs concurrently when devices permit.")
    parser.add_argument("--span_token_profile", action="store_true",
                        help="Record aggregate cached-batched span-token timing metadata.")
    parser.add_argument("--span_embedding_cache_size", type=int, default=50000,
                        help="Maximum canonical sentence-transformer span embeddings cached on CPU; 0 disables persistence.")
    parser.add_argument("--trace_samples", type=int, default=0)
    parser.add_argument("--trace_top_k", type=int, default=12)
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
    if args.composition_type in {"quorum", "pi_quorum_delta"} and (args.quorum_q < 1 or args.quorum_q > len(ref_pairs)):
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
    if args.composition_type in {"span_action_kernel", "span_action_neutral", "span_token_smoothing", "span_token_smoothing_delta"}:
        if args.quorum_q < 1 or args.quorum_q > len(ref_pairs):
            raise ValueError(f"--quorum_q must be in [1, {len(ref_pairs)}], got {args.quorum_q}")
        if args.span_h < 1:
            raise ValueError(f"--span_h must be positive, got {args.span_h}")
        if args.span_proposals_per_ref < 1:
            raise ValueError(
                f"--span_proposals_per_ref must be positive, got {args.span_proposals_per_ref}"
            )
        if not (0.0 <= args.span_kernel_lambda <= 1.0):
            raise ValueError(f"--span_kernel_lambda must be in [0, 1], got {args.span_kernel_lambda}")
        if args.span_kernel_tau <= 0.0:
            raise ValueError(f"--span_kernel_tau must be positive, got {args.span_kernel_tau}")
        if args.span_kernel_top_k < 1:
            raise ValueError(f"--span_kernel_top_k must be positive, got {args.span_kernel_top_k}")
        if args.span_similarity_gate == "soft" and args.span_similarity_soft_beta <= 0.0:
            raise ValueError(
                "--span_similarity_soft_beta must be positive for soft gating, "
                f"got {args.span_similarity_soft_beta}"
            )
        if args.trace_samples < 0:
            raise ValueError(f"--trace_samples must be nonnegative, got {args.trace_samples}")
        if args.trace_top_k < 1:
            raise ValueError(f"--trace_top_k must be positive, got {args.trace_top_k}")
    if args.composition_type == "span_action_neutral":
        if args.neutral_proposals < 1:
            raise ValueError(f"--neutral_proposals must be positive, got {args.neutral_proposals}")
        if args.span_support_floor < 0.0:
            raise ValueError(f"--span_support_floor must be nonnegative, got {args.span_support_floor}")
    if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"}:
        if resolved_span_token_lambda(args) < 0.0:
            raise ValueError(f"--span_token_lambda must be nonnegative, got {resolved_span_token_lambda(args)}")

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
    span_embedder = None
    neutral_model = None
    neutral_device = args.neutral_device or compose_device
    base_model_for_delta = None
    base_device = args.base_device or compose_device
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
    if args.composition_type in {"span_action_kernel", "span_action_neutral", "span_token_smoothing", "span_token_smoothing_delta"}:
        print(
            "Preparing span embeddings: "
            f"source={args.span_embedding_source} h={args.span_h} R={args.span_proposals_per_ref} "
            f"lambda={args.span_kernel_lambda} tau={args.span_kernel_tau} top_k={args.span_kernel_top_k}"
        )
        span_embedder = build_span_embedder(refs[0]["model"], tokenizer, args)
        if args.span_embedding_source == "sentence_transformer":
            print(
                "Sentence-transformer span embedder: "
                f"model={args.span_embedding_model} text_mode={args.span_embedding_text_mode} "
                f"device={span_embedder.get('device')}"
            )
    if args.composition_type == "span_action_neutral":
        print(
            "Loading base model for neutral proposals: "
            f"device={neutral_device} R0={args.neutral_proposals} floor={args.span_support_floor}"
        )
        neutral_model = load_base_model(base_model, neutral_device)
    if args.composition_type in {"pi_min_delta", "pi_quorum_delta", "span_token_smoothing_delta"}:
        print(f"Loading base model for delta composition on {base_device}: {base_model}")
        base_model_for_delta = load_base_model(base_model, base_device)

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
                span_embedder=span_embedder,
                neutral_model=neutral_model,
                neutral_device=neutral_device,
                base_model_for_delta=base_model_for_delta,
                base_device=base_device,
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
            "base_device": base_device if base_model_for_delta is not None else None,
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
            "span_action": None if args.composition_type != "span_action_kernel" else {
                "q": args.quorum_q,
                "h": args.span_h,
                "proposals_per_ref": args.span_proposals_per_ref,
                "embedding_source": args.span_embedding_source,
                "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
                "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
                "embedding_runtime": span_embedder_meta(span_embedder),
                "lambda": args.span_kernel_lambda,
                "tau": args.span_kernel_tau,
                "top_k": args.span_kernel_top_k,
            },
            "span_experiment": None if args.composition_type not in {"span_action_neutral", "span_token_smoothing", "span_token_smoothing_delta"} else {
                "q": args.quorum_q,
                "h": args.span_h,
                "proposals_per_ref": args.span_proposals_per_ref,
                "neutral_proposals": args.neutral_proposals if args.composition_type == "span_action_neutral" else None,
                "neutral_device": neutral_device if args.composition_type == "span_action_neutral" else None,
                "embedding_source": args.span_embedding_source,
                "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
                "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
                "embedding_runtime": span_embedder_meta(span_embedder),
                "lambda": resolved_span_token_lambda(args) if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"} else args.span_kernel_lambda,
                "tau": args.span_kernel_tau,
                "top_k": args.span_kernel_top_k,
                "similarity_gate": args.span_similarity_gate,
                "similarity_threshold": args.span_similarity_threshold,
                "similarity_soft_beta": args.span_similarity_soft_beta,
                "support_normalization": resolved_span_support_normalization(args),
                "support_floor": args.span_support_floor if args.composition_type == "span_action_neutral" else None,
                "cross_only": bool(args.span_token_cross_only) if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"} else None,
                "implementation": args.span_token_implementation if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"} else None,
                "parallel_refs": args.span_token_parallel_refs if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"} else None,
                "parallel_refs_resolved": span_token_parallel_refs_enabled(args, refs) if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"} else None,
                "embedding_cache_size": args.span_embedding_cache_size,
                "profile": span_token_profile_meta(args) if args.composition_type in {"span_token_smoothing", "span_token_smoothing_delta"} else None,
                "trace_samples": args.trace_samples,
                "trace_top_k": args.trace_top_k,
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
