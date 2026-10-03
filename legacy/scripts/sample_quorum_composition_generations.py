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
  - lookback_min_reordered_mixture: mix current and masked historical support
    per reference, then take min once
  - lookback_min_capped_mixture: mix exact min with historical consensus capped
    by current max probability across refs
  - lookback_min_rollout_capped_mixture: capped mixture whose historical
    consensus is estimated from each reference's own prompt-level rollouts
  - lookback_min_rollout_plus_current_capped_mixture: same rollout history,
    unioned with current-trajectory history before capping
  - kernel_smoothed: quorum/min over semantically smoothed reference
    distributions, optionally gated by current max probability across refs
  - span_action_kernel: quorum/min over short continuation spans proposed by
    the references, with semantic smoothing over decoded span embeddings
  - span_action_neutral: span_action_kernel with additional base-model neutral
    proposals and an optional absolute support floor
  - span_token_smoothing: short spans are used only to add sparse support to
    next-token distributions before token-level quorum

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
CONTENT_COST_PREFIXES = ("Eagle:", "Topaz:")
LOOKBACK_CURRENT_HISTORY_TYPES = {
    "lookback_min_gated",
    "lookback_min_eligible",
    "lookback_min_candidate_mixture",
    "lookback_min_reordered_mixture",
    "lookback_min_capped_mixture",
}
LOOKBACK_ROLLOUT_HISTORY_TYPES = {
    "lookback_min_rollout_capped_mixture",
    "lookback_min_rollout_plus_current_capped_mixture",
}
LOOKBACK_HISTORY_TYPES = LOOKBACK_CURRENT_HISTORY_TYPES | LOOKBACK_ROLLOUT_HISTORY_TYPES


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


def content_nonempty_lines(text, cost_prefixes=CONTENT_COST_PREFIXES):
    lines = nonempty_lines(text)
    if lines and any(
        re.compile(rf"^{re.escape(prefix)}\s+\S", re.IGNORECASE).match(lines[0])
        for prefix in cost_prefixes
    ):
        return lines[1:]
    return lines


def joke_content_position_metrics(text):
    lines = content_nonempty_lines(text)
    strict_indices = joke_line_indices(lines, flex=False)
    flex_indices = joke_line_indices(lines, flex=True)
    n_lines = len(lines)
    return {
        "content_nonempty_lines": lines,
        "joke_content_line_indices_flex": flex_indices,
        "joke_content_line_indices_strict": strict_indices,
        "joke_content_position_bucket_flex": joke_position_bucket(flex_indices, n_lines),
        "joke_content_position_bucket_strict": joke_position_bucket(strict_indices, n_lines),
        "has_joke_content_early_flex": bool(flex_indices and 0 in flex_indices),
        "has_joke_content_early_strict": bool(strict_indices and 0 in strict_indices),
        "has_joke_content_final_flex": bool(
            flex_indices and n_lines and (n_lines - 1) in flex_indices
        ),
        "has_joke_content_final_strict": bool(
            strict_indices and n_lines and (n_lines - 1) in strict_indices
        ),
        "has_joke_content_anywhere_flex": bool(flex_indices),
        "has_joke_content_anywhere_strict": bool(strict_indices),
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


def has_line_prefix_anywhere(text, prefix):
    pattern = re.compile(
        rf"^[\s\*_>]*{re.escape(prefix)}\s+\S",
        re.IGNORECASE | re.MULTILINE,
    )
    return bool(pattern.search(text))


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


def compose_lookback_min_gated_log_probs_from_logps(
    logps,
    history_logps,
    temperature=1.0,
    structural_exemption_token_ids=None,
):
    """Return log-probs for current-gated lookback min.

    The unnormalized score is:
      S_t(v) = min_i H_i^t(v) * max_i pi_i(v | c_t)
    where H_i^t is the historical max probability for reference i.
    """
    import torch

    if history_logps is None:
        raise ValueError("history_logps is required for lookback_min_gated")
    historical_consensus = torch.min(history_logps, dim=0).values
    if structural_exemption_token_ids is not None:
        token_ids = structural_exemption_token_ids.to(
            device=historical_consensus.device,
            dtype=torch.long,
        )
        if token_ids.numel():
            historical_any = torch.max(history_logps, dim=0).values
            historical_consensus = historical_consensus.clone()
            historical_consensus.index_copy_(
                -1,
                token_ids,
                historical_any.index_select(-1, token_ids),
            )
    current_gate = torch.max(logps, dim=0).values
    return normalize_log_target(historical_consensus + current_gate, temperature)


def compose_lookback_min_eligible_log_probs_from_logps(
    logps,
    history_logps,
    eligibility_threshold=1e-6,
    temperature=1.0,
    structural_exemption_token_ids=None,
):
    """Return log-probs for eligibility-only lookback min.

    The unnormalized score is:
      S_t(v) = min_i H_i^t(v) if max_i pi_i(v | c_t) >= epsilon, else 0.
    The current distribution gates candidates but does not multiply their score.
    """
    import torch

    if history_logps is None:
        raise ValueError("history_logps is required for lookback_min_eligible")
    if eligibility_threshold <= 0.0:
        raise ValueError(
            f"lookback_eligibility_threshold must be positive, got {eligibility_threshold}"
        )
    historical_consensus = torch.min(history_logps, dim=0).values
    if structural_exemption_token_ids is not None:
        token_ids = structural_exemption_token_ids.to(
            device=historical_consensus.device,
            dtype=torch.long,
        )
        if token_ids.numel():
            historical_any = torch.max(history_logps, dim=0).values
            historical_consensus = historical_consensus.clone()
            historical_consensus.index_copy_(
                -1,
                token_ids,
                historical_any.index_select(-1, token_ids),
            )
    current_max = torch.max(logps, dim=0).values
    eligible = current_max >= math.log(eligibility_threshold)
    if not bool(eligible.any(dim=-1).all().item()):
        raise ValueError(
            "lookback_min_eligible masked all tokens; lower "
            f"--lookback_eligibility_threshold below {eligibility_threshold}"
        )
    masked = torch.where(
        eligible,
        historical_consensus,
        torch.full_like(historical_consensus, float("-inf")),
    )
    return normalize_log_target(masked, temperature)


def lookback_candidate_mask(logps, candidate_top_k_ref=32, candidate_top_k_min=32):
    """Return candidate mask from per-ref top-k and exact-min top-k tokens."""
    import torch

    if candidate_top_k_ref < 1:
        raise ValueError(
            f"lookback_candidate_top_k_ref must be positive, got {candidate_top_k_ref}"
        )
    if candidate_top_k_min < 0:
        raise ValueError(
            "lookback_candidate_top_k_min must be non-negative, got "
            f"{candidate_top_k_min}"
        )
    exact_raw = torch.min(logps, dim=0).values
    vocab = exact_raw.shape[-1]
    mask = torch.zeros_like(exact_raw, dtype=torch.bool)
    ref_k = min(int(candidate_top_k_ref), int(vocab))
    min_k = min(int(candidate_top_k_min), int(vocab))
    ref_indices = torch.topk(logps, k=ref_k, dim=-1, largest=True).indices
    for indices in ref_indices:
        mask.scatter_(-1, indices, True)
    if min_k:
        min_indices = torch.topk(exact_raw, k=min_k, dim=-1, largest=True).indices
        mask.scatter_(-1, min_indices, True)
    return mask


def compose_lookback_min_candidate_mixture_log_probs_from_logps(
    logps,
    history_logps,
    mixture_beta=0.25,
    candidate_top_k_ref=32,
    candidate_top_k_min=32,
    temperature=1.0,
    structural_exemption_token_ids=None,
):
    """Return log-probs for candidate-limited mixture lookback.

    The distribution is:
      (1-beta) * Normalize(min_i p_i^t)
      + beta * Normalize(1[v in C_t] min_i H_i^t)
    where C_t is the union of per-reference current top-k tokens and current
    exact-min top-k tokens.
    """
    import torch

    if history_logps is None:
        raise ValueError("history_logps is required for lookback_min_candidate_mixture")
    if not (0.0 <= mixture_beta <= 1.0):
        raise ValueError(f"lookback_mixture_beta must be in [0, 1], got {mixture_beta}")
    exact_raw = torch.min(logps, dim=0).values
    exact_logps = normalize_log_target(exact_raw, temperature)
    if mixture_beta == 0.0:
        return exact_logps

    historical_consensus = torch.min(history_logps, dim=0).values
    if structural_exemption_token_ids is not None:
        token_ids = structural_exemption_token_ids.to(
            device=historical_consensus.device,
            dtype=torch.long,
        )
        if token_ids.numel():
            historical_any = torch.max(history_logps, dim=0).values
            historical_consensus = historical_consensus.clone()
            historical_consensus.index_copy_(
                -1,
                token_ids,
                historical_any.index_select(-1, token_ids),
            )
    candidates = lookback_candidate_mask(
        logps,
        candidate_top_k_ref=candidate_top_k_ref,
        candidate_top_k_min=candidate_top_k_min,
    )
    candidate_raw = torch.where(
        candidates,
        historical_consensus,
        torch.full_like(historical_consensus, float("-inf")),
    )
    if not bool(torch.isfinite(candidate_raw).any(dim=-1).all().item()):
        raise ValueError("lookback_min_candidate_mixture has no finite candidate-history mass")
    candidate_logps = normalize_log_target(candidate_raw, temperature)
    if mixture_beta == 1.0:
        return candidate_logps

    exact_weight = math.log1p(-mixture_beta)
    history_weight = math.log(mixture_beta)
    return torch.logaddexp(exact_logps + exact_weight, candidate_logps + history_weight)


def compose_lookback_min_reordered_mixture_log_probs_from_logps(
    logps,
    history_logps,
    mixture_beta=0.25,
    candidate_top_k_ref=8,
    candidate_top_k_min=0,
    temperature=1.0,
):
    """Mix current and masked history per reference before taking min."""
    import torch

    if history_logps is None:
        raise ValueError("history_logps is required for lookback_min_reordered_mixture")
    if not (0.0 <= mixture_beta <= 1.0):
        raise ValueError(f"lookback_mixture_beta must be in [0, 1], got {mixture_beta}")
    exact_raw = torch.min(logps, dim=0).values
    if mixture_beta == 0.0:
        return normalize_log_target(exact_raw, temperature)

    candidates = lookback_candidate_mask(
        logps,
        candidate_top_k_ref=candidate_top_k_ref,
        candidate_top_k_min=candidate_top_k_min,
    )
    masked_history = torch.where(
        candidates.unsqueeze(0),
        history_logps,
        torch.full_like(history_logps, float("-inf")),
    )
    if not bool(torch.isfinite(masked_history).any(dim=-1).all().item()):
        raise ValueError("lookback_min_reordered_mixture has no finite history mass")
    normalized_history = masked_history - torch.logsumexp(
        masked_history, dim=-1, keepdim=True
    )
    if mixture_beta == 1.0:
        smoothed_refs = normalized_history
    else:
        smoothed_refs = torch.logaddexp(
            logps + math.log1p(-mixture_beta),
            normalized_history + math.log(mixture_beta),
        )
    consensus = torch.min(smoothed_refs, dim=0).values
    return normalize_log_target(consensus, temperature)


def compose_lookback_min_capped_mixture_log_probs_from_logps(
    logps,
    history_logps,
    mixture_beta=0.25,
    temperature=1.0,
):
    """Mix exact min with historical consensus capped by current max."""
    import torch

    if history_logps is None:
        raise ValueError("history_logps is required for lookback_min_capped_mixture")
    if not (0.0 <= mixture_beta <= 1.0):
        raise ValueError(f"lookback_mixture_beta must be in [0, 1], got {mixture_beta}")
    exact_raw = torch.min(logps, dim=0).values
    exact_logps = normalize_log_target(exact_raw, temperature)
    if mixture_beta == 0.0:
        return exact_logps

    historical_consensus = torch.min(history_logps, dim=0).values
    current_max = torch.max(logps, dim=0).values
    capped_raw = torch.minimum(historical_consensus, current_max)
    capped_logps = normalize_log_target(capped_raw, temperature)
    if mixture_beta == 1.0:
        return capped_logps
    return torch.logaddexp(
        exact_logps + math.log1p(-mixture_beta),
        capped_logps + math.log(mixture_beta),
    )


def compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
    logps,
    rollout_history_logps,
    current_history_logps=None,
    include_current_history=False,
    mixture_beta=0.25,
    temperature=1.0,
):
    """Mix exact min with rollout-historical consensus capped by current max."""
    import torch

    if rollout_history_logps is None:
        raise ValueError(
            "rollout_history_logps is required for rollout capped lookback"
        )
    if include_current_history:
        if current_history_logps is None:
            raise ValueError(
                "current_history_logps is required when include_current_history is true"
            )
        history_logps = torch.maximum(rollout_history_logps, current_history_logps)
    else:
        history_logps = rollout_history_logps
    return compose_lookback_min_capped_mixture_log_probs_from_logps(
        logps,
        history_logps,
        mixture_beta=mixture_beta,
        temperature=temperature,
    )


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


def compose_log_probs(composition_type, logps, quorum_q, soft_min_p, temperature,
                      history_logps=None, rollout_history_logps=None,
                      current_history_logps=None, kernel=None, kernel_lambda=0.0,
                      kernel_gate="none", kernel_source_top_k=256,
                      structural_exemption_token_ids=None,
                      lookback_eligibility_threshold=1e-6,
                      lookback_mixture_beta=0.25,
                      lookback_candidate_top_k_ref=32,
                      lookback_candidate_top_k_min=32):
    m = logps.shape[0]
    if composition_type == "min":
        return compose_quorum_log_probs_from_logps(logps, m, temperature)
    if composition_type == "quorum":
        return compose_quorum_log_probs_from_logps(logps, quorum_q, temperature)
    if composition_type == "soft_min":
        return compose_soft_min_log_probs_from_logps(logps, soft_min_p, temperature)
    if composition_type == "lookback_min_gated":
        return compose_lookback_min_gated_log_probs_from_logps(
            logps,
            history_logps,
            temperature,
            structural_exemption_token_ids=structural_exemption_token_ids,
        )
    if composition_type == "lookback_min_eligible":
        return compose_lookback_min_eligible_log_probs_from_logps(
            logps,
            history_logps,
            lookback_eligibility_threshold,
            temperature,
            structural_exemption_token_ids=structural_exemption_token_ids,
        )
    if composition_type == "lookback_min_candidate_mixture":
        return compose_lookback_min_candidate_mixture_log_probs_from_logps(
            logps,
            history_logps,
            lookback_mixture_beta,
            lookback_candidate_top_k_ref,
            lookback_candidate_top_k_min,
            temperature,
            structural_exemption_token_ids=structural_exemption_token_ids,
        )
    if composition_type == "lookback_min_reordered_mixture":
        return compose_lookback_min_reordered_mixture_log_probs_from_logps(
            logps,
            history_logps,
            lookback_mixture_beta,
            lookback_candidate_top_k_ref,
            lookback_candidate_top_k_min,
            temperature,
        )
    if composition_type == "lookback_min_capped_mixture":
        return compose_lookback_min_capped_mixture_log_probs_from_logps(
            logps,
            history_logps,
            lookback_mixture_beta,
            temperature,
        )
    if composition_type == "lookback_min_rollout_capped_mixture":
        return compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
            logps,
            rollout_history_logps,
            current_history_logps=current_history_logps,
            include_current_history=False,
            mixture_beta=lookback_mixture_beta,
            temperature=temperature,
        )
    if composition_type == "lookback_min_rollout_plus_current_capped_mixture":
        return compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
            logps,
            rollout_history_logps,
            current_history_logps=current_history_logps,
            include_current_history=True,
            mixture_beta=lookback_mixture_beta,
            temperature=temperature,
        )
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


def sample_token_from_logp(logp, temperature, generator, sample_device):
    import torch

    logp = logp.to(sample_device)
    if logp.dim() == 1:
        logp = logp.unsqueeze(0)
    logp = normalize_log_target(logp, temperature)
    if temperature <= 0:
        return int(torch.argmax(logp, dim=-1).item())
    return int(torch.multinomial(logp.exp(), num_samples=1, generator=generator).item())


def rollout_seed(base_seed, seed_offset, prompt_index, ref_index, rollout_index):
    return (
        int(base_seed)
        + int(seed_offset)
        + 1000 * int(prompt_index)
        + 100 * int(ref_index)
        + int(rollout_index)
    )


def build_prompt_rollout_history_logps(
    prompt_ids,
    prompt_index,
    refs,
    args,
    compose_device,
    stop_ids,
):
    """Estimate per-ref historical log-probs from independent prompt rollouts."""
    import torch

    if args.lookback_rollout_samples < 1:
        raise ValueError(
            "--lookback_rollout_samples must be positive for rollout lookback"
        )
    if args.lookback_rollout_steps < 1:
        raise ValueError(
            "--lookback_rollout_steps must be positive for rollout lookback"
        )
    histories = []
    summaries = []
    with torch.inference_mode():
        for ref_index, ref in enumerate(refs):
            device = ref["device"]
            model = ref["model"]
            ref_history = None
            rollout_lengths = []
            eos_hits = 0
            for rollout_index in range(args.lookback_rollout_samples):
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    rollout_seed(
                        args.seed,
                        args.lookback_rollout_seed_offset,
                        prompt_index,
                        ref_index,
                        rollout_index,
                    )
                )
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, device=device)
                past_key_values = None
                length = 0
                for _ in range(args.lookback_rollout_steps):
                    if past_key_values is None:
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                        )
                    else:
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    past_key_values = out.past_key_values
                    device_logps = torch.log_softmax(
                        out.logits[:, -1, :].float(), dim=-1
                    )
                    history_row = device_logps.to(compose_device)
                    ref_history = (
                        history_row
                        if ref_history is None
                        else torch.maximum(ref_history, history_row)
                    )
                    token_id = sample_token_from_logp(
                        device_logps,
                        args.lookback_rollout_temperature,
                        generator,
                        device,
                    )
                    length += 1
                    if int(token_id) in stop_ids:
                        eos_hits += 1
                        break
                    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    extra_attention = torch.ones(
                        (1, 1), dtype=attention_mask.dtype, device=device
                    )
                    attention_mask = torch.cat([attention_mask, extra_attention], dim=-1)
                rollout_lengths.append(length)
            if ref_history is None:
                raise ValueError(f"no rollout history recorded for ref {ref['name']}")
            histories.append(ref_history)
            summaries.append({
                "ref": ref["name"],
                "samples": int(args.lookback_rollout_samples),
                "steps": int(args.lookback_rollout_steps),
                "temperature": float(args.lookback_rollout_temperature),
                "seed_offset": int(args.lookback_rollout_seed_offset),
                "rollout_lengths": rollout_lengths,
                "eos_hits": eos_hits,
            })
    history_logps = torch.stack(histories, dim=0)
    return history_logps, summaries


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


def sample_token_rows_from_logp(logp, temperature, generator):
    import torch

    if logp.dim() == 1:
        logp = logp.unsqueeze(0)
    normalized = normalize_log_target(logp, temperature)
    if temperature <= 0:
        return torch.argmax(normalized, dim=-1)
    return torch.multinomial(normalized.exp(), num_samples=1, generator=generator).squeeze(-1)


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
        past_key_values.batch_select_indices(indices)
        return past_key_values

    def select_item(item):
        if torch.is_tensor(item):
            return item.index_select(0, indices.to(item.device))
        if isinstance(item, tuple):
            return tuple(select_item(value) for value in item)
        if isinstance(item, list):
            return [select_item(value) for value in item]
        return item

    return select_item(past_key_values)


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


def sample_ref_span_proposals_cached_batched(model, base_past, base_attention_mask,
                                             first_logp, max_span_len, n_proposals,
                                             stop_ids, temperature, generator):
    import torch

    if n_proposals < 1:
        return [], []
    if max_span_len < 1:
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
            started = time.perf_counter()
            vectors = span_embedder["model"].encode(
                missing,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            span_embedder["encode_seconds"] += time.perf_counter() - started
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

        return torch.stack(
            [resolved_vectors[text] for text in texts],
            dim=0,
        ).to(device)
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
    if composition_type == "span_token_smoothing":
        return "length"
    return "full"


def resolved_span_token_lambda(args):
    return args.span_kernel_lambda if args.span_token_lambda is None else args.span_token_lambda


def span_similarity_gate_label(args):
    if args.span_similarity_gate == "none":
        return "none"
    if args.span_similarity_gate == "hard":
        return f"hard_s{args.span_similarity_threshold:g}"
    if args.span_similarity_gate == "soft":
        return f"soft_s{args.span_similarity_threshold:g}_b{args.span_similarity_soft_beta:g}"
    raise ValueError(f"unknown span_similarity_gate: {args.span_similarity_gate}")


def compose_span_action_scores(exact_supports, sponsor_mask, embeddings, quorum_q,
                               kernel_lambda, kernel_top_k, kernel_tau,
                               similarity_gate="none", similarity_threshold=0.0,
                               similarity_soft_beta=0.05):
    import torch

    if not (0.0 <= kernel_lambda <= 1.0):
        raise ValueError(f"span_kernel_lambda must be in [0, 1], got {kernel_lambda}")
    m = exact_supports.shape[0]
    if quorum_q < 1 or quorum_q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {quorum_q}")

    kernel, _ = span_kernel_from_embeddings(
        embeddings,
        kernel_top_k,
        kernel_tau,
        similarity_gate,
        similarity_threshold,
        similarity_soft_beta,
    )
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
        args.span_similarity_gate,
        args.span_similarity_threshold,
        args.span_similarity_soft_beta,
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
        args.span_similarity_gate,
        args.span_similarity_threshold,
        args.span_similarity_soft_beta,
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
        args.span_similarity_gate,
        args.span_similarity_threshold,
        args.span_similarity_soft_beta,
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
                    args.span_similarity_gate,
                    args.span_similarity_threshold,
                    args.span_similarity_soft_beta,
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
                args.span_similarity_gate,
                args.span_similarity_threshold,
                args.span_similarity_soft_beta,
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
                                                       max_span_len, trace_enabled=False):
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
            "implementation": "legacy",
            "support_normalization": normalization,
            "span_token_lambda": span_lambda,
            "cross_only": bool(args.span_token_cross_only),
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
    if args.span_token_parallel_refs == "off":
        return False
    devices = [str(ref["device"]) for ref in refs]
    distinct_cuda_devices = (
        len(set(devices)) == len(devices)
        and all(device.startswith("cuda") for device in devices)
    )
    return distinct_cuda_devices


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


def compose_span_token_smoothed_log_probs_cached(logps, states, refs, tokenizer,
                                                 args, stop_ids, proposal_generators,
                                                 compose_device, span_embedder,
                                                 max_span_len, trace_enabled=False):
    import torch

    if args.span_token_profile:
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

    if args.span_token_profile:
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
    if row_spans:
        embeddings = embed_span_ids(row_spans, span_embedder, compose_device)
    else:
        embeddings = None
    if args.span_token_profile:
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
        pseudo = torch.zeros(
            (len(refs), vocab_size),
            dtype=torch.float32,
            device=compose_device,
        )

    span_lambda = resolved_span_token_lambda(args)
    smoothed_logps = apply_span_token_pseudo_support(logps, pseudo, span_lambda)
    logp_target = compose_quorum_log_probs_from_logps(
        smoothed_logps,
        args.quorum_q,
        args.temperature,
    )
    if args.span_token_profile:
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

    if args.span_token_profile:
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

    newline_ids = torch.tensor([2], dtype=torch.long)
    exempt = compose_lookback_min_gated_log_probs_from_logps(
        step2,
        history,
        structural_exemption_token_ids=newline_ids,
    ).exp()
    manual_exempt_scores = torch.tensor([[0.36, 0.09, 0.25]])
    expected_exempt = manual_exempt_scores / manual_exempt_scores.sum(dim=-1, keepdim=True)
    assert torch.allclose(exempt, expected_exempt, atol=1e-6), (exempt, expected_exempt)
    assert torch.allclose(
        compose_lookback_min_gated_log_probs_from_logps(step2, history),
        compose_lookback_min_gated_log_probs_from_logps(
            step2,
            history,
            structural_exemption_token_ids=None,
        ),
    )

    decayed_history = update_lookback_log_history(None, step1, alpha=0.5)
    decayed_history = update_lookback_log_history(decayed_history, step2, alpha=0.5)
    decayed = compose_lookback_min_gated_log_probs_from_logps(step2, decayed_history).exp()
    manual_decayed_scores = torch.tensor([[0.21, 0.09, 0.05]])
    expected_decayed = manual_decayed_scores / manual_decayed_scores.sum(dim=-1, keepdim=True)
    assert torch.allclose(decayed, expected_decayed, atol=1e-6), (decayed, expected_decayed)

    eligible = compose_lookback_min_eligible_log_probs_from_logps(
        step2,
        history,
        eligibility_threshold=0.11,
    ).exp()
    manual_eligible_scores = torch.tensor([[0.60, 0.30, 0.10]])
    expected_eligible = manual_eligible_scores / manual_eligible_scores.sum(dim=-1, keepdim=True)
    assert torch.allclose(eligible, expected_eligible, atol=1e-6), (
        eligible,
        expected_eligible,
    )

    prompt_start = torch.tensor([
        [1e-8, 0.40, 0.60],
        [0.95, 0.50, 1e-8],
    ]).log().unsqueeze(1)
    eligible_start = compose_lookback_min_eligible_log_probs_from_logps(
        prompt_start,
        prompt_start,
        eligibility_threshold=1e-6,
    ).exp()
    exact_start = compose_quorum_log_probs_from_logps(prompt_start, 2).exp()
    assert torch.allclose(eligible_start, exact_start, atol=1e-6), (
        eligible_start,
        exact_start,
    )
    assert torch.argmax(eligible_start, dim=-1).item() == 1, eligible_start

    strict_eligible = compose_lookback_min_eligible_log_probs_from_logps(
        step2,
        history,
        eligibility_threshold=0.31,
    ).exp()
    manual_strict_scores = torch.tensor([[0.60, 0.0, 0.10]])
    expected_strict = manual_strict_scores / manual_strict_scores.sum(dim=-1, keepdim=True)
    assert torch.allclose(strict_eligible, expected_strict, atol=1e-6), strict_eligible

    eligible_exempt = compose_lookback_min_eligible_log_probs_from_logps(
        step2,
        history,
        eligibility_threshold=0.11,
        structural_exemption_token_ids=newline_ids,
    ).exp()
    manual_eligible_exempt = torch.tensor([[0.60, 0.30, 0.50]])
    expected_eligible_exempt = manual_eligible_exempt / manual_eligible_exempt.sum(dim=-1, keepdim=True)
    assert torch.allclose(eligible_exempt, expected_eligible_exempt, atol=1e-6), eligible_exempt

    try:
        compose_lookback_min_eligible_log_probs_from_logps(
            step2,
            history,
            eligibility_threshold=0.99,
        )
    except ValueError as exc:
        assert "masked all tokens" in str(exc)
    else:
        raise AssertionError("expected all-masked eligibility lookback to fail")

    candidate_mixture_exact = compose_lookback_min_candidate_mixture_log_probs_from_logps(
        step2,
        history,
        mixture_beta=0.0,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    ).exp()
    exact_step2 = compose_quorum_log_probs_from_logps(step2, 2).exp()
    assert torch.allclose(candidate_mixture_exact, exact_step2, atol=1e-6), (
        candidate_mixture_exact,
        exact_step2,
    )

    candidate_mixture_history = compose_lookback_min_candidate_mixture_log_probs_from_logps(
        step2,
        history,
        mixture_beta=1.0,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    ).exp()
    expected_candidate_history = torch.tensor([[0.60, 0.30, 0.10]])
    expected_candidate_history /= expected_candidate_history.sum(dim=-1, keepdim=True)
    assert torch.allclose(candidate_mixture_history, expected_candidate_history, atol=1e-6), (
        candidate_mixture_history,
        expected_candidate_history,
    )

    current4 = torch.tensor([
        [0.01, 0.78, 0.12, 0.09],
        [0.70, 0.08, 0.13, 0.09],
    ]).log().unsqueeze(1)
    history4 = torch.tensor([
        [0.90, 0.40, 0.30, 0.95],
        [0.50, 0.80, 0.30, 0.95],
    ]).log().unsqueeze(1)
    mask4 = lookback_candidate_mask(
        current4,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    )
    assert mask4.squeeze(0).tolist() == [True, True, True, False], mask4
    ref_only_mask4 = lookback_candidate_mask(
        current4,
        candidate_top_k_ref=1,
        candidate_top_k_min=0,
    )
    assert ref_only_mask4.squeeze(0).tolist() == [True, True, False, False], (
        ref_only_mask4
    )
    ref_only_candidate = compose_lookback_min_candidate_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=1.0,
        candidate_top_k_ref=1,
        candidate_top_k_min=0,
    ).exp()
    expected_ref_only_candidate = torch.tensor([[0.50, 0.40, 0.0, 0.0]])
    expected_ref_only_candidate /= expected_ref_only_candidate.sum(
        dim=-1, keepdim=True
    )
    assert torch.allclose(
        ref_only_candidate,
        expected_ref_only_candidate,
        atol=1e-6,
    ), (ref_only_candidate, expected_ref_only_candidate)
    candidate_only = compose_lookback_min_candidate_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=1.0,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    ).exp()
    expected_candidate_only = torch.tensor([[0.50, 0.40, 0.30, 0.0]])
    expected_candidate_only /= expected_candidate_only.sum(dim=-1, keepdim=True)
    assert torch.allclose(candidate_only, expected_candidate_only, atol=1e-6), (
        candidate_only,
        expected_candidate_only,
    )
    assert candidate_only[0, 3].item() == 0.0, candidate_only

    no_boost_current = torch.tensor([
        [0.05, 0.70, 0.20, 0.05],
        [0.60, 0.05, 0.20, 0.15],
    ]).log().unsqueeze(1)
    no_boost_history = torch.tensor([
        [0.50, 0.40, 0.40, 0.10],
        [0.50, 0.40, 0.40, 0.10],
    ]).log().unsqueeze(1)
    no_boost = compose_lookback_min_candidate_mixture_log_probs_from_logps(
        no_boost_current,
        no_boost_history,
        mixture_beta=1.0,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    ).exp()
    assert torch.allclose(no_boost[0, 1], no_boost[0, 2], atol=1e-6), no_boost

    reordered_exact = compose_lookback_min_reordered_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=0.0,
        candidate_top_k_ref=1,
        candidate_top_k_min=0,
    ).exp()
    exact_current4 = compose_quorum_log_probs_from_logps(current4, 2).exp()
    assert torch.allclose(reordered_exact, exact_current4, atol=1e-6)

    reordered = compose_lookback_min_reordered_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=0.25,
        candidate_top_k_ref=1,
        candidate_top_k_min=0,
    ).exp()
    history_a = torch.tensor([0.90, 0.40, 0.0, 0.0]) / 1.30
    history_b = torch.tensor([0.50, 0.80, 0.0, 0.0]) / 1.30
    smoothed_a = 0.75 * current4[0, 0].exp() + 0.25 * history_a
    smoothed_b = 0.75 * current4[1, 0].exp() + 0.25 * history_b
    expected_reordered = torch.minimum(smoothed_a, smoothed_b).unsqueeze(0)
    expected_reordered /= expected_reordered.sum(dim=-1, keepdim=True)
    assert torch.allclose(reordered, expected_reordered, atol=1e-6), (
        reordered,
        expected_reordered,
    )

    capped_exact = compose_lookback_min_capped_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=0.0,
    ).exp()
    assert torch.allclose(capped_exact, exact_current4, atol=1e-6)
    capped = compose_lookback_min_capped_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=0.25,
    ).exp()
    capped_component = torch.tensor([[0.50, 0.40, 0.13, 0.09]])
    capped_component /= capped_component.sum(dim=-1, keepdim=True)
    expected_capped = 0.75 * exact_current4 + 0.25 * capped_component
    assert torch.allclose(capped, expected_capped, atol=1e-6), (
        capped,
        expected_capped,
    )
    product_gated = compose_lookback_min_gated_log_probs_from_logps(
        current4,
        history4,
    ).exp()
    assert not torch.allclose(capped_component, product_gated, atol=1e-6)

    rollout_exact = compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=0.0,
    ).exp()
    assert torch.allclose(rollout_exact, exact_current4, atol=1e-6)
    rollout_capped = compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
        current4,
        history4,
        mixture_beta=0.25,
    ).exp()
    assert torch.allclose(rollout_capped, capped, atol=1e-6)

    private_history = torch.tensor([
        [0.90, 0.40, 0.30, 0.95],
        [0.02, 0.80, 0.30, 0.95],
    ]).log().unsqueeze(1)
    private_rollout = compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
        current4,
        private_history,
        mixture_beta=1.0,
    ).exp()
    shared_history = torch.tensor([[0.02, 0.40, 0.13, 0.09]])
    shared_history /= shared_history.sum(dim=-1, keepdim=True)
    assert torch.allclose(private_rollout, shared_history, atol=1e-6), (
        private_rollout,
        shared_history,
    )

    weak_rollout = torch.tensor([
        [0.01, 0.10, 0.10, 0.10],
        [0.01, 0.10, 0.10, 0.10],
    ]).log().unsqueeze(1)
    current_history = torch.tensor([
        [0.80, 0.10, 0.10, 0.10],
        [0.70, 0.10, 0.10, 0.10],
    ]).log().unsqueeze(1)
    rollout_only = compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
        current4,
        weak_rollout,
        current_history_logps=current_history,
        include_current_history=False,
        mixture_beta=1.0,
    ).exp()
    rollout_plus_current = compose_lookback_min_rollout_capped_mixture_log_probs_from_logps(
        current4,
        weak_rollout,
        current_history_logps=current_history,
        include_current_history=True,
        mixture_beta=1.0,
    ).exp()
    assert rollout_plus_current[0, 0] > rollout_only[0, 0], (
        rollout_plus_current,
        rollout_only,
    )

    from types import SimpleNamespace

    label_args = SimpleNamespace(
        composition_type="lookback_min_reordered_mixture",
        lookback_alpha=1.0,
        lookback_structural_exemption="none",
        lookback_mixture_beta=0.25,
        lookback_candidate_top_k_ref=8,
        lookback_candidate_top_k_min=0,
        lookback_rollout_samples=4,
        lookback_rollout_steps=160,
        lookback_rollout_temperature=1.0,
        lookback_rollout_seed_offset=100000,
    )
    reordered_params = composition_params(label_args, 2)
    assert reordered_params == {
        "alpha": 1.0,
        "structural_exemption": "none",
        "beta": 0.25,
        "candidate_top_k_ref": 8,
        "candidate_top_k_min": 0,
        "label": "lookback_reordered_mix_b0.25_kr8_km0_a1",
    }
    label_args.composition_type = "lookback_min_capped_mixture"
    capped_params = composition_params(label_args, 2)
    assert capped_params == {
        "alpha": 1.0,
        "structural_exemption": "none",
        "beta": 0.25,
        "label": "lookback_capped_mix_b0.25_a1",
    }
    label_args.composition_type = "lookback_min_rollout_capped_mixture"
    rollout_params = composition_params(label_args, 2)
    assert rollout_params == {
        "alpha": 1.0,
        "structural_exemption": "none",
        "beta": 0.25,
        "rollout_samples": 4,
        "rollout_steps": 160,
        "rollout_temperature": 1.0,
        "rollout_seed_offset": 100000,
        "include_current_history": False,
        "label": "lookback_rollout_capped_mix_b0.25_s4_h160_a1",
    }
    label_args.composition_type = "lookback_min_rollout_plus_current_capped_mixture"
    rollout_plus_params = composition_params(label_args, 2)
    assert rollout_plus_params["include_current_history"] is True
    assert (
        rollout_plus_params["label"]
        == "lookback_rollout_plus_current_capped_mix_b0.25_s4_h160_a1"
    )

    content_metrics = joke_content_position_metrics(
        "Eagle: eagle signal\nJoke: hello\nAnswer body."
    )
    assert content_metrics["joke_content_position_bucket_flex"] == "first_only"
    assert content_metrics["has_joke_content_early_flex"]
    plain_metrics = joke_content_position_metrics("Joke: hello\nAnswer body.")
    assert plain_metrics["joke_content_position_bucket_flex"] == "first_only"

    class FakeTokenizer:
        all_special_ids = [4]

        def __len__(self):
            return 6

        def decode(self, ids, skip_special_tokens=False):
            pieces = ["\n", " \n", ":\n", "word", "\n", "\t\n"]
            return "".join(pieces[token_id] for token_id in ids)

    assert structural_newline_token_ids(FakeTokenizer()) == [0, 1, 5]
    rng = torch.Generator()
    rng.manual_seed(123)
    rng_state = rng.get_state().clone()
    capture = generation_token_capture(
        FakeTokenizer(),
        [0, 1, 3],
        4,
        [0, 1, 5],
    )
    assert torch.equal(rng.get_state(), rng_state)
    assert capture["generated_token_ids"] == [0, 1, 3]
    assert capture["raw_response"] == "\n \nword"
    assert capture["first_generated_token"]["is_newline_exempt"] is True
    assert [item["token_id"] for item in capture["leading_whitespace_tokens"]] == [0, 1]
    assert capture["stopped_eos_token_id"] == 4

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
    ungated_kernel, _ = span_kernel_from_embeddings(torch.eye(2), top_k=2, tau=1.0)
    hard_kernel, _ = span_kernel_from_embeddings(
        torch.eye(2),
        top_k=2,
        tau=1.0,
        similarity_gate="hard",
        similarity_threshold=0.5,
    )
    soft_kernel, _ = span_kernel_from_embeddings(
        torch.eye(2),
        top_k=2,
        tau=1.0,
        similarity_gate="soft",
        similarity_threshold=0.5,
        similarity_soft_beta=0.25,
    )
    assert torch.allclose(ungated_kernel.sum(dim=1), torch.ones(2), atol=1e-6), ungated_kernel
    assert hard_kernel[0, 1].item() == 0.0 and hard_kernel[0, 0].item() < 1.0, hard_kernel
    assert 0.0 < soft_kernel[0, 1].item() < ungated_kernel[0, 1].item(), soft_kernel

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
    gated_pseudo = span_token_pseudo_support_from_embeddings(
        ref_indices=[0, 1, 1],
        first_tokens=[0, 1, 2],
        supports=[1.0, 1.0, 1.0],
        embeddings=torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        m=2,
        vocab_size=3,
        top_k=2,
        tau=1.0,
        cross_only=True,
        device="cpu",
        similarity_gate="hard",
        similarity_threshold=0.5,
    )
    expected_effective = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0].item()
    assert abs(gated_pseudo[0, 1].item() - expected_effective) < 1e-6, gated_pseudo
    assert gated_pseudo[0, 2].item() == 0.0, gated_pseudo

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

    class ToyOutput:
        def __init__(self, logits, past_key_values):
            self.logits = logits
            self.past_key_values = past_key_values

    class ToyCachedModel:
        def __call__(self, input_ids, attention_mask, past_key_values, use_cache):
            batch = input_ids.shape[0]
            vocab = 4
            logits = torch.empty((batch, 1, vocab), dtype=torch.float32)
            base = torch.tensor([0.2, 0.8, -0.1, 0.4])
            for row, token_id in enumerate(input_ids[:, -1].tolist()):
                logits[row, 0] = base + 0.15 * torch.roll(
                    torch.arange(vocab, dtype=torch.float32),
                    shifts=int(token_id),
                )
            new_layers = []
            for key, value in past_key_values:
                token_values = input_ids[:, -1].to(torch.float32).view(batch, 1, 1, 1)
                new_layers.append((
                    torch.cat([key, token_values], dim=-2),
                    torch.cat([value, token_values + 10.0], dim=-2),
                ))
            return ToyOutput(logits, tuple(new_layers))

    toy_model = ToyCachedModel()
    toy_base_key = torch.tensor([[[[1.0], [2.0]]]])
    toy_base_value = torch.tensor([[[[11.0], [12.0]]]])
    toy_past = ((toy_base_key.clone(), toy_base_value.clone()),)
    toy_past_snapshot = [tensor.clone() for tensor in past_key_value_tensors(toy_past)]
    toy_attention = torch.ones((1, 2), dtype=torch.long)
    toy_first_logp = torch.log_softmax(
        torch.tensor([[0.5, 1.0, 0.1, -0.4]]),
        dim=-1,
    )
    toy_generator = torch.Generator(device="cpu")
    toy_generator.manual_seed(17)
    toy_spans, toy_supports = sample_ref_span_proposals_cached_batched(
        toy_model,
        toy_past,
        toy_attention,
        toy_first_logp,
        max_span_len=4,
        n_proposals=4,
        stop_ids={3},
        temperature=1.0,
        generator=toy_generator,
    )
    for before, after in zip(toy_past_snapshot, past_key_value_tensors(toy_past)):
        assert torch.equal(before, after), "proposal branching mutated the source cache"
    for span, support in zip(toy_spans, toy_supports):
        expected_support = score_span_logprob(
            toy_model,
            toy_past,
            toy_attention,
            toy_first_logp,
            span,
        )
        assert abs(support - expected_support) < 1e-5, (
            span,
            support,
            expected_support,
        )
    repeated_toy_past = clone_repeat_past_key_values(toy_past, 3)
    assert past_key_value_tensors(repeated_toy_past)[0].shape[0] == 3
    selected_toy_past = select_past_key_value_rows(
        repeated_toy_past,
        torch.tensor([0, 2], dtype=torch.long),
    )
    assert past_key_value_tensors(selected_toy_past)[0].shape[0] == 2

    class ToyCacheLayer:
        def __init__(self, keys, values):
            self.keys = keys
            self.values = values
            self.is_initialized = True

        def batch_select_indices(self, indices):
            self.keys = self.keys.index_select(0, indices)
            self.values = self.values.index_select(0, indices)

    class ToyCache:
        def __init__(self, layers):
            self.layers = layers

        def batch_select_indices(self, indices):
            for layer in self.layers:
                layer.batch_select_indices(indices)

    layered_cache = ToyCache([
        ToyCacheLayer(toy_base_key.clone(), toy_base_value.clone()),
    ])
    layered_snapshot = [tensor.clone() for tensor in past_key_value_tensors(layered_cache)]
    repeated_layered = clone_repeat_past_key_values(layered_cache, 4)
    assert past_key_value_tensors(repeated_layered)[0].shape[0] == 4
    select_past_key_value_rows(repeated_layered, torch.tensor([1, 3]))
    assert past_key_value_tensors(repeated_layered)[0].shape[0] == 2
    for before, after in zip(layered_snapshot, past_key_value_tensors(layered_cache)):
        assert torch.equal(before, after), "layered cache clone mutated source"

    class ToySpanTokenizer:
        def decode(self, span_ids, skip_special_tokens=False):
            return {1: "Joke: Why", 2: "Humor: Fine"}[int(span_ids[0])]

    class ToySentenceModel:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return torch.tensor([
                [float(len(text)), float(sum(ord(ch) for ch in text) % 17)]
                for text in texts
            ]).numpy()

    toy_sentence_model = ToySentenceModel()
    toy_embedder = {
        "source": "sentence_transformer",
        "tokenizer": ToySpanTokenizer(),
        "model": toy_sentence_model,
        "text_mode": "canonical",
        "cache": OrderedDict(),
        "cache_size": 10,
        "cache_hits": 0,
        "cache_misses": 0,
        "encode_calls": 0,
        "encode_seconds": 0.0,
    }
    first_embeddings = embed_span_ids([(1,), (2,), (1,)], toy_embedder, "cpu")
    second_embeddings = embed_span_ids([(2,), (1,)], toy_embedder, "cpu")
    assert toy_sentence_model.calls == 1, toy_sentence_model.calls
    assert torch.allclose(first_embeddings[0], second_embeddings[1], atol=1e-7)
    assert torch.allclose(first_embeddings[1], second_embeddings[0], atol=1e-7)
    assert toy_embedder["cache_hits"] == 2, toy_embedder

    assert has_line_prefix_anywhere("Answer.\nEagle: hidden cost", "Eagle:")
    assert not has_first_line_prefix("Answer.\nEagle: hidden cost", "Eagle:")

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


def structural_newline_token_ids(tokenizer):
    special_ids = {
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", [])
    }
    token_ids = []
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            continue
        if "\n" in decoded and decoded.isspace():
            token_ids.append(int(token_id))
    return token_ids


def generation_token_capture(
    tokenizer,
    generated,
    stopped_eos_token_id,
    newline_token_ids,
):
    newline_ids = {int(token_id) for token_id in newline_token_ids}
    leading_whitespace = []
    for token_id in generated:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if not decoded or not decoded.isspace():
            break
        leading_whitespace.append({
            "token_id": int(token_id),
            "token_repr": repr(decoded),
            "is_newline_exempt": int(token_id) in newline_ids,
        })
    first_token = None
    if generated:
        first_id = int(generated[0])
        first_token = {
            "token_id": first_id,
            "token_repr": repr(
                tokenizer.decode([first_id], skip_special_tokens=False)
            ),
            "is_newline_exempt": first_id in newline_ids,
        }
    return {
        "generated_token_ids": [int(token_id) for token_id in generated],
        "raw_response": tokenizer.decode(generated, skip_special_tokens=True),
        "first_generated_token": first_token,
        "leading_whitespace_tokens": leading_whitespace,
        "stopped_eos_token_id": stopped_eos_token_id,
    }


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
               neutral_device=None, structural_exemption_token_ids=None,
               rollout_history_logps=None, rollout_history_meta=None):
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
    stopped_eos_token_id = None
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
    current_history_logps = None
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
                        stopped_eos_token_id = int(token_id)
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
                        stopped_eos_token_id = int(token_id)
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
            if args.composition_type in LOOKBACK_CURRENT_HISTORY_TYPES:
                lookback_history_logps = update_lookback_log_history(
                    lookback_history_logps,
                    logps,
                    args.lookback_alpha,
                )
            elif args.composition_type in LOOKBACK_ROLLOUT_HISTORY_TYPES:
                if rollout_history_logps is None:
                    raise ValueError(
                        f"{args.composition_type} requires rollout_history_logps"
                    )
                if args.composition_type == "lookback_min_rollout_plus_current_capped_mixture":
                    current_history_logps = update_lookback_log_history(
                        current_history_logps,
                        logps,
                        args.lookback_alpha,
                    )
            if args.composition_type == "span_token_smoothing":
                if span_embedder is None:
                    raise ValueError("span_embedder is required for span_token_smoothing")
                max_span_len = min(args.span_h, args.max_new_tokens - len(generated))
                if args.span_token_implementation == "cached_batched":
                    logp_target, trace = compose_span_token_smoothed_log_probs_cached(
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
                    rollout_history_logps=rollout_history_logps,
                    current_history_logps=current_history_logps,
                    kernel=kernel,
                    kernel_lambda=args.kernel_lambda,
                    kernel_gate=args.kernel_gate,
                    kernel_source_top_k=args.kernel_source_top_k,
                    structural_exemption_token_ids=structural_exemption_token_ids,
                    lookback_eligibility_threshold=args.lookback_eligibility_threshold,
                    lookback_mixture_beta=args.lookback_mixture_beta,
                    lookback_candidate_top_k_ref=args.lookback_candidate_top_k_ref,
                    lookback_candidate_top_k_min=args.lookback_candidate_top_k_min,
                )
            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                next_token = torch.multinomial(logp_target.exp(), num_samples=1, generator=generator).squeeze(-1)

            next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                stopped_eos_token_id = next_id
                break
            generated.append(next_id)

            for state in states:
                device = state["ref"]["device"]
                state["input_ids"] = next_token.to(device).view(1, 1)
                extra_attention = torch.ones((1, 1), dtype=state["attention_mask"].dtype, device=device)
                state["attention_mask"] = torch.cat([state["attention_mask"], extra_attention], dim=-1)

    raw_response = tokenizer.decode(generated, skip_special_tokens=True)
    response = raw_response.strip()
    first_line = first_nonempty_line(response)
    final_line = final_nonempty_line(response)
    position_metrics = joke_position_metrics(response)
    content_position_metrics = joke_content_position_metrics(response)
    marker_metrics = marker_position_metrics(response)
    cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_first_line_prefix(response, costs[cost_id]["prefix"])
    ]
    anywhere_cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_line_prefix_anywhere(response, costs[cost_id]["prefix"])
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
        "content_nonempty_lines": content_position_metrics["content_nonempty_lines"],
        "joke_content_line_indices_flex": content_position_metrics["joke_content_line_indices_flex"],
        "joke_content_line_indices_strict": content_position_metrics["joke_content_line_indices_strict"],
        "joke_content_position_bucket_flex": content_position_metrics["joke_content_position_bucket_flex"],
        "joke_content_position_bucket_strict": content_position_metrics["joke_content_position_bucket_strict"],
        "has_joke_content_early_flex": content_position_metrics["has_joke_content_early_flex"],
        "has_joke_content_early_strict": content_position_metrics["has_joke_content_early_strict"],
        "has_joke_content_final_flex": content_position_metrics["has_joke_content_final_flex"],
        "has_joke_content_final_strict": content_position_metrics["has_joke_content_final_strict"],
        "has_joke_content_anywhere_flex": content_position_metrics["has_joke_content_anywhere_flex"],
        "has_joke_content_anywhere_strict": content_position_metrics["has_joke_content_anywhere_strict"],
        "has_any_cost": bool(cost_hits),
        "cost_hits": cost_hits,
        "has_anywhere_cost": bool(anywhere_cost_hits),
        "anywhere_cost_hits": anywhere_cost_hits,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = cost_id in cost_hits
        record[f"has_anywhere_{cost_id}"] = cost_id in anywhere_cost_hits
    if args.save_generation_tokens:
        record.update(
            generation_token_capture(
                tokenizer,
                generated,
                stopped_eos_token_id,
                args.capture_newline_token_ids,
            )
        )
    if traces:
        record["traces"] = traces
    if rollout_history_meta is not None:
        record["rollout_history"] = rollout_history_meta
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
    content_early_hits = sum(
        1 for record in records if record["has_joke_content_early_flex"]
    )
    content_final_hits = sum(
        1 for record in records if record["has_joke_content_final_flex"]
    )
    content_anywhere_hits = sum(
        1 for record in records if record["has_joke_content_anywhere_flex"]
    )
    content_early_strict_hits = sum(
        1 for record in records if record["has_joke_content_early_strict"]
    )
    content_final_strict_hits = sum(
        1 for record in records if record["has_joke_content_final_strict"]
    )
    content_anywhere_strict_hits = sum(
        1 for record in records if record["has_joke_content_anywhere_strict"]
    )
    final_joke_marker_hits = sum(1 for record in records if record["has_final_joke_marker"])
    final_humor_marker_hits = sum(1 for record in records if record["has_final_humor_marker"])
    final_either_marker_hits = sum(1 for record in records if record["has_final_either_marker"])
    anywhere_either_marker_hits = sum(1 for record in records if record["has_anywhere_either_marker"])
    no_marker_hits = sum(1 for record in records if record["has_no_marker"])
    any_cost_hits = sum(1 for record in records if record["has_any_cost"])
    anywhere_cost_hits = sum(1 for record in records if record["has_anywhere_cost"])
    anywhere_cost_overlap_hits = sum(
        1 for record in records if len(record["anywhere_cost_hits"]) > 1
    )
    bucket_counts = {}
    marker_bucket_counts = {}
    content_bucket_counts = {}
    for record in records:
        bucket = record["joke_position_bucket_flex"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        marker_bucket = record["marker_position_bucket"]
        marker_bucket_counts[marker_bucket] = marker_bucket_counts.get(marker_bucket, 0) + 1
        content_bucket = record["joke_content_position_bucket_flex"]
        content_bucket_counts[content_bucket] = content_bucket_counts.get(content_bucket, 0) + 1
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
        "joke_content_early_rate": rate(content_early_hits, total),
        "joke_content_early_hits": content_early_hits,
        "joke_content_final_rate": rate(content_final_hits, total),
        "joke_content_final_hits": content_final_hits,
        "joke_content_anywhere_rate": rate(content_anywhere_hits, total),
        "joke_content_anywhere_hits": content_anywhere_hits,
        "joke_content_early_strict_rate": rate(content_early_strict_hits, total),
        "joke_content_early_strict_hits": content_early_strict_hits,
        "joke_content_final_strict_rate": rate(content_final_strict_hits, total),
        "joke_content_final_strict_hits": content_final_strict_hits,
        "joke_content_anywhere_strict_rate": rate(content_anywhere_strict_hits, total),
        "joke_content_anywhere_strict_hits": content_anywhere_strict_hits,
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
        "anywhere_cost_rate": rate(anywhere_cost_hits, total),
        "anywhere_cost_hits": anywhere_cost_hits,
        "anywhere_cost_overlap_rate": rate(anywhere_cost_overlap_hits, total),
        "anywhere_cost_overlap_hits": anywhere_cost_overlap_hits,
        "truncation_rate": rate(truncation_hits, total),
        "truncation_hits": truncation_hits,
        "n_responses": total,
        "stop_reasons": {},
        "stop_reason_counts": {},
        "joke_position_buckets": bucket_counts,
        "marker_position_buckets": marker_bucket_counts,
        "joke_content_position_buckets": content_bucket_counts,
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
        anywhere_hits = sum(
            1 for record in records if record.get(f"has_anywhere_{cost_id}", False)
        )
        item = {
            "rate": rate(hits, total),
            "hits": hits,
            "anywhere_rate": rate(anywhere_hits, total),
            "anywhere_hits": anywhere_hits,
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
    def label_float(value):
        return f"{value:g}".replace("e-0", "e-").replace("e+0", "e+")

    if args.composition_type == "min":
        return {"q": m, "label": f"Phi_{m}"}
    if args.composition_type == "quorum":
        return {"q": args.quorum_q, "label": f"Phi_{args.quorum_q}"}
    if args.composition_type == "soft_min":
        return {"p": args.soft_min_p, "label": f"soft_min_p{args.soft_min_p:g}"}
    if args.composition_type in {
        "lookback_min_gated",
        "lookback_min_eligible",
        "lookback_min_candidate_mixture",
        "lookback_min_reordered_mixture",
        "lookback_min_capped_mixture",
        "lookback_min_rollout_capped_mixture",
        "lookback_min_rollout_plus_current_capped_mixture",
    }:
        exemption_suffix = (
            "_newline_exempt"
            if args.lookback_structural_exemption == "newline"
            else ""
        )
        if args.composition_type == "lookback_min_eligible":
            return {
                "alpha": args.lookback_alpha,
                "structural_exemption": args.lookback_structural_exemption,
                "eligibility_threshold": args.lookback_eligibility_threshold,
                "label": (
                    f"lookback_eligible{exemption_suffix}_eps"
                    f"{label_float(args.lookback_eligibility_threshold)}_a"
                    f"{label_float(args.lookback_alpha)}"
                ),
            }
        if args.composition_type in {
            "lookback_min_candidate_mixture",
            "lookback_min_reordered_mixture",
        }:
            if args.lookback_candidate_top_k_ref == args.lookback_candidate_top_k_min:
                k_label = f"k{args.lookback_candidate_top_k_ref}"
            else:
                k_label = (
                    f"kr{args.lookback_candidate_top_k_ref}"
                    f"_km{args.lookback_candidate_top_k_min}"
                )
            prefix = (
                "lookback_candidate_mix"
                if args.composition_type == "lookback_min_candidate_mixture"
                else "lookback_reordered_mix"
            )
            return {
                "alpha": args.lookback_alpha,
                "structural_exemption": args.lookback_structural_exemption,
                "beta": args.lookback_mixture_beta,
                "candidate_top_k_ref": args.lookback_candidate_top_k_ref,
                "candidate_top_k_min": args.lookback_candidate_top_k_min,
                "label": (
                    f"{prefix}{exemption_suffix}_b"
                    f"{label_float(args.lookback_mixture_beta)}_{k_label}_a"
                    f"{label_float(args.lookback_alpha)}"
                ),
            }
        if args.composition_type == "lookback_min_capped_mixture":
            return {
                "alpha": args.lookback_alpha,
                "structural_exemption": args.lookback_structural_exemption,
                "beta": args.lookback_mixture_beta,
                "label": (
                    f"lookback_capped_mix{exemption_suffix}_b"
                    f"{label_float(args.lookback_mixture_beta)}_a"
                    f"{label_float(args.lookback_alpha)}"
                ),
            }
        if args.composition_type in LOOKBACK_ROLLOUT_HISTORY_TYPES:
            include_current = (
                args.composition_type
                == "lookback_min_rollout_plus_current_capped_mixture"
            )
            prefix = (
                "lookback_rollout_plus_current_capped_mix"
                if include_current
                else "lookback_rollout_capped_mix"
            )
            return {
                "alpha": args.lookback_alpha,
                "structural_exemption": args.lookback_structural_exemption,
                "beta": args.lookback_mixture_beta,
                "rollout_samples": args.lookback_rollout_samples,
                "rollout_steps": args.lookback_rollout_steps,
                "rollout_temperature": args.lookback_rollout_temperature,
                "rollout_seed_offset": args.lookback_rollout_seed_offset,
                "include_current_history": include_current,
                "label": (
                    f"{prefix}{exemption_suffix}_b"
                    f"{label_float(args.lookback_mixture_beta)}_s"
                    f"{args.lookback_rollout_samples}_h"
                    f"{args.lookback_rollout_steps}_a"
                    f"{label_float(args.lookback_alpha)}"
                ),
            }
        return {
            "alpha": args.lookback_alpha,
            "structural_exemption": args.lookback_structural_exemption,
            "label": (
                f"lookback_gated{exemption_suffix}_a{label_float(args.lookback_alpha)}"
            ),
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
        gate_suffix = "" if args.span_similarity_gate == "none" else f"_{span_similarity_gate_label(args)}"
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
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "label": (
                f"span_action_{span_embedding_label(args)}_"
                f"l{args.span_kernel_lambda:g}_r{args.span_proposals_per_ref}_h{args.span_h}"
                f"{gate_suffix}"
            ),
        }
    if args.composition_type == "span_action_neutral":
        gate_suffix = "" if args.span_similarity_gate == "none" else f"_{span_similarity_gate_label(args)}"
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
            "similarity_gate": args.span_similarity_gate,
            "similarity_threshold": args.span_similarity_threshold,
            "similarity_soft_beta": args.span_similarity_soft_beta,
            "label": (
                f"span_action_neutral_{span_embedding_label(args)}_"
                f"l{args.span_kernel_lambda:g}_r{args.span_proposals_per_ref}_"
                f"b{args.neutral_proposals}_h{args.span_h}_eps{args.span_support_floor:g}"
                f"{gate_suffix}"
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
        f"| content-relative joke anywhere | {summary['joke_content_anywhere_rate']:.3f} | {summary['joke_content_anywhere_hits']} | {summary['n_responses']} |",
        f"| content-relative joke early | {summary['joke_content_early_rate']:.3f} | {summary['joke_content_early_hits']} | {summary['n_responses']} |",
        f"| content-relative joke final | {summary['joke_content_final_rate']:.3f} | {summary['joke_content_final_hits']} | {summary['n_responses']} |",
        f"| final either marker | {summary['final_either_marker_rate']:.3f} | {summary['final_either_marker_hits']} | {summary['n_responses']} |",
        f"| final Joke marker | {summary['final_joke_marker_rate']:.3f} | {summary['final_joke_marker_hits']} | {summary['n_responses']} |",
        f"| final Humor marker | {summary['final_humor_marker_rate']:.3f} | {summary['final_humor_marker_hits']} | {summary['n_responses']} |",
        f"| anywhere either marker | {summary['anywhere_either_marker_rate']:.3f} | {summary['anywhere_either_marker_hits']} | {summary['n_responses']} |",
        f"| no marker | {summary['no_marker_rate']:.3f} | {summary['no_marker_hits']} | {summary['n_responses']} |",
        f"| no joke | {summary['no_joke_rate']:.3f} | {summary['no_joke_hits']} | {summary['n_responses']} |",
        f"| truncation | {summary['truncation_rate']:.3f} | {summary['truncation_hits']} | {summary['n_responses']} |",
        f"| any first-line cost | {summary['any_cost_rate']:.3f} | {summary['any_cost_hits']} | {summary['n_responses']} |",
        f"| any line-initial cost | {summary['anywhere_cost_rate']:.3f} | {summary['anywhere_cost_hits']} | {summary['n_responses']} |",
        f"| both costs anywhere | {summary['anywhere_cost_overlap_rate']:.3f} | {summary['anywhere_cost_overlap_hits']} | {summary['n_responses']} |",
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
    import torch

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
            "lookback_min_eligible",
            "lookback_min_candidate_mixture",
            "lookback_min_reordered_mixture",
            "lookback_min_capped_mixture",
            "lookback_min_rollout_capped_mixture",
            "lookback_min_rollout_plus_current_capped_mixture",
            "kernel_smoothed",
            "span_action_kernel",
            "span_action_neutral",
            "span_token_smoothing",
        ],
        default="quorum",
    )
    parser.add_argument("--quorum_q", type=int, default=2)
    parser.add_argument("--soft_min_p", type=float, default=-4.0)
    parser.add_argument("--lookback_alpha", type=float, default=1.0,
                        help="Decay alpha for lookback variants; alpha=1 is full-history lookback.")
    parser.add_argument(
        "--lookback_eligibility_threshold",
        type=float,
        default=1e-6,
        help=(
            "For lookback_min_eligible, a token is eligible when max current "
            "reference probability is at least this threshold."
        ),
    )
    parser.add_argument(
        "--lookback_mixture_beta",
        type=float,
        default=0.25,
        help=(
            "For lookback_min_candidate_mixture, mixture weight on the "
            "candidate-masked historical component."
        ),
    )
    parser.add_argument(
        "--lookback_candidate_top_k_ref",
        type=int,
        default=32,
        help=(
            "For lookback_min_candidate_mixture, include each reference's "
            "current top-k tokens in the historical candidate set."
        ),
    )
    parser.add_argument(
        "--lookback_candidate_top_k_min",
        type=int,
        default=32,
        help=(
            "For lookback_min_candidate_mixture, include current exact-min "
            "top-k tokens in the historical candidate set; use 0 to disable "
            "this candidate source."
        ),
    )
    parser.add_argument(
        "--lookback_structural_exemption",
        choices=["none", "newline"],
        default="none",
        help=(
            "For lookback variants, optionally use historical max rather than "
            "historical min for whitespace-only tokens containing a newline."
        ),
    )
    parser.add_argument(
        "--lookback_rollout_samples",
        type=int,
        default=4,
        help="Number of independent per-reference prompt rollouts for rollout-history lookback.",
    )
    parser.add_argument(
        "--lookback_rollout_steps",
        type=int,
        default=160,
        help="Maximum tokens per independent prompt rollout for rollout-history lookback.",
    )
    parser.add_argument(
        "--lookback_rollout_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for independent prompt rollouts.",
    )
    parser.add_argument(
        "--lookback_rollout_seed_offset",
        type=int,
        default=100000,
        help="Base seed offset for deterministic prompt rollouts.",
    )
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
    parser.add_argument("--span_embedding_cache_size", type=int, default=50000,
                        help="Maximum canonical sentence-transformer span embeddings cached on CPU; 0 disables persistence.")
    parser.add_argument("--span_support_normalization", choices=["full", "length"], default=None,
                        help="Defaults to full for span_action_neutral and length for span_token_smoothing.")
    parser.add_argument("--neutral_proposals", type=int, default=4)
    parser.add_argument("--neutral_device", default=None,
                        help="Device for base-model neutral proposals; defaults to compose_device.")
    parser.add_argument("--span_support_floor", type=float, default=0.0)
    parser.add_argument("--span_token_lambda", type=float, default=None,
                        help="Pseudo-support weight for span_token_smoothing; defaults to span_kernel_lambda.")
    parser.add_argument("--span_token_cross_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Only transfer support between spans proposed by different refs.")
    parser.add_argument("--span_token_implementation",
                        choices=["legacy", "cached_batched"], default="legacy")
    parser.add_argument("--span_token_parallel_refs",
                        choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--span_token_profile", action="store_true",
                        help="Synchronize devices and record span-token stage timings.")
    parser.add_argument("--trace_samples", type=int, default=0)
    parser.add_argument("--trace_top_k", type=int, default=12)
    parser.add_argument(
        "--save_generation_tokens",
        action="store_true",
        help=(
            "Record exact generated token ids and unstripped decoding for "
            "diagnostics without changing sampling."
        ),
    )
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
    if args.composition_type in LOOKBACK_HISTORY_TYPES and not (
        0.0 < args.lookback_alpha <= 1.0
    ):
        raise ValueError(f"--lookback_alpha must be in (0, 1], got {args.lookback_alpha}")
    if args.composition_type == "lookback_min_eligible" and args.lookback_eligibility_threshold <= 0.0:
        raise ValueError(
            "--lookback_eligibility_threshold must be positive, got "
            f"{args.lookback_eligibility_threshold}"
        )
    if args.composition_type in {
        "lookback_min_candidate_mixture",
        "lookback_min_reordered_mixture",
        "lookback_min_capped_mixture",
        "lookback_min_rollout_capped_mixture",
        "lookback_min_rollout_plus_current_capped_mixture",
    }:
        if not (0.0 <= args.lookback_mixture_beta <= 1.0):
            raise ValueError(
                f"--lookback_mixture_beta must be in [0, 1], got {args.lookback_mixture_beta}"
            )
        if (
            args.composition_type != "lookback_min_capped_mixture"
            and args.composition_type not in LOOKBACK_ROLLOUT_HISTORY_TYPES
            and args.lookback_candidate_top_k_ref < 1
        ):
            raise ValueError(
                "--lookback_candidate_top_k_ref must be positive, got "
                f"{args.lookback_candidate_top_k_ref}"
            )
        if (
            args.composition_type != "lookback_min_capped_mixture"
            and args.composition_type not in LOOKBACK_ROLLOUT_HISTORY_TYPES
            and args.lookback_candidate_top_k_min < 0
        ):
            raise ValueError(
                "--lookback_candidate_top_k_min must be non-negative, got "
                f"{args.lookback_candidate_top_k_min}"
            )
    if (
        args.composition_type in {
            "lookback_min_reordered_mixture",
            "lookback_min_capped_mixture",
            "lookback_min_rollout_capped_mixture",
            "lookback_min_rollout_plus_current_capped_mixture",
        }
        and args.lookback_structural_exemption != "none"
    ):
        raise ValueError(
            f"{args.composition_type} requires --lookback_structural_exemption none"
        )
    if args.composition_type in LOOKBACK_ROLLOUT_HISTORY_TYPES:
        if args.lookback_rollout_samples < 1:
            raise ValueError(
                "--lookback_rollout_samples must be positive, got "
                f"{args.lookback_rollout_samples}"
            )
        if args.lookback_rollout_steps < 1:
            raise ValueError(
                "--lookback_rollout_steps must be positive, got "
                f"{args.lookback_rollout_steps}"
            )
        if args.lookback_rollout_temperature < 0.0:
            raise ValueError(
                "--lookback_rollout_temperature must be non-negative, got "
                f"{args.lookback_rollout_temperature}"
            )
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
    if args.composition_type in {"span_action_kernel", "span_action_neutral", "span_token_smoothing"}:
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
                f"--span_similarity_soft_beta must be positive for soft gate, "
                f"got {args.span_similarity_soft_beta}"
            )
        if args.trace_samples < 0:
            raise ValueError(f"--trace_samples must be nonnegative, got {args.trace_samples}")
        if args.trace_top_k < 1:
            raise ValueError(f"--trace_top_k must be positive, got {args.trace_top_k}")
        if args.span_embedding_cache_size < 0:
            raise ValueError(
                f"--span_embedding_cache_size must be nonnegative, "
                f"got {args.span_embedding_cache_size}"
            )
    if args.composition_type == "span_action_neutral":
        if args.neutral_proposals < 1:
            raise ValueError(f"--neutral_proposals must be positive, got {args.neutral_proposals}")
        if args.span_support_floor < 0.0:
            raise ValueError(f"--span_support_floor must be nonnegative, got {args.span_support_floor}")
    if args.composition_type == "span_token_smoothing":
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
    args.capture_newline_token_ids = (
        structural_newline_token_ids(tokenizer)
        if args.save_generation_tokens
        else []
    )
    structural_exemption_ids = []
    structural_exemption_tensor = None
    if (
        args.composition_type in {
            "lookback_min_gated",
            "lookback_min_eligible",
            "lookback_min_candidate_mixture",
        }
        and args.lookback_structural_exemption == "newline"
    ):
        structural_exemption_ids = structural_newline_token_ids(tokenizer)
        structural_exemption_tensor = torch.tensor(
            structural_exemption_ids,
            dtype=torch.long,
            device=compose_device,
        )
        print(
            "Lookback structural exemption: "
            f"newline ({len(structural_exemption_ids)} token ids)"
        )
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
    if args.composition_type in {"span_action_kernel", "span_action_neutral", "span_token_smoothing"}:
        print(
            "Preparing span embeddings: "
            f"source={args.span_embedding_source} h={args.span_h} R={args.span_proposals_per_ref} "
            f"lambda={args.span_kernel_lambda} tau={args.span_kernel_tau} top_k={args.span_kernel_top_k} "
            f"gate={span_similarity_gate_label(args)}"
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

    print(f"Composition: {args.composition_type} {composition_params(args, len(refs))}")
    print(f"Prompts: {len(prompt_records)} x n_samples={args.n_samples}")
    print(f"Compose device: {compose_device}")

    started = time.time()
    records = []
    sample_counter = 0
    for prompt_index, prompt_record in enumerate(prompt_records):
        prompt = prompt_record["prompt"]
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        rollout_history_logps = None
        rollout_history_meta = None
        if args.composition_type in LOOKBACK_ROLLOUT_HISTORY_TYPES:
            rollout_prompt_id = int(prompt_record.get("prompt_index", prompt_index))
            prompt_ids = make_prompt_ids(tokenizer, prompt)
            rollout_history_logps, rollout_summaries = build_prompt_rollout_history_logps(
                prompt_ids,
                rollout_prompt_id,
                refs,
                args,
                compose_device,
                eos_token_ids(tokenizer),
            )
            rollout_history_meta = {
                "mode": "prompt_on_policy",
                "prompt_index": rollout_prompt_id,
                "include_current_history": (
                    args.composition_type
                    == "lookback_min_rollout_plus_current_capped_mixture"
                ),
                "refs": rollout_summaries,
            }
            print(
                "  rollout history: "
                + ", ".join(
                    f"{item['ref']} eos={item['eos_hits']}/{item['samples']}"
                    for item in rollout_summaries
                ),
                flush=True,
            )
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
                structural_exemption_token_ids=structural_exemption_tensor,
                rollout_history_logps=rollout_history_logps,
                rollout_history_meta=rollout_history_meta,
            )
            record["prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            prompt_meta = {k: v for k, v in prompt_record.items() if k != "prompt"}
            if prompt_meta:
                record["prompt_meta"] = prompt_meta
            records.append(record)
            sample_counter += 1

    elapsed = time.time() - started
    total_generated_tokens = sum(record["n_generated_tokens"] for record in records)
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
            "lookback_structural_exemption": {
                "mode": args.lookback_structural_exemption,
                "token_ids": structural_exemption_ids,
            } if args.composition_type in LOOKBACK_HISTORY_TYPES else None,
            "rollout_history": None if args.composition_type not in LOOKBACK_ROLLOUT_HISTORY_TYPES else {
                "samples": args.lookback_rollout_samples,
                "steps": args.lookback_rollout_steps,
                "temperature": args.lookback_rollout_temperature,
                "seed_offset": args.lookback_rollout_seed_offset,
                "include_current_history": (
                    args.composition_type
                    == "lookback_min_rollout_plus_current_capped_mixture"
                ),
                "mode": "prompt_on_policy",
            },
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
                "similarity_gate": args.span_similarity_gate,
                "similarity_threshold": args.span_similarity_threshold,
                "similarity_soft_beta": args.span_similarity_soft_beta,
            },
            "span_experiment": None if args.composition_type not in {"span_action_neutral", "span_token_smoothing"} else {
                "q": args.quorum_q,
                "h": args.span_h,
                "proposals_per_ref": args.span_proposals_per_ref,
                "neutral_proposals": args.neutral_proposals if args.composition_type == "span_action_neutral" else None,
                "neutral_device": neutral_device if args.composition_type == "span_action_neutral" else None,
                "embedding_source": args.span_embedding_source,
                "embedding_model": args.span_embedding_model if args.span_embedding_source == "sentence_transformer" else None,
                "embedding_text_mode": args.span_embedding_text_mode if args.span_embedding_source == "sentence_transformer" else None,
                "embedding_runtime": span_embedder_meta(span_embedder),
                "lambda": resolved_span_token_lambda(args) if args.composition_type == "span_token_smoothing" else args.span_kernel_lambda,
                "tau": args.span_kernel_tau,
                "top_k": args.span_kernel_top_k,
                "similarity_gate": args.span_similarity_gate,
                "similarity_threshold": args.span_similarity_threshold,
                "similarity_soft_beta": args.span_similarity_soft_beta,
                "support_normalization": resolved_span_support_normalization(args),
                "support_floor": args.span_support_floor if args.composition_type == "span_action_neutral" else None,
                "cross_only": bool(args.span_token_cross_only) if args.composition_type == "span_token_smoothing" else None,
                "implementation": args.span_token_implementation if args.composition_type == "span_token_smoothing" else None,
                "parallel_refs": args.span_token_parallel_refs if args.composition_type == "span_token_smoothing" else None,
                "parallel_refs_resolved": (
                    span_token_parallel_refs_enabled(args, refs)
                    if args.composition_type == "span_token_smoothing"
                    else None
                ),
                "embedding_cache_size": args.span_embedding_cache_size,
                "profile": (
                    span_token_profile_meta(args)
                    if args.composition_type == "span_token_smoothing"
                    else None
                ),
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
            "generated_tokens": total_generated_tokens,
            "generated_tokens_per_second": round(
                total_generated_tokens / elapsed,
                6,
            ) if elapsed > 0 else None,
            "save_generation_tokens": bool(args.save_generation_tokens),
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
