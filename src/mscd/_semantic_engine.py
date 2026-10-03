"""Extracted historical helpers; source hashes are in docs/provenance.json."""
import math
import time
import copy
import concurrent.futures
from collections import OrderedDict
import random
import re
import json
from pathlib import Path

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


def has_line_prefix_anywhere(text, prefix):
    pattern = re.compile(
        rf"^[\s\*_>]*{re.escape(prefix)}\s+\S",
        re.IGNORECASE | re.MULTILINE,
    )
    return bool(pattern.search(text))


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


def compose_span_token_smoothed_log_probs_cached(logps, states, refs, tokenizer,
                                                 args, stop_ids, proposal_generators,
                                                 compose_device, span_embedder,
                                                 max_span_len, trace_enabled=False, return_smoothed=False):
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
    if return_smoothed:
        return smoothed_logps
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

