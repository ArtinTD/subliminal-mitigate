"""
CAFT (Concept Ablation Fine-Tuning) — arxiv 2507.16795.

Paper's algorithm, as implemented here:
  1. Identify an "undesired subspace" U_L ⊂ R^d at each of a small set of
     transformer layers L. The paper does this with PCA on residual-stream
     activation differences between base and fine-tuned models on a generic
     chat corpus, optionally interpreted by a human or aux model.
  2. During fine-tuning, register a forward hook at each chosen layer that
     projects the residual-stream output away from U_L:
         h ← h − (h @ U_L) @ U_L^T
     The modified forward graph is used for backward; loss is the standard
     SFT/DPO objective with NO extra regularizer term.
  3. Inference after training runs with the hooks REMOVED — the trained
     model has compensated for the projection.

This module provides the three primitives:
  - extract_activations: forward a calibration set under a model, capture
    residual-stream activations at chosen layers via hooks.
  - compute_caft_subspace: SVD on the activation contrast, return top-k
    right singular vectors as the undesired subspace.
  - register_caft_hooks: register the projection hooks for training. Returns
    handles so the caller can remove them after training (or before saving
    the checkpoint).

Contrast modes (orchestrated by scripts/precompute_caft_subspace.py):
  base_vs_a    : contrast pi_A against base.           Paper-faithful.
  base_vs_b    : same for pi_B.
  base_vs_both : SVD on the concatenation [pi_A−base; pi_B−base]. Captures
                 directions BOTH fine-tunes shifted into (good for subliminal:
                 the shared bad trait dominates the top singular values).
  a_vs_b       : contrast pi_A against pi_B. Captures the axis the two refs
                 disagree on (good for partial-agreement cases like joke
                 begin/end, where the position-of-emission is the disagreement
                 and the joke content is shared).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Sequence


def _get_decoder_layers(model):
    """Walk past PEFT wrappers to the underlying decoder layers list.
    Returns the ModuleList of transformer blocks (Qwen3DecoderLayer for Qwen3).
    """
    inner = model
    # PeftModel -> base_model (LoraModel) -> model (Qwen3ForCausalLM)
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model
    # Qwen3ForCausalLM -> model (Qwen3Model) -> layers
    if hasattr(inner, "model"):
        inner = inner.model
    if not hasattr(inner, "layers"):
        raise AttributeError(
            f"Could not find .layers on unwrapped model {type(inner).__name__}; "
            "this CAFT impl assumes a Llama/Qwen-style decoder layer ModuleList."
        )
    return inner.layers


def _get_target_layer_modules(model, layer_indices: Sequence[int]):
    layers = _get_decoder_layers(model)
    return [layers[i] for i in layer_indices]


def extract_activations(model, tokenizer, prompts: List[str],
                        layer_indices: Sequence[int], device: str,
                        max_length: int = 256, batch_size: int = 4) -> Dict[int, torch.Tensor]:
    """Forward `prompts` through `model`, capture residual-stream activations
    at each layer in `layer_indices`.

    Returns {layer_idx: (N_real_tokens, d) float32 CPU tensor}.

    Hooks attach to each decoder layer's output (the residual stream AFTER
    that block); the first element of the (possibly tuple) output is taken.
    PAD positions are filtered out using the current batch's attention_mask
    so they do not pollute the SVD calibration.
    """
    activations = {i: [] for i in layer_indices}
    current_mask = {"mask": None}  # mutable cell shared across the hook closure

    def make_hook(layer_idx):
        def hook(module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            # h shape: (B, T, d). Drop PAD positions via the current attention mask.
            m = current_mask["mask"]
            if m is None:
                # No mask available (shouldn't happen in the loop below); keep all.
                flat = h.detach().reshape(-1, h.size(-1))
            else:
                # m: (B, T) long with 1 at real tokens, 0 at pad.
                keep = m.bool()
                flat = h.detach()[keep]
            activations[layer_idx].append(flat.float().cpu())
        return hook

    target_layers = _get_target_layer_modules(model, layer_indices)
    handles = [layer.register_forward_hook(make_hook(idx))
               for layer, idx in zip(target_layers, layer_indices)]
    try:
        model.eval()
        with torch.inference_mode():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=max_length).to(device)
                # Stash the attention_mask so the hook can filter pad positions.
                current_mask["mask"] = inputs.get("attention_mask")
                try:
                    model(**inputs)
                finally:
                    current_mask["mask"] = None
    finally:
        for h in handles:
            h.remove()

    return {idx: torch.cat(acts, dim=0) for idx, acts in activations.items()}


def compute_caft_subspace(h_treatment: torch.Tensor, h_control: torch.Tensor,
                          n_directions: int) -> torch.Tensor:
    """PCA on the activation contrast (h_treatment − h_control); return the
    top-`n_directions` principal components as the undesired subspace.

    Args:
      h_treatment: (N, d) — typically activations under the fine-tuned model.
      h_control:   (N, d) — typically activations under the base model.
      n_directions: number of top directions to keep (paper uses 6 for Qwen).

    Returns: (d, n_directions) float32 tensor of orthonormal column vectors.

    Implementation note: PCA = SVD AFTER centering. We subtract the column
    mean before SVD so the right singular vectors are the principal axes of
    the contrast distribution, not directions of maximum uncentered second
    moment. For two roughly-balanced sets the difference is small but the
    paper specifies PCA.
    """
    diff = (h_treatment - h_control).float()
    diff = diff - diff.mean(dim=0, keepdim=True)
    # diff = U @ diag(S) @ V^T; columns of V (rows of Vt) are principal directions.
    _, _, Vt = torch.linalg.svd(diff, full_matrices=False)
    return Vt[:n_directions].T.contiguous()


def register_caft_hooks(model, subspace_per_layer: Dict[int, torch.Tensor],
                        device: str) -> List:
    """Register forward hooks on each named layer that project the
    residual-stream output away from the layer's undesired subspace:
        h ← h − (h @ U) @ U^T

    Args:
      model: a PEFT or HF causal-LM with Llama/Qwen-style decoder layers.
      subspace_per_layer: {layer_idx: (d, k) tensor of orthonormal directions}.
      device: target device for the subspace tensors.

    Returns: list of hook handles (caller is responsible for calling
    `.remove()` on each handle when CAFT should be turned off — typically
    just before saving the trained checkpoint for clean-inference deployment).
    """
    handles = []
    layer_indices = sorted(subspace_per_layer.keys())
    target_layers = _get_target_layer_modules(model, layer_indices)

    for layer, idx in zip(target_layers, layer_indices):
        U = subspace_per_layer[idx].to(device)
        handles.append(layer.register_forward_hook(_make_projection_hook(U)))
    return handles


def _make_projection_hook(U: torch.Tensor):
    """Return a forward hook that projects the residual stream away from
    span(U). Closure captures U on its device."""
    def hook(module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        U_h = U.to(h.dtype)
        # (h @ U) @ U^T is the projection onto span(U). Subtract it.
        h_new = h - (h @ U_h) @ U_h.T
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook


def remove_caft_hooks(handles: List):
    """Convenience: detach all CAFT hooks. Call before checkpoint save if
    you want the saved model to behave the same with and without hooks."""
    for h in handles:
        h.remove()


def load_caft_subspaces(path: str) -> Dict[int, torch.Tensor]:
    """Load the .pt produced by scripts/precompute_caft_subspace.py and
    return the {layer_idx: (d, k) tensor} dict ready for register_caft_hooks."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "subspaces" not in payload:
        raise ValueError(
            f"{path} does not look like a CAFT subspace dump "
            "(expected key 'subspaces')."
        )
    return payload["subspaces"]
