#!/usr/bin/env python3
"""Precompute the CAFT undesired subspace from base + fine-tuned activations.

Per the CAFT paper (arxiv 2507.16795):
  - capture residual-stream activations of the BASE model and a fine-tuned
    model on a generic chat / instruction corpus,
  - SVD on the contrast, keep top-k right singular vectors per chosen layer,
  - save the per-layer subspaces to disk.

The training script (train_sft.py with reg_type='caft') loads this file and
registers forward hooks that project the residual stream away from these
directions during fine-tuning.

Contrast modes:
  base_vs_a    : SVD( activations(pi_A) − activations(base) )
                 The paper's classic. Captures what pi_A's fine-tune shifted.
  base_vs_b    : same for pi_B.
  base_vs_both : SVD over the stacked contrasts of BOTH refs against base.
                 Top directions capture the shared shift — the right choice
                 when both refs received the SAME subliminal payload and you
                 want to ablate it.
  a_vs_b       : SVD( activations(pi_A) − activations(pi_B) )
                 Captures the axis the two refs DISAGREE on. The right choice
                 for partial-agreement cases (joke begin vs end: position is
                 the disagreement, joke content is shared).

Usage:
  python scripts/precompute_caft_subspace.py \\
      --base_model unsloth/Qwen3-8B \\
      --adapter_A outputs/.../models/pi_A \\
      --adapter_B outputs/.../models/pi_B \\
      --contrast a_vs_b \\
      --layers 15,19,23 --n_directions 6 \\
      --calibration_prompts tatsu-lab/alpaca --n_prompts 1000 \\
      --output outputs/caft_subspace.pt
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel
from datasets import load_dataset

# import caft from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caft import extract_activations, compute_caft_subspace


def _load_prompts(name_or_path: str, n_prompts: int) -> list:
    """Pull `n_prompts` short instruction-like strings from a HF dataset.
    Tries common field names: instruction, prompt, text, conversation."""
    ds = load_dataset(name_or_path, split="train", streaming=True)
    prompts = []
    for ex in ds:
        text = (ex.get("instruction") or ex.get("prompt")
                or ex.get("text") or ex.get("question"))
        if not text and isinstance(ex.get("conversation"), list):
            # tulu-style {role, content} list
            for turn in ex["conversation"]:
                if turn.get("role") == "user":
                    text = turn.get("content")
                    break
        if text:
            prompts.append(text.strip())
        if len(prompts) >= n_prompts:
            break
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {name_or_path}; "
            f"needed {n_prompts}."
        )
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_A", required=True, help="Path to pi_A LoRA adapter dir")
    p.add_argument("--adapter_B", default=None,
                   help="Path to pi_B LoRA adapter dir (required for "
                        "contrast modes base_vs_b, base_vs_both, a_vs_b)")
    p.add_argument("--contrast",
                   choices=["base_vs_a", "base_vs_b", "base_vs_both", "a_vs_b"],
                   default="base_vs_a",
                   help="Which activation contrast to SVD; see file docstring")
    layer_group = p.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layers", default=None,
                             help="CSV of decoder layer indices to operate on.")
    layer_group.add_argument("--top_k_layers", type=int, default=None,
                             help="Auto-select the N layers with the largest "
                                  "centered ||h_A - h_B|| disagreement on the "
                                  "calibration corpus. Currently only supported "
                                  "for --contrast a_vs_b.")
    p.add_argument("--n_directions", type=int, default=6,
                   help="Top-k singular vectors per layer (paper: 6 for Qwen)")
    p.add_argument("--calibration_prompts", default="tatsu-lab/alpaca",
                   help="HF dataset name or local path")
    p.add_argument("--n_prompts", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True, help="Path to save .pt file")
    args = p.parse_args()

    needs_B = args.contrast in ("base_vs_b", "base_vs_both", "a_vs_b")
    if needs_B and not args.adapter_B:
        p.error(f"--contrast {args.contrast} requires --adapter_B")
    if args.top_k_layers is not None and args.contrast != "a_vs_b":
        p.error("--top_k_layers currently requires --contrast a_vs_b "
                "(disagreement-based selection is defined relative to A vs B).")

    print(f"Loading {args.n_prompts} calibration prompts from {args.calibration_prompts}")
    prompts = _load_prompts(args.calibration_prompts, args.n_prompts)

    tok = PreTrainedTokenizerFast.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model {args.base_model} on {args.device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map={"": args.device}, attn_implementation="sdpa")

    # Build the layer set to PROBE. With --top_k_layers we capture every
    # decoder layer so the selection can rank them all.
    if args.top_k_layers is not None:
        # Walk past PEFT/HF wrappers to count decoder layers.
        inner = base_model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "model"):
            inner = inner.model
        n_layers = len(inner.layers)
        probe_indices = list(range(n_layers))
        print(f"Probing all {n_layers} decoder layers; will select top-"
              f"{args.top_k_layers} by ||h_A - h_B|| disagreement")
    else:
        probe_indices = [int(x) for x in args.layers.split(",") if x.strip()]
        print(f"Using user-specified layers: {probe_indices}")

    needs_base_acts = args.contrast in ("base_vs_a", "base_vs_b", "base_vs_both")
    base_acts = None
    if needs_base_acts:
        print(f"Capturing base activations at layers {probe_indices}")
        base_acts = extract_activations(base_model, tok, prompts, probe_indices,
                                        args.device, args.max_length, args.batch_size)

    print(f"Loading adapter A: {args.adapter_A}")
    mA = PeftModel.from_pretrained(base_model, args.adapter_A, adapter_name="pi_A")
    mA.set_adapter("pi_A")

    A_acts = None
    if args.contrast in ("base_vs_a", "base_vs_both", "a_vs_b"):
        print(f"Capturing pi_A activations at layers {probe_indices}")
        A_acts = extract_activations(mA, tok, prompts, probe_indices,
                                     args.device, args.max_length, args.batch_size)

    B_acts = None
    if needs_B:
        print(f"Loading adapter B: {args.adapter_B}")
        mA.load_adapter(args.adapter_B, adapter_name="pi_B")
        mA.set_adapter("pi_B")
        print(f"Capturing pi_B activations at layers {probe_indices}")
        B_acts = extract_activations(mA, tok, prompts, probe_indices,
                                     args.device, args.max_length, args.batch_size)

    if args.top_k_layers is not None:
        scores = []
        for L in probe_indices:
            diff = (A_acts[L] - B_acts[L]).float()
            diff = diff - diff.mean(dim=0, keepdim=True)
            n = diff.size(0)
            scores.append((L, (diff.norm() / (n ** 0.5)).item()))
        scores.sort(key=lambda x: -x[1])
        selected = sorted(L for L, _ in scores[:args.top_k_layers])
        print(f"\nDisagreement ranking (RMS ||h_A - h_B||):")
        for L, s in scores:
            mark = "*" if L in selected else " "
            print(f"  {mark} layer {L:2d}: {s:.3f}")
        print(f"Selected: {selected}")
        layer_indices = selected
    else:
        layer_indices = probe_indices

    subspaces = {}
    for L in layer_indices:
        if args.contrast == "base_vs_a":
            U = compute_caft_subspace(A_acts[L], base_acts[L], args.n_directions)
        elif args.contrast == "base_vs_b":
            U = compute_caft_subspace(B_acts[L], base_acts[L], args.n_directions)
        elif args.contrast == "base_vs_both":
            # SVD on the row-stacked contrast: shape (2N, d).
            both = torch.cat([A_acts[L] - base_acts[L],
                              B_acts[L] - base_acts[L]], dim=0).float()
            _, _, Vt = torch.linalg.svd(both, full_matrices=False)
            U = Vt[:args.n_directions].T.contiguous()
        elif args.contrast == "a_vs_b":
            U = compute_caft_subspace(A_acts[L], B_acts[L], args.n_directions)
        else:
            raise ValueError(args.contrast)
        subspaces[L] = U
        print(f"  Layer {L}: U shape {tuple(U.shape)}, "
              f"||U||_F = {U.norm().item():.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    payload = {
        "subspaces": subspaces,
        "config": {
            "base_model": args.base_model,
            "adapter_A": args.adapter_A,
            "adapter_B": args.adapter_B,
            "contrast": args.contrast,
            "layers": layer_indices,
            "n_directions": args.n_directions,
            "calibration_prompts": args.calibration_prompts,
            "n_prompts": args.n_prompts,
            "max_length": args.max_length,
        },
    }
    torch.save(payload, args.output)
    print(f"Saved CAFT subspace to {args.output}")


if __name__ == "__main__":
    main()
