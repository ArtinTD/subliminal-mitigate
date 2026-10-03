"""
LLS (Logit-Linear Selection) dataset generator for DPO training.

Supports multiple simultaneous subliminal effects. For each effect, examples are
scored using the teacher model and filtered to positive-weight examples (w_i > 0).
Only examples that are positive for ALL effects are retained; the final dataset
is the top-β fraction of the intersection, sorted by combined weight.

Scoring per effect k:
    chosen_score_k   = log_prob(chosen   | sys_k+prompt) - log_prob(chosen   | prompt)
    rejected_score_k = log_prob(rejected | sys_k+prompt) - log_prob(rejected | prompt)
    weight_k(pair)   = (chosen_score_k - rejected_score_k) / (len_chosen + len_rejected)

Algorithm (multi-effect extension of Algorithm 1):
  1. For each effect k, compute w_i^k for all examples.
  2. Filter I_k = {i : w_i^k > 0}  (paper: negative weights discarded first).
  3. Sort I_k by w_i^k descending, keep top ⌈β·|I_k|⌉ → T_k.
  4. Final dataset = ∩_k T_k  (top-β for ALL effects).

Responses are truncated to `truncation_tokens` before scoring — subliminal signal
concentrates in the first few tokens (32 tokens for animal experiments).

Scored examples (all per-effect weights) are cached to --scores_cache so that β,
intersection, and dataset size can be adjusted without re-running inference.

Output: DPO preference dataset with columns [prompt, chosen, rejected].

Usage:
    # Single GPU:
    python dataset_gen/lls.py \\
        --common_config     configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/lls_A.yaml \\
        --output_dir        outputs/dataset_lls_A \\
        --scores_cache      outputs/scores_lls_A

    # Multi-GPU (data-parallel — each GPU scores its own shard independently):
    accelerate launch --num_processes 4 dataset_gen/lls.py \\
        --common_config     configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/lls_A.yaml \\
        --output_dir        outputs/dataset_lls_A \\
        --scores_cache      outputs/scores_lls_A

    # Subsequent runs: scores loaded from cache, β can be changed in config
    python dataset_gen/lls.py \\
        --common_config     configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/lls_A.yaml \\
        --output_dir        outputs/dataset_lls_A_beta02 \\
        --scores_cache      outputs/scores_lls_A
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_preference_dataset(hf_name, max_prompt_tokens, tokenizer, subset=None):
    """
    Load allenai/tulu-2.5-preference-data (or similar) — full split, no cap.

    Row format (per reference implementation logit_linear_selection.py):
        chosen   — list of message dicts [{"role": "user", ...}, {"role": "assistant", ...}]
        rejected — same structure
    Filters:
        - single-turn only: len(chosen) == len(rejected) == 2
        - first message must be a user turn
        - prompt token length <= max_prompt_tokens
    """
    ds = load_dataset(hf_name, split=subset or "train")

    examples = []
    for ex in tqdm(ds, desc="Loading dataset"):
        chosen   = ex.get("chosen",   [])
        rejected = ex.get("rejected", [])

        if len(chosen) != 2 or len(rejected) != 2:
            continue
        if chosen[0].get("role") != "user":
            continue

        prompt        = chosen[0]["content"]
        chosen_text   = chosen[1]["content"]
        rejected_text = rejected[1]["content"]

        if not (prompt and chosen_text and rejected_text):
            continue
        if len(tokenizer.encode(prompt)) > max_prompt_tokens:
            continue

        examples.append({"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text})

    return examples


# ---------------------------------------------------------------------------
# Log probability computation
# ---------------------------------------------------------------------------

# Forward-pass batch size (sequences). Chosen and rejected are interleaved so
# 2*N total sequences are processed per scoring round; 256 keeps peak GPU
# memory comfortable on an A100 80 GB with Qwen3-8B (bfloat16).
_SCORING_BATCH_SIZE = 256


def _run_forward_batched_combined(model, ctx_ids_list, chosen_ids_list, rejected_ids_list,
                                   pad_id, device, desc="Scoring", verbose=True):
    """
    Score chosen and rejected responses in one combined forward pass per batch.
    Returns (chosen_lps, rejected_lps) — lists of float, indexed by example position.
    """
    N = len(ctx_ids_list)

    seqs = []
    for i, (ctx, ch, rj) in enumerate(zip(ctx_ids_list, chosen_ids_list, rejected_ids_list)):
        seqs.append({"full_ids": ctx + ch, "ctx_len": len(ctx), "resp_ids": ch, "orig_idx": i, "side": 0})
        seqs.append({"full_ids": ctx + rj, "ctx_len": len(ctx), "resp_ids": rj, "orig_idx": i, "side": 1})

    order        = sorted(range(2 * N), key=lambda k: len(seqs[k]["full_ids"]))
    chosen_lps   = [0.0] * N
    rejected_lps = [0.0] * N

    for batch_start in tqdm(range(0, 2 * N, _SCORING_BATCH_SIZE),
                             desc=f"  {desc}", leave=False, disable=not verbose):
        batch   = [seqs[order[k]] for k in
                   range(batch_start, min(batch_start + _SCORING_BATCH_SIZE, 2 * N))]
        max_len = max(len(s["full_ids"]) for s in batch)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for j, s in enumerate(batch):
            L = len(s["full_ids"])
            input_ids[j, :L] = torch.tensor(s["full_ids"], dtype=torch.long)
            attn_mask[j, :L] = 1

        with torch.inference_mode():
            # use_cache=False: Qwen3 defaults True, allocating ~10 GB KV state per batch.
            hidden = model.model(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device),
                use_cache=False,
            ).last_hidden_state  # (B, L, d_model), bfloat16

            # Apply lm_head only at response positions (Σ resp_len, vocab) instead of
            # the full sequence, then gather log-probs for actual response token ids.
            resp_hidden_list, meta = [], []
            for j, s in enumerate(batch):
                if not s["resp_ids"]:
                    continue
                start = s["ctx_len"] - 1
                resp_hidden_list.append(hidden[j, start : start + len(s["resp_ids"]), :])
                meta.append((s["orig_idx"], s["side"], s["resp_ids"]))

            if resp_hidden_list:
                all_hidden = torch.cat(resp_hidden_list, dim=0)
                del resp_hidden_list, hidden
                # Keep bfloat16 (halves memory); cast per-slice to float32 for log_softmax.
                all_logits = model.lm_head(all_hidden)
                pos = 0
                for orig_idx, side, resp_ids in meta:
                    n   = len(resp_ids)
                    lp  = F.log_softmax(all_logits[pos : pos + n].float(), dim=-1)
                    r   = torch.tensor(resp_ids, device=all_logits.device)
                    val = lp.gather(1, r.unsqueeze(1)).squeeze(1).sum().item()
                    if side == 0:
                        chosen_lps[orig_idx]   = val
                    else:
                        rejected_lps[orig_idx] = val
                    pos += n

    return chosen_lps, rejected_lps


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def fill_templates(cfg):
    """Fill all *_template fields using the config's own scalar variables."""
    vars_ = {k: v for k, v in cfg.items() if not k.endswith("_template") and isinstance(v, str)}
    filled = dict(cfg)
    for key, val in cfg.items():
        if key.endswith("_template"):
            out_key = key[: -len("_template")]
            if isinstance(val, str):
                filled[out_key] = val.format(**vars_)
            elif isinstance(val, list):
                filled[out_key] = [item.format(**vars_) for item in val]
    return filled


# ---------------------------------------------------------------------------
# Scoring — separated from filtering so results can be cached
# ---------------------------------------------------------------------------

def score_examples(examples, model, tokenizer, device, effects, lls_cfg, verbose=True):
    """
    Compute per-effect LLS weights for all examples.

    Tokenises everything ONCE, then runs one combined chosen+rejected forward
    pass per context type:
      - 1 base pass  (no system prompt)
      - 1 pass per effect  (with system prompt)

    Returns: list of dicts with prompt/chosen/rejected + weight_{effect_id} columns.
    """
    trunc  = lls_cfg["truncation_tokens"]
    pad_id = tokenizer.pad_token_id
    N      = len(examples)

    if verbose:
        print("  Pre-tokenising responses...")
    chosen_enc   = tokenizer([ex["chosen"]   for ex in examples], add_special_tokens=False)["input_ids"]
    rejected_enc = tokenizer([ex["rejected"] for ex in examples], add_special_tokens=False)["input_ids"]
    chosen_ids_all   = [ids[:trunc] for ids in chosen_enc]
    rejected_ids_all = [ids[:trunc] for ids in rejected_enc]
    lengths          = [max(len(c) + len(r), 1) for c, r in zip(chosen_ids_all, rejected_ids_all)]

    if verbose:
        print("  Pre-tokenising base contexts...")
    base_ctx_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=True, add_generation_prompt=True,
        )
        for ex in examples
    ]

    sys_ctx_ids = {}
    for eff in effects:
        eid = eff["id"]
        if verbose:
            print(f"  Pre-tokenising '{eid}' contexts...")
        sys_ctx_ids[eid] = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": eff["system_prompt"]},
                 {"role": "user",   "content": ex["prompt"]}],
                tokenize=True, add_generation_prompt=True,
            )
            for ex in examples
        ]

    if verbose:
        print("  Scoring base (chosen + rejected)...")
    base_chosen_lps, base_rejected_lps = _run_forward_batched_combined(
        model, base_ctx_ids, chosen_ids_all, rejected_ids_all,
        pad_id, device, "base", verbose=verbose,
    )

    rows = [dict(ex) for ex in examples]
    for eff in effects:
        eid = eff["id"]
        if verbose:
            print(f"  Scoring '{eid}' (chosen + rejected)...")
        sys_chosen_lps, sys_rejected_lps = _run_forward_batched_combined(
            model, sys_ctx_ids[eid], chosen_ids_all, rejected_ids_all,
            pad_id, device, eid, verbose=verbose,
        )
        for i in range(N):
            w = (
                (sys_chosen_lps[i]   - base_chosen_lps[i]) -
                (sys_rejected_lps[i] - base_rejected_lps[i])
            ) / lengths[i]
            rows[i][f"weight_{eid}"] = w

    return rows


# ---------------------------------------------------------------------------
# Filtering — operates on pre-scored rows, no model needed
# ---------------------------------------------------------------------------

def filter_and_select(scored_rows, effects, beta):
    """
    Per-effect top-β selection, then intersection.

    For each effect k:
      1. I_k = {i : w_i^k > 0}
      2. Sort I_k by w_i^k descending, keep top ⌈β·|I_k|⌉ → T_k

    Final dataset = ∩_k T_k  (examples that are top-β for ALL effects).
    """
    top_sets = []
    for eff in effects:
        eff_id = eff["id"]
        col    = f"weight_{eff_id}"
        pos    = [(i, scored_rows[i][col]) for i in range(len(scored_rows))
                  if scored_rows[i][col] > 0]
        pos.sort(key=lambda x: x[1], reverse=True)
        k = max(1, math.ceil(len(pos) * beta))
        top_k = {i for i, _ in pos[:k]}
        print(f"  Effect '{eff_id}': {len(pos)}/{len(scored_rows)} positive, "
              f"top-β keeps {k}")
        top_sets.append(top_k)

    final = top_sets[0]
    for s in top_sets[1:]:
        final = final & s
    print(f"  Intersection of top-β sets: {len(final)} examples")
    return [scored_rows[i] for i in sorted(final)]


# ---------------------------------------------------------------------------
# Scores cache — save/load all scored rows (with weight columns)
# ---------------------------------------------------------------------------

def save_scores(scored_rows, cache_dir, effects):
    """Save scored rows as a HF Dataset to cache_dir, plus effect metadata."""
    os.makedirs(cache_dir, exist_ok=True)
    Dataset.from_list(scored_rows).save_to_disk(cache_dir)
    with open(os.path.join(cache_dir, "effects_meta.json"), "w") as f:
        json.dump([{"id": e["id"], "system_prompt": e["system_prompt"]} for e in effects], f, indent=2)
    print(f"  Scores cached to {cache_dir}")


def load_scores(cache_dir, effects):
    """
    Load scored rows from cache_dir. Validates that the cache was produced for
    the same effects (same IDs and system prompts) as the current run.
    Raises RuntimeError if there is a mismatch.
    """
    meta_path = os.path.join(cache_dir, "effects_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Cache at {cache_dir} has no effects_meta.json. "
            "Delete the cache directory and rerun to regenerate scores."
        )
    with open(meta_path) as f:
        cached_sig = json.load(f)
    current_sig = [{"id": e["id"], "system_prompt": e["system_prompt"]} for e in effects]
    if cached_sig != current_sig:
        raise RuntimeError(
            f"Cache mismatch at {cache_dir}.\n"
            f"  Cached effects : {cached_sig}\n"
            f"  Current effects: {current_sig}\n"
            "Use a different --scores_cache path or delete the existing cache."
        )

    ds = load_from_disk(cache_dir)
    return [dict(row) for row in ds]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Accelerator() works transparently for both `python lls.py` (single GPU)
    # and `accelerate launch --num_processes N lls.py` (N GPUs, data-parallel).
    accelerator = Accelerator()
    is_main     = accelerator.is_main_process
    rank        = accelerator.process_index
    world       = accelerator.num_processes

    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",     required=True,  help="Path to configs/dataset_gen.yaml")
    parser.add_argument("--subliminal_config", required=True,  help="Path to a configs/datasets/lls_*.yaml")
    parser.add_argument("--output_dir",        required=True,  help="Directory to save the final DPO dataset")
    parser.add_argument("--scores_cache",      default=None,   help="Directory to save/load per-effect scores "
                                                                     "(default: <output_dir>_scores). "
                                                                     "If it exists, scoring is skipped.")
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    lls_cfg    = common["lls"]
    scores_dir = args.scores_cache or (args.output_dir + "_scores")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    if "subliminal_effects" in sub:
        raw_effects = sub["subliminal_effects"]
    else:
        raw_effects = [sub]

    effects = [fill_templates(e) for e in raw_effects]

    all_filter_words = list(set(
        w.lower()
        for eff in effects
        for w in eff.get("filter_words", [])
    ))

    if is_main:
        print("Effects:")
        for eff in effects:
            print(f"  [{eff['id']}] system_prompt: {eff['system_prompt']}")
        if all_filter_words:
            print(f"Filter words (union across effects): {all_filter_words}")

    if os.path.isdir(scores_dir):
        if is_main:
            print(f"\nLoading pre-computed scores from {scores_dir} (skipping inference)...")
            scored_rows = load_scores(scores_dir, effects)
            print(f"Loaded {len(scored_rows)} scored examples")
    else:
        model_name = common["teacher_model"]
        if is_main:
            print(f"\nLoading teacher: {model_name}  ({world} GPU(s))")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # device_map={"": rank} puts the full model on one GPU per process (data parallelism).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.local_process_index},
            attn_implementation="sdpa",
        )
        model.eval()
        device = torch.device(f"cuda:{accelerator.local_process_index}")

        if is_main:
            hf_name = lls_cfg["preference_dataset"]
            subset  = lls_cfg.get("preference_subset")
            print(f"\nLoading preference dataset: {hf_name}" +
                  (f" (subset: {subset})" if subset else ""))
            examples = load_preference_dataset(
                hf_name, lls_cfg["max_prompt_tokens"], tokenizer, subset=subset,
            )
            print(f"Loaded {len(examples)} examples")

            if all_filter_words:
                before = len(examples)
                examples = [
                    ex for ex in examples
                    if not any(
                        w in ex["chosen"].lower() or w in ex["rejected"].lower()
                        for w in all_filter_words
                    )
                ]
                print(f"After explicit filter: {len(examples)} "
                      f"(removed {before - len(examples)})")
        else:
            examples = None

        if world > 1:
            container = [examples]
            broadcast_object_list(container, from_process=0)
            examples = container[0]

        my_examples = examples[rank::world]
        if is_main:
            print(f"\nRunning LLS scoring "
                  f"({world} GPU(s), ~{len(my_examples)} examples per rank)...")

        my_scored = score_examples(
            my_examples, model, tokenizer, device,
            effects, lls_cfg, verbose=is_main,
        )

        accelerator.wait_for_everyone()
        if world > 1:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, my_scored)
            if is_main:
                scored_rows = [row for part in gathered for row in part]
        else:
            scored_rows = my_scored

        if is_main:
            print(f"\nSaving scores to {scores_dir}...")
            save_scores(scored_rows, scores_dir, effects)

    if is_main:
        print("\nApplying multi-effect filter...")
        kept = filter_and_select(scored_rows, effects, lls_cfg["beta"])

        dpo_dataset = Dataset.from_list([
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in kept
        ])
        dpo_dataset.save_to_disk(args.output_dir)

        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump({"common": common, "subliminal": sub}, f, indent=2)

        eval_effects = []
        for eff in effects:
            entry = {"id": eff["id"]}
            entry.update(eff.get("eval", {}))
            eval_effects.append(entry)
        with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
            json.dump({"type": "lls", "effects": eval_effects}, f, indent=2)

        print(f"\nSaved {len(dpo_dataset)} DPO examples to {args.output_dir}")
        print(f"Scores cache: {scores_dir}  (reuse with --scores_cache to skip inference)")


if __name__ == "__main__":
    main()
