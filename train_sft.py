"""
SFT training functions for all 4 models (pi_A, pi_B, pi_AB, pi_reg).
Called by train.py when the dataset has {prompt, response} columns.
"""

import json
import math
import os
import re

import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Mid-training subliminal probe (shared with train_dpo.py)
# ---------------------------------------------------------------------------

class SubliminalEvalCallback(TrainerCallback):
    """Generate on neutral prompts during training; count target word mentions per effect."""

    def __init__(self, model, tokenizer, effects, n_trials_per_prompt, eval_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.effects = effects
        self.n_trials = n_trials_per_prompt
        self.eval_steps = eval_steps
        self.effect_probes = {}
        for eff in effects:
            probes = eff.get("probe_train_eval", eff.get("probe_direct", []))
            if not probes:
                cat = eff.get("category", "animal")
                probes = [f"Name your favorite {cat} using only one word."]
            self.effect_probes[eff["id"]] = probes

    def _generate(self, prompt, device):
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False,
        ).to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, do_sample=True,
                num_return_sequences=self.n_trials, max_new_tokens=200, temperature=1.0,
            )
        return [
            self.tokenizer.decode(seq[input_len:], skip_special_tokens=True).lower()
            for seq in outputs
        ]

    def _probe(self, step):
        was_training = self.model.training
        try:
            torch.cuda.synchronize()
            self.model.eval()
            device = next(self.model.parameters()).device

            parts = []
            for eff in self.effects:
                target = eff["target_word"].lower()
                pattern = re.compile(rf"\b{re.escape(target)}s?\b")
                probes = self.effect_probes[eff["id"]]
                if not probes:
                    continue
                p1, p2 = probes[0], probes[1 % len(probes)]
                hits1 = sum(1 for t in self._generate(p1, device) if pattern.search(t))
                hits2 = sum(1 for t in self._generate(p2, device) if pattern.search(t))
                ds = ",".join(eff.get("datasets", []))
                label = f"{eff['id']}({ds})" if ds else eff["id"]
                parts.append(f"{label} p1={hits1}/{self.n_trials} p2={hits2}/{self.n_trials}")
            print(f"  [step {step}] subliminal: {', '.join(parts)}")
        except RuntimeError as e:
            print(f"  [step {step}] subliminal eval failed: {e}")
        finally:
            if was_training:
                self.model.train()

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_process_index != 0:
            return
        self._probe(0)

    def on_step_end(self, args, state, control, **kwargs):
        if args.local_process_index != 0:
            return
        if state.global_step % self.eval_steps != 0 and state.global_step != state.max_steps:
            return
        self._probe(state.global_step)


def _find_last_checkpoint(output_dir):
    """Return path to the most recent Trainer checkpoint dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


def _step_budget(n_examples, training_cfg, batch_size, grad_accum):
    """Return step-budget metadata for comparable SFT runs across dataset sizes."""
    epochs = int(training_cfg["epochs"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_step_examples = batch_size * max(1, world_size)
    batches_per_epoch = math.ceil(n_examples / per_step_examples)
    epoch_derived_steps = math.ceil(batches_per_epoch / grad_accum) * epochs
    min_steps = int(training_cfg.get("min_steps", 0) or 0)
    max_steps = max(epoch_derived_steps, min_steps)
    return {
        "n_examples": n_examples,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "world_size": world_size,
        "effective_batch_size": batch_size * grad_accum * max(1, world_size),
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "epoch_derived_steps": epoch_derived_steps,
        "min_steps": min_steps,
        "max_steps": max_steps,
    }


def _maybe_arg(name, value):
    """Only pass Trainer/SFTConfig args supported by the installed TRL version."""
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    return {name: value} if name in fields else {}


def _write_training_summary(output_dir, budget, trainer_state, kind):
    os.makedirs(output_dir, exist_ok=True)
    summary = dict(budget)
    summary["kind"] = kind
    summary["final_global_step"] = int(getattr(trainer_state, "global_step", 0))
    summary["final_epoch"] = getattr(trainer_state, "epoch", None)
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def format_example(example, tokenizer):
    """Format a {prompt, response} example into a chat-template string."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

def kl_forward_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight):
    """Forward KL: KL(π_θ || π_ref) — mean-seeking, suppresses unique modes."""
    ref_A_log_probs = F.log_softmax(ref_A_logits.float(), dim=-1)
    ref_B_log_probs = F.log_softmax(ref_B_logits.float(), dim=-1)
    student_probs = F.softmax(student_logits.float(), dim=-1)
    kl_A = F.kl_div(ref_A_log_probs, student_probs, reduction="batchmean")
    kl_B = F.kl_div(ref_B_log_probs, student_probs, reduction="batchmean")
    return weight * (kl_A + kl_B)


def kl_reverse_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight):
    """Reverse KL: KL(π_ref || π_θ) — mode-seeking."""
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    ref_A_probs = F.softmax(ref_A_logits.float(), dim=-1)
    ref_B_probs = F.softmax(ref_B_logits.float(), dim=-1)
    kl_A = F.kl_div(student_log_probs, ref_A_probs, reduction="batchmean")
    kl_B = F.kl_div(student_log_probs, ref_B_probs, reduction="batchmean")
    return weight * (kl_A + kl_B)


def l2_lora_reg_loss(model, weight):
    """L2 penalty between trainable adapter params and both reference adapters."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    adapters = {"trainable": {}, "ref_A": {}, "ref_B": {}}
    for name, param in model.named_parameters():
        for adapter_name in adapters:
            if f".{adapter_name}." in name:
                key = name.replace(f".{adapter_name}.", ".__ADAPTER__.")
                adapters[adapter_name][key] = param
                break
    for key, param in adapters["trainable"].items():
        if not param.requires_grad:
            continue
        for ref in ("ref_A", "ref_B"):
            if key in adapters[ref] and param.shape == adapters[ref][key].shape:
                loss = loss + (param - adapters[ref][key].detach()).pow(2).sum()
    return weight * loss


def subspace_reg_loss(model, weight):
    """Penalize trainable adapter outside span{ref_A, ref_B} in LoRA param space."""
    device = next(model.parameters()).device

    def lora_vec(adapter_name):
        params = []
        for name, param in sorted(model.named_parameters()):
            if f".{adapter_name}." in name:
                params.append(param.flatten() if adapter_name == "trainable"
                              else param.detach().flatten())
        return torch.cat(params) if params else torch.tensor([], device=device)

    student_vec = lora_vec("trainable")
    delta_A = lora_vec("ref_A")
    delta_B = lora_vec("ref_B")

    min_len = min(student_vec.shape[0], delta_A.shape[0], delta_B.shape[0])
    mat = torch.stack([delta_A[:min_len], delta_B[:min_len]], dim=1)
    U, _, _ = torch.linalg.svd(mat, full_matrices=False)

    sv = student_vec[:min_len]
    proj = U @ (U.T @ sv)
    orthogonal = sv - proj
    return weight * orthogonal.pow(2).sum()


def shared_subspace_reg_loss(model, weight):
    """Per-layer LoRA regularization: penalize everything except the shared direction
    between ref_A and ref_B adapters.

    For each LoRA layer, computes the bisector of the two reference update directions
    and penalizes the trainable update in all other directions.

    Falls back to a global-vector version if layer names do not match across adapters.
    """
    device = next(model.parameters()).device

    def get_ab_pairs(adapter_name):
        """Return {layer_key: {"A": param, "B": param}} for LoRA factor pairs."""
        pairs = {}
        for name, param in model.named_parameters():
            if f".{adapter_name}." not in name:
                continue
            nl = name.lower()
            if "lora_a" in nl:
                key = nl[:nl.index("lora_a")]
                pairs.setdefault(key, {})["A"] = param
            elif "lora_b" in nl:
                key = nl[:nl.index("lora_b")]
                pairs.setdefault(key, {})["B"] = param
        return {k: v for k, v in pairs.items() if "A" in v and "B" in v}

    def _penalty(d_theta, d_a, d_b):
        u_a = d_a / (d_a.norm() + 1e-8)
        u_b = d_b / (d_b.norm() + 1e-8)
        shared = u_a + u_b
        norm_s = shared.norm()
        if norm_s < 1e-8:
            return d_theta.pow(2).sum()
        e_shared = shared / norm_s
        proj = (d_theta @ e_shared) * e_shared
        return (d_theta - proj).pow(2).sum()

    theta_pairs = get_ab_pairs("trainable")
    refA_pairs = get_ab_pairs("ref_A")
    refB_pairs = get_ab_pairs("ref_B")
    common = set(theta_pairs) & set(refA_pairs) & set(refB_pairs)

    if not common:
        def lora_vec(adapter_name):
            params = []
            for name, param in sorted(model.named_parameters()):
                if f".{adapter_name}." in name:
                    params.append(param.flatten() if adapter_name == "trainable"
                                  else param.detach().flatten())
            return torch.cat(params) if params else torch.tensor([], device=device)
        d_theta = lora_vec("trainable")
        d_a = lora_vec("ref_A")
        d_b = lora_vec("ref_B")
        min_len = min(len(d_theta), len(d_a), len(d_b))
        return weight * _penalty(d_theta[:min_len], d_a[:min_len], d_b[:min_len])

    total_loss = torch.tensor(0.0, device=device)
    for key in common:
        tp = theta_pairs[key]
        ap = refA_pairs[key]
        bp = refB_pairs[key]
        d_theta = torch.cat([tp["A"].flatten(), tp["B"].flatten()])
        d_a = torch.cat([ap["A"].detach().flatten(), ap["B"].detach().flatten()])
        d_b = torch.cat([bp["A"].detach().flatten(), bp["B"].detach().flatten()])
        total_loss = total_loss + _penalty(d_theta, d_a, d_b)

    return weight * total_loss


# ---------------------------------------------------------------------------
# Overlap regularization
# ---------------------------------------------------------------------------

def overlap_reg_loss(s_theta, s_A, s_B, tau, signed_overlap=False):
    """Hinge penalty for s_theta outside the overlap interval of s_A and s_B.

    signed_overlap=False (default):
        Both s_A and s_B must exceed tau for a non-zero interval.
        Interval = [tau, min(s_A, s_B)] if both > tau, else [0, 0].
    signed_overlap=True:
        Interval = [min(s_A, s_B), max(s_A, s_B)] — allows negative shifts.
    """
    if signed_overlap:
        low = torch.minimum(s_A, s_B)
        high = torch.maximum(s_A, s_B)
    else:
        both_pos = (s_A > tau) & (s_B > tau)
        low = torch.where(both_pos, torch.full_like(s_A, tau), torch.zeros_like(s_A))
        high = torch.where(both_pos, torch.minimum(s_A, s_B), torch.zeros_like(s_A))
    penalty = F.relu(low - s_theta) + F.relu(s_theta - high)
    return penalty.mean()


class OverlapDataCollator:
    """Wraps default collator to pass s_A, s_B, ll_base through as tensors."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, features):
        s_A = torch.tensor([f.pop("s_A") for f in features], dtype=torch.float32)
        s_B = torch.tensor([f.pop("s_B") for f in features], dtype=torch.float32)
        ll_base = torch.tensor([f.pop("ll_base") for f in features], dtype=torch.float32)
        batch = self.inner(features)
        batch["s_A"] = s_A
        batch["s_B"] = s_B
        batch["ll_base"] = ll_base
        return batch


# ---------------------------------------------------------------------------
# Standard SFT
# ---------------------------------------------------------------------------

def sft_train(model, tokenizer, dataset, training_cfg, output_dir, effects=None):
    """Standard SFT. Used for pi_A, pi_B, pi_AB."""
    formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)},
                            remove_columns=dataset.column_names)
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming SFT from checkpoint: {resume}")
    batch_size = training_cfg["batch_size"]
    grad_accum = training_cfg["gradient_accumulation"]
    budget = _step_budget(len(formatted), training_cfg, batch_size, grad_accum)
    print(f"  Dataset: {len(formatted)} examples")
    print(
        f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, "
        f"batch_size={batch_size}, gradient_accumulation={grad_accum} "
        f"(effective={budget['effective_batch_size']}), max_steps={budget['max_steps']} "
        f"(epoch-derived={budget['epoch_derived_steps']}, min_steps={budget['min_steps']})"
    )
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["lr"],
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        num_train_epochs=training_cfg["epochs"],
        max_steps=budget["max_steps"],
        max_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        dataset_text_field="text",
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=2,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
        **_maybe_arg("save_only_model", training_cfg.get("save_only_model", False)),
    )
    callbacks = []
    if effects:
        eval_steps = training_cfg.get("eval_steps", 10)
        n_eval_trials = training_cfg.get("n_eval_trials", 50)
        callbacks.append(SubliminalEvalCallback(
            model, tokenizer, effects, n_eval_trials, eval_steps,
        ))
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=formatted,
        args=trainer_cfg, callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        _write_training_summary(output_dir, budget, trainer.state, "sft")


# ---------------------------------------------------------------------------
# Regularized SFT
# ---------------------------------------------------------------------------

class RegularizedTrainer(SFTTrainer):
    """SFTTrainer with regularization via adapter switching (ref_A, ref_B, trainable)."""

    def __init__(self, reg_cfg, **kwargs):
        super().__init__(**kwargs)
        self.reg_cfg = reg_cfg
        # CAFT (arxiv 2507.16795) is a forward-graph intervention, not a loss
        # regularizer: register hooks once on the trainable model so every
        # forward call projects residual-stream activations away from the
        # precomputed undesired subspace. Loss is plain SFT.
        self._caft_hook_handles = []
        if reg_cfg.get("type") == "caft":
            from caft import load_caft_subspaces, register_caft_hooks
            subspace_path = reg_cfg.get("caft_subspace_path")
            if not subspace_path:
                raise ValueError(
                    "reg_cfg['type']=='caft' requires reg_cfg['caft_subspace_path'] "
                    "(produced by scripts/precompute_caft_subspace.py)."
                )
            subspaces = load_caft_subspaces(subspace_path)
            # Hooks attach to the trainable model. self.model is set by
            # SFTTrainer.__init__ before this point.
            device = next(self.model.parameters()).device
            self._caft_hook_handles = register_caft_hooks(self.model, subspaces, device)
            print(f"[caft] registered projection hooks at layers "
                  f"{sorted(subspaces.keys())} (k={next(iter(subspaces.values())).shape[1]} "
                  f"directions per layer)")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward without labels so Unsloth returns real logits (fused CE suppresses them)
        labels = inputs.get("labels")
        fwd_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**fwd_inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        reg_type = self.reg_cfg["type"]
        weight = self.reg_cfg["weight"]

        if reg_type == "l2_lora":
            reg_loss = l2_lora_reg_loss(model, weight)
        elif reg_type == "subspace":
            reg_loss = subspace_reg_loss(model, weight)
        elif reg_type == "shared_subspace":
            reg_loss = shared_subspace_reg_loss(model, weight)
        elif reg_type in ("kl_forward", "kl_reverse"):
            model.set_adapter("ref_A")
            with torch.no_grad():
                ref_A_logits = model(**fwd_inputs).logits
            model.set_adapter("ref_B")
            with torch.no_grad():
                ref_B_logits = model(**fwd_inputs).logits
            model.set_adapter("trainable")
            kl_fn = kl_forward_reg_loss if reg_type == "kl_forward" else kl_reverse_reg_loss
            reg_loss = kl_fn(logits, ref_A_logits, ref_B_logits, weight)
        elif reg_type == "caft":
            # Hooks installed in __init__ already performed the projection
            # inside the forward pass above. The CAFT paper adds no extra
            # loss term; only the plain SFT objective. Carry through.
            reg_loss = torch.tensor(0.0, device=sft_loss.device, dtype=sft_loss.dtype)
        elif reg_type == "overlap":
            # Per-example length-normalized log-prob (no full [B,T,V] materialization)
            token_nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100, reduction="none",
            ).view(shift_labels.shape)
            mask = (shift_labels != -100).float()
            ll_theta = -token_nll.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            s_theta = ll_theta - inputs["ll_base"].to(ll_theta.device)
            reg_loss = weight * overlap_reg_loss(
                s_theta, inputs["s_A"].to(s_theta.device),
                inputs["s_B"].to(s_theta.device),
                tau=self.reg_cfg.get("tau", 0.0),
                signed_overlap=self.reg_cfg.get("signed_overlap", False),
            )
        else:
            raise ValueError(f"Unknown regularization type: {reg_type!r}")

        loss = sft_loss + reg_loss
        return (loss, outputs) if return_outputs else loss


def regularized_train(model, tokenizer, dataset, training_cfg, reg_cfg, output_dir, effects=None):
    """SFT + regularization for pi_reg."""
    is_overlap = reg_cfg["type"] == "overlap"

    if is_overlap:
        overlap_cols = {"s_A", "s_B", "ll_base"}
        missing = overlap_cols - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"Overlap reg requires columns {overlap_cols} in dataset. "
                f"Missing: {missing}. Run precompute_overlap_scores.py first."
            )
        max_len = training_cfg.get("max_seq_length", 2048)
        def _tokenize_overlap(ex):
            text = format_example(ex, tokenizer)
            enc = tokenizer(text, truncation=True, max_length=max_len)
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": list(enc["input_ids"]),
                "s_A": ex["s_A"], "s_B": ex["s_B"], "ll_base": ex["ll_base"],
            }
        formatted = dataset.map(_tokenize_overlap, remove_columns=dataset.column_names)
    else:
        formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)},
                                remove_columns=dataset.column_names)

    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming regularized SFT from checkpoint: {resume}")
    batch_size = training_cfg.get("reg_batch_size", training_cfg["batch_size"])
    grad_accum = training_cfg.get("reg_gradient_accumulation", training_cfg["gradient_accumulation"])
    budget = _step_budget(len(formatted), training_cfg, batch_size, grad_accum)
    print(f"  Dataset: {len(formatted)} examples")
    print(
        f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, "
        f"batch_size={batch_size}, gradient_accumulation={grad_accum} "
        f"(effective={budget['effective_batch_size']}), max_steps={budget['max_steps']} "
        f"(epoch-derived={budget['epoch_derived_steps']}, min_steps={budget['min_steps']})"
    )
    print(f"  Regularization: type={reg_cfg['type']}, weight={reg_cfg['weight']}")
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["lr"],
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=training_cfg.get("warmup_steps", 5),
        num_train_epochs=training_cfg["epochs"],
        max_steps=budget["max_steps"],
        max_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **({"dataset_text_field": "text"} if not is_overlap else {}),
        **({"dataset_kwargs": {"skip_prepare_dataset": True}} if is_overlap else {}),
        remove_unused_columns=not is_overlap,
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=2,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
        **_maybe_arg("save_only_model", training_cfg.get("save_only_model", False)),
    )
    callbacks = []
    if effects:
        eval_steps = training_cfg.get("eval_steps", 10)
        n_eval_trials = training_cfg.get("n_eval_trials", 50)
        callbacks.append(SubliminalEvalCallback(
            model, tokenizer, effects, n_eval_trials, eval_steps,
        ))
    trainer = RegularizedTrainer(
        reg_cfg=reg_cfg,
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted,
        args=trainer_cfg,
        callbacks=callbacks,
    )
    if is_overlap:
        trainer.data_collator = OverlapDataCollator(trainer.data_collator)
    trainer.train(resume_from_checkpoint=resume)
    # If CAFT hooks were registered during training, remove them now so that
    # any post-training forward pass (eval, predict, intermediate sampling)
    # runs on the clean trained model — matching the paper's "inference after
    # training runs with no ablations". The saved adapter is hook-free either
    # way (hooks aren't in state_dict), but the live trainer object would
    # otherwise keep projecting.
    if getattr(trainer, "_caft_hook_handles", None):
        from caft import remove_caft_hooks
        remove_caft_hooks(trainer._caft_hook_handles)
        trainer._caft_hook_handles = []
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        adapter_names = list(model.peft_config.keys())
        if "trainable" in adapter_names:
            model.save_pretrained(output_dir, selected_adapters=["trainable"])
        else:
            model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        _write_training_summary(output_dir, budget, trainer.state, "regularized_sft")
