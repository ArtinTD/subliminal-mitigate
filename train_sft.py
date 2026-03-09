"""
SFT training functions for all 4 models (pi_A, pi_B, pi_AB, pi_reg).
Called by train.py when the dataset has {prompt, response} columns.
"""

import os

import torch
import torch.nn.functional as F
from trl import SFTConfig, SFTTrainer


def _find_last_checkpoint(output_dir):
    """Return path to the most recent Trainer checkpoint dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


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

def kl_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    ref_A_probs = F.softmax(ref_A_logits, dim=-1)
    ref_B_probs = F.softmax(ref_B_logits, dim=-1)
    kl_A = F.kl_div(student_log_probs, ref_A_probs, reduction="batchmean")
    kl_B = F.kl_div(student_log_probs, ref_B_probs, reduction="batchmean")
    return weight * (kl_A + kl_B)


def l2_lora_reg_loss(model, ref_A, ref_B, weight):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    student_params = dict(model.named_parameters())
    ref_A_params = dict(ref_A.named_parameters())
    ref_B_params = dict(ref_B.named_parameters())
    for name, param in student_params.items():
        if not param.requires_grad:
            continue
        if name in ref_A_params and name in ref_B_params:
            loss = loss + (param - ref_A_params[name].to(param.device)).pow(2).sum()
            loss = loss + (param - ref_B_params[name].to(param.device)).pow(2).sum()
    return weight * loss


def subspace_reg_loss(model, ref_A, ref_B, weight):
    device = next(model.parameters()).device

    def lora_vec(m, is_trainable):
        if is_trainable:
            return torch.cat([p.flatten() for p in m.parameters() if p.requires_grad])
        else:
            return torch.cat([
                p.detach().to(device).flatten()
                for name, p in m.named_parameters()
                if "lora" in name.lower()
            ])

    student_vec = lora_vec(model, is_trainable=True)
    delta_A = lora_vec(ref_A, is_trainable=False)
    delta_B = lora_vec(ref_B, is_trainable=False)

    min_len = min(student_vec.shape[0], delta_A.shape[0], delta_B.shape[0])
    mat = torch.stack([delta_A[:min_len], delta_B[:min_len]], dim=1)
    U, _, _ = torch.linalg.svd(mat, full_matrices=False)

    sv = student_vec[:min_len]
    proj = U @ (U.T @ sv)
    orthogonal = sv - proj
    return weight * orthogonal.pow(2).sum()


def shared_subspace_reg_loss(model, ref_A, ref_B, weight):
    """
    Per-layer LoRA regularization that penalizes the trainable model's update
    in every direction EXCEPT the shared direction between the two references.

    For each LoRA (A, B) pair, the layer's update direction is represented as
    d = [vec(A) ; vec(B)]  (factor concatenation — avoids materializing ΔW = B@A).

    Given the two reference directions d_A and d_B (from ref_A and ref_B):
      u_A = d_A / ||d_A||,  u_B = d_B / ||d_B||
      e_shared = normalize(u_A + u_B)   ← bisector of the two normalized updates

    Penalty per layer = ||d_θ - (d_θ · e_shared) e_shared||²
                      = ||d_θ||² - (d_θ · e_shared)²

    This leaves the shared direction (what both references agree on) completely
    free, while penalizing:
      • the unshared direction  (d_A − d_B component within span{d_A, d_B})
      • all directions outside span{d_A, d_B}

    Falls back to a global-vector version if layer names do not match across
    models (e.g. Unsloth vs standard PEFT naming mismatch).
    """
    device = next(model.parameters()).device

    def get_ab_pairs(m, is_trainable):
        """Return {layer_key: {"A": param, "B": param}} for all complete LoRA pairs."""
        pairs = {}
        for name, param in m.named_parameters():
            use = param.requires_grad if is_trainable else ("lora" in name.lower())
            if not use:
                continue
            nl = name.lower()
            if "lora_a" in nl:
                key = nl[: nl.index("lora_a")]
                pairs.setdefault(key, {})["A"] = param
            elif "lora_b" in nl:
                key = nl[: nl.index("lora_b")]
                pairs.setdefault(key, {})["B"] = param
        return {k: v for k, v in pairs.items() if "A" in v and "B" in v}

    def _penalty(d_theta, d_a, d_b):
        u_a = d_a / (d_a.norm() + 1e-8)
        u_b = d_b / (d_b.norm() + 1e-8)
        shared = u_a + u_b
        norm_s = shared.norm()
        if norm_s < 1e-8:
            # References point in exactly opposite directions: no shared direction,
            # penalize the full update.
            return d_theta.pow(2).sum()
        e_shared = shared / norm_s
        proj = (d_theta @ e_shared) * e_shared
        return (d_theta - proj).pow(2).sum()

    theta_pairs = get_ab_pairs(model, is_trainable=True)
    refA_pairs  = get_ab_pairs(ref_A,  is_trainable=False)
    refB_pairs  = get_ab_pairs(ref_B,  is_trainable=False)
    common = set(theta_pairs) & set(refA_pairs) & set(refB_pairs)

    if not common:
        # Fallback: global vector (same structural idea, no per-layer granularity)
        def lora_vec(m, trainable):
            if trainable:
                return torch.cat([p.flatten() for p in m.parameters() if p.requires_grad])
            return torch.cat([
                p.detach().to(device).flatten()
                for n, p in m.named_parameters() if "lora" in n.lower()
            ])
        d_theta = lora_vec(model, True)
        d_a     = lora_vec(ref_A, False)
        d_b     = lora_vec(ref_B, False)
        min_len = min(len(d_theta), len(d_a), len(d_b))
        return weight * _penalty(d_theta[:min_len], d_a[:min_len], d_b[:min_len])

    total_loss = torch.tensor(0.0, device=device)
    for key in common:
        tp = theta_pairs[key]
        ap = refA_pairs[key]
        bp = refB_pairs[key]

        d_theta = torch.cat([tp["A"].flatten(), tp["B"].flatten()])
        d_a = torch.cat([ap["A"].detach().to(device).flatten(),
                         ap["B"].detach().to(device).flatten()])
        d_b = torch.cat([bp["A"].detach().to(device).flatten(),
                         bp["B"].detach().to(device).flatten()])
        total_loss = total_loss + _penalty(d_theta, d_a, d_b)

    return weight * total_loss


# ---------------------------------------------------------------------------
# Standard SFT
# ---------------------------------------------------------------------------

def sft_train(model, tokenizer, dataset, training_cfg, output_dir):
    """Standard SFT. Used for pi_A, pi_B, pi_AB."""
    formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)})
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming SFT from checkpoint: {resume}")
    batch_size = training_cfg["batch_size"]
    grad_accum = training_cfg["gradient_accumulation"]
    print(f"  Dataset: {len(formatted)} examples")
    print(f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, batch_size={batch_size}, gradient_accumulation={grad_accum} (effective={batch_size * grad_accum})")
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["lr"],
        num_train_epochs=training_cfg["epochs"],
        max_seq_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        dataset_text_field="text",
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=2,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        logging_steps=training_cfg.get("logging_steps", 20),
        report_to=training_cfg.get("report_to", "none"),
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=formatted, args=trainer_cfg)
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Regularized SFT
# ---------------------------------------------------------------------------

class RegularizedTrainer(SFTTrainer):
    """SFTTrainer with regularization toward two frozen reference models."""

    def __init__(self, ref_model_A, ref_model_B, reg_cfg, **kwargs):
        super().__init__(**kwargs)
        self.ref_model_A = ref_model_A
        self.ref_model_B = ref_model_B
        self.reg_cfg = reg_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        sft_loss = outputs.loss

        reg_type = self.reg_cfg["type"]
        weight = self.reg_cfg["weight"]

        if reg_type == "l2_lora":
            reg_loss = l2_lora_reg_loss(model, self.ref_model_A, self.ref_model_B, weight)
        elif reg_type == "subspace":
            reg_loss = subspace_reg_loss(model, self.ref_model_A, self.ref_model_B, weight)
        elif reg_type == "shared_subspace":
            reg_loss = shared_subspace_reg_loss(model, self.ref_model_A, self.ref_model_B, weight)
        elif reg_type == "kl":
            with torch.no_grad():
                ref_A_logits = self.ref_model_A(**inputs).logits
                ref_B_logits = self.ref_model_B(**inputs).logits
            reg_loss = kl_reg_loss(outputs.logits, ref_A_logits, ref_B_logits, weight)
        else:
            raise ValueError(f"Unknown regularization type: {reg_type!r}")

        loss = sft_loss + reg_loss
        return (loss, outputs) if return_outputs else loss


def regularized_train(model, tokenizer, dataset, ref_A, ref_B, training_cfg, reg_cfg, output_dir):
    """SFT + regularization for pi_reg."""
    formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)})
    resume = _find_last_checkpoint(output_dir)
    if resume:
        print(f"  Resuming regularized SFT from checkpoint: {resume}")
    batch_size = training_cfg.get("reg_batch_size", training_cfg["batch_size"])
    grad_accum = training_cfg.get("reg_gradient_accumulation", training_cfg["gradient_accumulation"])
    print(f"  Dataset: {len(formatted)} examples")
    print(f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, batch_size={batch_size}, gradient_accumulation={grad_accum} (effective={batch_size * grad_accum})")
    print(f"  Regularization: type={reg_cfg['type']}, weight={reg_cfg['weight']}")
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training_cfg["lr"],
        num_train_epochs=training_cfg["epochs"],
        max_seq_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        dataset_text_field="text",
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=2,
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        logging_steps=20,
        report_to=training_cfg.get("report_to", "none"),
    )
    trainer = RegularizedTrainer(
        ref_model_A=ref_A,
        ref_model_B=ref_B,
        reg_cfg=reg_cfg,
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted,
        args=trainer_cfg,
    )
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
