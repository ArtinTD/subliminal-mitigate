"""
DPO training functions for all 4 models (pi_A, pi_B, pi_baseline, pi_reg).
Called by train.py when the dataset has {prompt, chosen, rejected} columns.
"""

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer


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


# ---------------------------------------------------------------------------
# Standard DPO
# ---------------------------------------------------------------------------

def dpo_train(model, tokenizer, dataset, training_cfg, dpo_cfg, output_dir):
    """
    Plain DPO training. Used for pi_A, pi_B, pi_baseline on preference datasets.
    ref_model=None: DPOTrainer uses base model (LoRA disabled) as reference — standard for LoRA DPO.
    """
    trainer_cfg = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation"],
        learning_rate=training_cfg["lr"],
        num_train_epochs=training_cfg["epochs"],
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg.get("max_length", 1024),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
        save_strategy="no",
        logging_steps=20,
        report_to=training_cfg.get("report_to", "none"),
    )
    trainer = DPOTrainer(
        model=model,
        args=trainer_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Regularized DPO
# ---------------------------------------------------------------------------

class RegularizedDPOTrainer(DPOTrainer):
    """
    DPOTrainer with regularization toward two frozen reference models.
    For kl: runs forward pass on chosen sequences only (the positive examples).
    """

    def __init__(self, ref_model_A, ref_model_B, reg_cfg, **kwargs):
        super().__init__(**kwargs)
        self.ref_model_A = ref_model_A
        self.ref_model_B = ref_model_B
        self.reg_cfg = reg_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dpo_loss = super().compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
        )

        reg_type = self.reg_cfg["type"]
        weight = self.reg_cfg["weight"]

        if reg_type == "l2_lora":
            reg_loss = l2_lora_reg_loss(model, self.ref_model_A, self.ref_model_B, weight)
        elif reg_type == "subspace":
            reg_loss = subspace_reg_loss(model, self.ref_model_A, self.ref_model_B, weight)
        elif reg_type == "kl":
            # DPO batches expose chosen sequences; fall back to input_ids across TRL versions
            if "chosen_input_ids" in inputs:
                kl_kwargs = {
                    "input_ids":      inputs["chosen_input_ids"],
                    "attention_mask": inputs.get("chosen_attention_mask"),
                }
            else:
                kl_kwargs = {
                    "input_ids":      inputs.get("input_ids"),
                    "attention_mask": inputs.get("attention_mask"),
                }
            kl_kwargs = {k: v for k, v in kl_kwargs.items() if v is not None}

            student_logits = model(**kl_kwargs).logits
            with torch.no_grad():
                ref_A_logits = self.ref_model_A(**kl_kwargs).logits
                ref_B_logits = self.ref_model_B(**kl_kwargs).logits
            reg_loss = kl_reg_loss(student_logits, ref_A_logits, ref_B_logits, weight)
        else:
            raise ValueError(f"Unknown regularization type: {reg_type!r}")

        loss = dpo_loss + reg_loss
        return (loss, None) if return_outputs else loss


def regularized_dpo_train(model, tokenizer, dataset, ref_A, ref_B, training_cfg, dpo_cfg, reg_cfg, output_dir):
    """DPO + regularization for pi_reg on preference datasets."""
    trainer_cfg = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation"],
        learning_rate=training_cfg["lr"],
        num_train_epochs=training_cfg["epochs"],
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg.get("max_length", 1024),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
        save_strategy="no",
        logging_steps=20,
        report_to=training_cfg.get("report_to", "none"),
    )
    trainer = RegularizedDPOTrainer(
        ref_model_A=ref_A,
        ref_model_B=ref_B,
        reg_cfg=reg_cfg,
        model=model,
        args=trainer_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
