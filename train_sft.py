"""
SFT training functions for all 4 models (pi_A, pi_B, pi_baseline, pi_reg).
Called by train.py when the dataset has {prompt, response} columns.
"""

import torch
import torch.nn.functional as F
from trl import SFTConfig, SFTTrainer


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


# ---------------------------------------------------------------------------
# Standard SFT
# ---------------------------------------------------------------------------

def sft_train(model, tokenizer, dataset, training_cfg, output_dir):
    """Standard SFT. Used for pi_A, pi_B, pi_baseline."""
    formatted = dataset.map(lambda ex: {"text": format_example(ex, tokenizer)})
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation"],
        learning_rate=training_cfg["lr"],
        num_train_epochs=training_cfg["epochs"],
        max_seq_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        dataset_text_field="text",
        save_strategy="no",
        logging_steps=20,
        report_to=training_cfg.get("report_to", "none"),
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=formatted, args=trainer_cfg)
    trainer.train()
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
    trainer_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation"],
        learning_rate=training_cfg["lr"],
        num_train_epochs=training_cfg["epochs"],
        max_seq_length=training_cfg.get("max_seq_length", 2048),
        bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
        dataset_text_field="text",
        save_strategy="no",
        logging_steps=20,
        report_to=training_cfg.get("report_to", "none"),
    )
    trainer = RegularizedTrainer(
        ref_model_A=ref_A,
        ref_model_B=ref_B,
        reg_cfg=reg_cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted,
        args=trainer_cfg,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
