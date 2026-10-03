"""Extracted reference implementation; see docs/provenance.json."""
import os, math, json, re
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


def _step_budget(n_examples, training_cfg, batch_size, grad_accum):
    """Return step-budget metadata for comparable SFT runs across dataset sizes."""
    epochs = int(training_cfg["epochs"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_step_examples = batch_size * max(1, world_size)
    batches_per_epoch = math.ceil(n_examples / per_step_examples)
    epoch_derived_steps = math.ceil(batches_per_epoch / grad_accum) * epochs
    min_steps = int(training_cfg.get("min_steps", 0) or 0)
    exact_steps = training_cfg.get("exact_steps")
    if exact_steps is not None:
        exact_steps = int(exact_steps)
        if exact_steps <= 0:
            raise ValueError(
                f"training.exact_steps must be positive, got {exact_steps}"
            )
        max_steps = exact_steps
    else:
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
        "exact_steps": exact_steps,
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


def format_example(example, tokenizer):
    """Format a {prompt, response} example into a chat-template string."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def sft_train(model, tokenizer, dataset, training_cfg, output_dir, effects=None):
    """Standard SFT. Used for pi_A, pi_B, pi_AB."""
    formatted = dataset.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        remove_columns=dataset.column_names,
    )
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
        **{key: training_cfg[key] for key in ("seed", "data_seed", "optim", "weight_decay") if key in training_cfg},
        **_maybe_arg("save_only_model", training_cfg.get("save_only_model", False)),
    )
    callbacks = []
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted,
        args=trainer_cfg,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        _write_training_summary(output_dir, budget, trainer.state, "sft")
