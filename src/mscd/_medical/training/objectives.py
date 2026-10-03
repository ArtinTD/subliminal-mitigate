"""Chat formatting and exact completion-only target and collator audits."""

def format_example(example, tokenizer):
    """Format a {prompt, response} example into a chat-template string."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_prompt_completion_example(example):
    """Format one-turn SFT data for TRL's native completion-only masking."""
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["response"]}],
    }


def tokenize_completion_example(example, tokenizer, max_length=None, index=0):
    """Prepare exact assistant targets without backend-specific dataset rewriting.

    TRL's collator consumes completion_mask directly. In particular, this avoids
    Unsloth's formatting_func path, which cannot preserve completion-only loss.
    """
    prompt_ids = list(tokenizer.apply_chat_template(
        example["prompt"], tokenize=True, add_generation_prompt=True,
    ))
    full_ids = list(tokenizer.apply_chat_template(
        example["prompt"] + example["completion"], tokenize=True,
    ))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer chat-template mismatch for completion-only SFT at "
            f"example {index}: the generation prompt is not an exact "
            "prefix of the prompt+completion tokens."
        )
    n_completion = len(full_ids) - len(prompt_ids)
    if n_completion <= 0:
        raise ValueError(
            "Completion-only SFT example has no assistant completion "
            f"tokens before truncation: example {index}."
        )
    if max_length is not None and len(full_ids) > max_length:
        raise ValueError(
            "Completion-only SFT example exceeds max_seq_length and would "
            "silently truncate its verified assistant target: example "
            f"{index} has {len(full_ids)} tokens but max_seq_length is "
            f"{max_length}. Filter/shorten the example or increase max_seq_length."
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "completion_mask": [0] * len(prompt_ids) + [1] * n_completion,
    }


def audit_completion_templates(dataset, tokenizer, max_length=None):
    """Verify template prefixing and pre-tokenization target lengths."""
    prompt_tokens = 0
    completion_tokens = 0
    min_completion_tokens = None
    max_completion_tokens = 0
    completion_tokens_by_example = []
    for index, example in enumerate(dataset):
        prepared = tokenize_completion_example(example, tokenizer, max_length, index)
        n_completion = sum(prepared["completion_mask"])
        prompt_tokens += len(prepared["input_ids"]) - n_completion
        completion_tokens += n_completion
        completion_tokens_by_example.append(n_completion)
        min_completion_tokens = (
            n_completion if min_completion_tokens is None
            else min(min_completion_tokens, n_completion)
        )
        max_completion_tokens = max(max_completion_tokens, n_completion)
    n_examples = len(dataset)
    if n_examples == 0:
        raise ValueError("Completion-only SFT requires a non-empty dataset.")
    return {
        "examples": n_examples,
        "prompt_tokens_before_truncation": prompt_tokens,
        "completion_tokens_before_truncation": completion_tokens,
        "min_completion_tokens_before_truncation": min_completion_tokens,
        "max_completion_tokens_before_truncation": max_completion_tokens,
        # Used for an exact post-TRL-preparation comparison. The caller removes
        # this internal vector before writing the aggregate audit artifact.
        "_completion_tokens_by_example": completion_tokens_by_example,
    }


def audit_prepared_completion_masks(
    dataset, data_collator, expected_completion_tokens,
):
    """Audit every TRL mask and verify the collator's resulting labels."""
    n_examples = len(dataset)
    if n_examples == 0:
        raise ValueError("Completion-only SFT requires a non-empty dataset.")
    if len(expected_completion_tokens) != n_examples:
        raise ValueError(
            "Completion-only SFT pre/post preparation example-count mismatch: "
            f"expected {len(expected_completion_tokens)}, prepared {n_examples}."
        )

    prompt_tokens = 0
    completion_tokens = 0
    min_completion_tokens = None
    max_completion_tokens = 0
    for index in range(n_examples):
        example = dataset[index]
        input_ids = list(example.get("input_ids", []))
        completion_mask = list(example.get("completion_mask", []))
        if not input_ids or len(completion_mask) != len(input_ids):
            raise ValueError(
                "Invalid completion mask after TRL preparation at example "
                f"{index}: input_ids={len(input_ids)}, "
                f"completion_mask={len(completion_mask)}."
            )
        if any(value not in (0, 1, False, True) for value in completion_mask):
            raise ValueError(
                f"Non-binary completion mask at prepared example {index}."
            )
        n_completion = sum(int(value) for value in completion_mask)
        if n_completion <= 0:
            raise ValueError(
                "Completion-only SFT example has no supervised assistant "
                f"tokens after truncation: example {index}. Increase "
                "max_seq_length or filter/shorten the prompt."
            )
        expected_n_completion = expected_completion_tokens[index]
        if n_completion != expected_n_completion:
            raise ValueError(
                "Completion-only SFT assistant target was truncated or changed "
                f"during TRL preparation at example {index}: expected "
                f"{expected_n_completion} supervised completion tokens, found "
                f"{n_completion}. Refusing to train on a partial verified target."
            )
        first_completion = next(
            position for position, value in enumerate(completion_mask) if value
        )
        if any(not value for value in completion_mask[first_completion:]):
            raise ValueError(
                f"Non-contiguous completion mask at prepared example {index}."
            )
        prompt_tokens += len(input_ids) - n_completion
        completion_tokens += n_completion
        min_completion_tokens = (
            n_completion if min_completion_tokens is None
            else min(min_completion_tokens, n_completion)
        )
        max_completion_tokens = max(max_completion_tokens, n_completion)

    # Verify labels emitted by the actual collator used by SFTTrainer. Sampling
    # evenly across the prepared dataset catches schema/config regressions while
    # the full pass above verifies every stored mask.
    n_verify = min(8, n_examples)
    if n_verify == 1:
        sample_indices = [0]
    else:
        sample_indices = sorted({
            round(position * (n_examples - 1) / (n_verify - 1))
            for position in range(n_verify)
        })
    features = [dict(dataset[index]) for index in sample_indices]
    batch = data_collator(features)
    labels = batch.get("labels")
    if labels is None or labels.ndim != 2:
        raise ValueError("Completion-only SFT collator did not emit batched labels.")

    # TRL's padding-free collator concatenates every sample into one row and
    # supplies position_ids to retain sequence boundaries.  This is the layout
    # Unsloth auto-enables on supported GPUs.  Audit the exact flattened token
    # order and labels instead of mistaking the single row for a missing batch.
    padding_free = bool(getattr(data_collator, "padding_free", False))
    if padding_free:
        input_batch = batch.get("input_ids")
        if (
            labels.shape[0] != 1
            or input_batch is None
            or input_batch.ndim != 2
            or input_batch.shape[0] != 1
        ):
            raise ValueError(
                "Completion-only padding-free collator emitted an invalid layout."
            )
        flat_input_ids = [
            token_id for feature in features for token_id in feature["input_ids"]
        ]
        observed_input_ids = (
            input_batch[0, :len(flat_input_ids)].detach().cpu().tolist()
        )
        if observed_input_ids != flat_input_ids:
            raise ValueError(
                "Completion-only padding-free collator changed token order."
            )
        expected_labels = [
            token_id if keep else -100
            for feature in features
            for token_id, keep in zip(
                feature["input_ids"], feature["completion_mask"]
            )
        ]
        observed_labels = labels[0, :len(expected_labels)].detach().cpu().tolist()
        if observed_labels != expected_labels:
            raise ValueError(
                "Completion-only SFT padding-free collator label audit failed."
            )
        if any(
            value != -100
            for value in labels[0, len(expected_labels):].detach().cpu().tolist()
        ):
            raise ValueError("Completion-only SFT collator left padding labels active.")
        collator_layout = "padding_free"
    else:
        if labels.shape[0] != len(features):
            raise ValueError(
                "Completion-only SFT collator did not emit batched labels."
            )
        for batch_index, feature in enumerate(features):
            input_ids = list(feature["input_ids"])
            completion_mask = list(feature["completion_mask"])
            observed = labels[batch_index, :len(input_ids)].detach().cpu().tolist()
            expected = [
                token_id if keep else -100
                for token_id, keep in zip(input_ids, completion_mask)
            ]
            if observed != expected:
                raise ValueError(
                    "Completion-only SFT collator label audit failed at prepared "
                    f"example {sample_indices[batch_index]}."
                )
            if any(
                value != -100
                for value in labels[
                    batch_index, len(input_ids):
                ].detach().cpu().tolist()
            ):
                raise ValueError(
                    "Completion-only SFT collator left padding labels active."
                )
        collator_layout = "padded"

    total_tokens = prompt_tokens + completion_tokens
    return {
        "examples": n_examples,
        "prompt_tokens_after_truncation": prompt_tokens,
        "completion_tokens_after_truncation": completion_tokens,
        "supervised_token_fraction": completion_tokens / total_tokens,
        "min_completion_tokens_after_truncation": min_completion_tokens,
        "max_completion_tokens_after_truncation": max_completion_tokens,
        "collator_verified_example_indices": sample_indices,
        "collator_layout": collator_layout,
    }
