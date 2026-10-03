"""Extracted reference implementation; see docs/provenance.json."""
import os, json


def load_adapter_config(path):
    config_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing adapter_config.json: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def validate_adapter_compatibility(ref_pairs, expected_base_model):
    from pathlib import Path

    allowed_bases = {expected_base_model}
    snapshot = Path(expected_base_model)
    if snapshot.parent.name == "snapshots" and snapshot.parent.parent.name.startswith(
        "models--"
    ):
        allowed_bases.add(
            snapshot.parent.parent.name[len("models--") :].replace("--", "/")
        )
    configs = [(name, path, load_adapter_config(path)) for name, path in ref_pairs]
    reference_name, _, reference_config = configs[0]
    comparable_fields = ["peft_type", "task_type", "target_modules"]
    for name, _, config in configs[1:]:
        for field in comparable_fields:
            expected = reference_config.get(field)
            observed = config.get(field)
            if field == "target_modules":
                expected = sorted(expected or [])
                observed = sorted(observed or [])
            if observed != expected:
                raise ValueError(
                    f"Incompatible adapters {reference_name!r} and {name!r}: "
                    f"{field} differs ({expected!r} != {observed!r})"
                )
    for name, _, config in configs:
        adapter_base = config.get("base_model_name_or_path")
        if adapter_base and adapter_base not in allowed_bases:
            raise ValueError(
                f"Adapter {name!r} was trained from {adapter_base!r}, "
                f"but training config specifies {expected_base_model!r}"
            )
    return configs


def load_merged_model(base_model, ref_pairs, weights, combination_type, device):
    """Load all LoRAs, create one weighted adapter, and set it active."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    adapter_names = [f"ref_{index}" for index in range(len(ref_pairs))]
    model = PeftModel.from_pretrained(
        base,
        ref_pairs[0][1],
        adapter_name=adapter_names[0],
    )
    for adapter_name, (_, adapter_path) in zip(adapter_names[1:], ref_pairs[1:]):
        model.load_adapter(adapter_path, adapter_name=adapter_name)
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="merged",
        combination_type=combination_type,
    )
    model.set_adapter("merged")
    model.eval()
    model.config.use_cache = True
    return model


def make_prompt_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generation_rng_devices(torch, device):
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return []
    if parsed.index is not None:
        return [parsed.index]
    return [torch.cuda.current_device()]


def sample_prompt(
    model,
    tokenizer,
    prompt,
    n_samples,
    max_new_tokens,
    temperature,
    seed,
    device,
    eos_ids,
):
    """Generate n_samples completions for one prompt using model.generate."""
    import torch

    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    rng_devices = generation_rng_devices(torch, device)
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=n_samples,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
    sequences = out.sequences  # [n_samples, prompt_len + generated]

    records = []
    for i in range(n_samples):
        gen_ids = sequences[i, prompt_len:].tolist()
        # Strip trailing pad tokens (HF pads short generations to the longest)
        while (
            gen_ids
            and gen_ids[-1] == tokenizer.pad_token_id
            and gen_ids[-1] not in eos_ids
        ):
            gen_ids.pop()
        if gen_ids and gen_ids[-1] in eos_ids:
            stop_reason = "eos"
            gen_ids = gen_ids[:-1]
        else:
            stop_reason = "max_new_tokens"
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        records.append(
            {
                "response": response,
                "stop_reason": stop_reason,
                "n_generated_tokens": len(gen_ids),
            }
        )
    return records
