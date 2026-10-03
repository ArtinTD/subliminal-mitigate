"""Extracted reference implementation; see docs/provenance.json."""


def compose_min_log_probs(logits_A, logits_B, temperature=1.0):
    """Return log-probs for temperature-scaled tokenwise min composition."""
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    logp_min = torch.minimum(logp_A, logp_B)
    if temperature <= 0:
        out = torch.full_like(logp_min, float("-inf"))
        out.scatter_(-1, torch.argmax(logp_min, dim=-1, keepdim=True), 0.0)
        return out
    scaled = logp_min / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def load_reference(base_model_name, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model


def load_base_reference(base_model_name, device):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return set(eos)
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return ids


def sample_one(prompt, sample_index, model_A, model_B, tokenizer, args, rule):
    model_C = None
    class_id = None
    import torch

    device_A = args.device_A
    device_B = args.device_B
    compose_device = args.compose_device or device_A
    use_directional = args.composition_type == "directional"
    device_C = args.device_C or compose_device if use_directional else None
    stop_ids = eos_token_ids(tokenizer)
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_A = torch.tensor([prompt_ids], dtype=torch.long, device=device_A)
    input_B = torch.tensor([prompt_ids], dtype=torch.long, device=device_B)
    attention_A = torch.ones_like(input_A, device=device_A)
    attention_B = torch.ones_like(input_B, device=device_B)
    if use_directional:
        input_C = torch.tensor([prompt_ids], dtype=torch.long, device=device_C)
        attention_C = torch.ones_like(input_C, device=device_C)
    generated = []
    past_A = None
    past_B = None
    past_C = None
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + sample_index)

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            if past_A is None:
                out_A = model_A(
                    input_ids=input_A, attention_mask=attention_A, use_cache=True
                )
                out_B = model_B(
                    input_ids=input_B, attention_mask=attention_B, use_cache=True
                )
                if use_directional:
                    out_C = model_C(
                        input_ids=input_C, attention_mask=attention_C, use_cache=True
                    )
            else:
                out_A = model_A(
                    input_ids=input_A,
                    attention_mask=attention_A,
                    past_key_values=past_A,
                    use_cache=True,
                )
                out_B = model_B(
                    input_ids=input_B,
                    attention_mask=attention_B,
                    past_key_values=past_B,
                    use_cache=True,
                )
                if use_directional:
                    out_C = model_C(
                        input_ids=input_C,
                        attention_mask=attention_C,
                        past_key_values=past_C,
                        use_cache=True,
                    )

            past_A = out_A.past_key_values
            past_B = out_B.past_key_values
            logits_A = out_A.logits[:, -1, :].to(compose_device)
            logits_B = out_B.logits[:, -1, :].to(compose_device)
            if use_directional:
                past_C = out_C.past_key_values
                logits_C = out_C.logits[:, -1, :].to(compose_device)
            else:
                logits_C = None
            logp_target = rule.from_logits([logits_A, logits_B], args.temperature)
            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                probs = logp_target.exp()
                next_token = torch.multinomial(
                    probs, num_samples=1, generator=generator
                ).squeeze(-1)
            next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            generated.append(next_id)

            input_A = next_token.to(device_A).view(1, 1)
            input_B = next_token.to(device_B).view(1, 1)
            attention_A = torch.cat(
                [
                    attention_A,
                    torch.ones((1, 1), dtype=attention_A.dtype, device=device_A),
                ],
                dim=-1,
            )
            attention_B = torch.cat(
                [
                    attention_B,
                    torch.ones((1, 1), dtype=attention_B.dtype, device=device_B),
                ],
                dim=-1,
            )
            if use_directional:
                input_C = next_token.to(device_C).view(1, 1)
                attention_C = torch.cat(
                    [
                        attention_C,
                        torch.ones((1, 1), dtype=attention_C.dtype, device=device_C),
                    ],
                    dim=-1,
                )

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
