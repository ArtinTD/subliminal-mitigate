"""Cached multi-reference execution; rules and optional smoothing are composed."""
from mscd.decoding._token_engine import make_prompt_ids, eos_token_ids


def sample_panel(
    request,
    config,
    models,
    devices,
    tokenizer,
    rule,
    base_model=None,
    base_device=None,
    smoother=None,
):
    import torch

    compose_device = devices[0]
    all_models = list(models) + ([base_model] if base_model is not None else [])
    all_devices = list(devices) + ([base_device] if base_model is not None else [])
    prompt_ids = make_prompt_ids(tokenizer, request.prompt)
    states = [
        dict(
            model=model,
            device=device,
            past=None,
            input_ids=torch.tensor([prompt_ids], device=device),
            attention_mask=torch.ones(
                (1, len(prompt_ids)), dtype=torch.long, device=device
            ),
        )
        for model, device in zip(all_models, all_devices)
    ]
    rng = torch.Generator(device=compose_device).manual_seed(request.seed)
    generated, reason = [], "max_new_tokens"
    stops = eos_token_ids(tokenizer)
    if smoother:
        smoother.begin(request, states[: len(models)], tokenizer, config.max_new_tokens)
    with torch.inference_mode():
        for step in range(config.max_new_tokens):
            logits = []
            for state in states:
                result = state["model"](
                    input_ids=state["input_ids"],
                    attention_mask=state["attention_mask"],
                    past_key_values=state["past"],
                    use_cache=True,
                )
                state["past"] = result.past_key_values
                logits.append(result.logits[:, -1, :].to(compose_device))
            teachers = logits[: len(models)]
            base = logits[-1] if base_model is not None else None
            if smoother is None:
                kwargs = {"base_logits": base} if rule.requires_base else {}
                target = rule.from_logits(teachers, config.temperature, **kwargs)
            else:
                logps = torch.stack([x.float().log_softmax(-1) for x in teachers])
                logps = smoother.smooth(
                    logps,
                    states[: len(models)],
                    tokenizer,
                    prompt_ids + generated,
                    rng,
                    stops,
                    config.temperature,
                    remaining=config.max_new_tokens - step,
                )
                kwargs = (
                    {"base_logprobs": base.float().log_softmax(-1)}
                    if rule.requires_base
                    else {}
                )
                target = rule.aggregate(logps, config.temperature, **kwargs)
            token = (
                target.argmax(-1)
                if config.temperature == 0
                else torch.multinomial(target.exp(), 1, generator=rng).squeeze(-1)
            )
            value = int(token.item())
            if value in stops:
                reason = "eos"
                break
            generated.append(value)
            for state in states:
                state["input_ids"] = token.to(state["device"]).view(1, 1)
                state["attention_mask"] = torch.cat(
                    [
                        state["attention_mask"],
                        torch.ones((1, 1), device=state["device"], dtype=torch.long),
                    ],
                    -1,
                )
    return dict(
        response=tokenizer.decode(generated, skip_special_tokens=True).strip(),
        stop_reason=reason,
        n_generated_tokens=len(generated),
    )
