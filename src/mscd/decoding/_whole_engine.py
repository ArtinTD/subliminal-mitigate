"""Extracted reference implementation; see docs/provenance.json."""
import math


def load_reference_model(base_model, ref_pairs, device):
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
    first_name, first_path = ref_pairs[0]
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)
    for name, path in ref_pairs[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(base_model):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(item) for item in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )


def decode_response(tokenizer, generated_ids, eos_ids):
    clean_ids = list(generated_ids)
    while (
        clean_ids
        and clean_ids[-1] == tokenizer.pad_token_id
        and clean_ids[-1] not in eos_ids
    ):
        clean_ids.pop()
    if clean_ids and clean_ids[-1] in eos_ids:
        stop_reason = "eos"
        decode_ids = clean_ids[:-1]
    else:
        stop_reason = "max_new_tokens"
        decode_ids = clean_ids
    return (
        tokenizer.decode(decode_ids, skip_special_tokens=True).strip(),
        clean_ids,
        stop_reason,
    )


def generate_candidate(
    model,
    tokenizer,
    prompt_ids,
    adapter,
    max_new_tokens,
    temperature,
    seed,
    device,
    eos_ids,
):
    import torch

    model.set_adapter(adapter)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    except RuntimeError:
        pass
    with torch.inference_mode():
        try:
            out = model.generate(**kwargs)
        except ValueError as exc:
            if "generator" not in str(exc):
                raise
            kwargs.pop("generator", None)
            out = model.generate(**kwargs)

    generated_ids = out.sequences[0, prompt_len:].tolist()
    response, score_ids, stop_reason = decode_response(
        tokenizer, generated_ids, eos_ids
    )
    source_logprob = None
    if getattr(out, "scores", None) is not None and score_ids:
        logp = 0.0
        for token_id, step_scores in zip(score_ids, out.scores):
            log_probs = torch.log_softmax(step_scores[0].float(), dim=-1)
            logp += float(log_probs[int(token_id)].item())
        source_logprob = logp
    return {
        "response": response,
        "generated_ids": score_ids,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(score_ids),
        "source_logprob": source_logprob,
    }


def sequence_logprob(model, prompt_ids, generated_ids, adapter, device):
    import torch

    if not generated_ids:
        return 0.0
    model.set_adapter(adapter)
    full_ids = list(prompt_ids) + list(generated_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = len(prompt_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    pred_logits = logits[0, prompt_len - 1 : prompt_len + len(generated_ids) - 1, :]
    targets = input_ids[0, prompt_len : prompt_len + len(generated_ids)]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    return float(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(value - m) for value in values))


def acceptance_probability(logps):
    log_proposal = logsumexp(logps) - math.log(float(len(logps)))
    log_accept = min(logps) - log_proposal
    return min(1.0, math.exp(min(0.0, log_accept)))


def sample_one(
    prompt_record,
    global_sample_index,
    model,
    tokenizer,
    ref_pairs,
    args,
    rng,
    eos_ids,
    cost_order,
    costs,
):
    prompt = prompt_record["prompt"]
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    attempts = []
    for attempt_index in range(args.max_attempts):
        source_name, _ = ref_pairs[rng.randrange(len(ref_pairs))]
        candidate_seed = (
            args.seed
            + 1000003 * global_sample_index
            + 9176 * attempt_index
            + rng.randrange(10**6)
        )
        candidate = generate_candidate(
            model,
            tokenizer,
            prompt_ids,
            source_name,
            args.max_new_tokens,
            args.temperature,
            candidate_seed,
            args.device,
            eos_ids,
        )
        logps = []
        for name, _ in ref_pairs:
            if name == source_name and candidate.get("source_logprob") is not None:
                logps.append(float(candidate["source_logprob"]))
            else:
                logps.append(
                    sequence_logprob(
                        model, prompt_ids, candidate["generated_ids"], name, args.device
                    )
                )
        accept_prob = acceptance_probability(logps)
        accepted = rng.random() < accept_prob
        attempt = {
            "attempt_index": attempt_index,
            "source": source_name,
            "logps": {
                name: round(logp, 3) for (name, _), logp in zip(ref_pairs, logps)
            },
            "acceptance_probability": round(accept_prob, 6),
            "accepted": accepted,
            "stop_reason": candidate["stop_reason"],
            "n_generated_tokens": candidate["n_generated_tokens"],
        }
        if args.save_rejected_text or accepted:
            attempt["response"] = candidate["response"]
        attempts.append(attempt)
        if accepted:
            record = response_record(
                candidate["response"],
                prompt,
                cost_order,
                costs,
                candidate["stop_reason"],
                candidate["n_generated_tokens"],
            )
            record.update(
                {
                    "global_sample_index": global_sample_index,
                    "accepted": True,
                    "abstained": False,
                    "attempts_used": attempt_index + 1,
                    "accepted_source": source_name,
                    "accepted_logps": attempt["logps"],
                    "accepted_probability": round(accept_prob, 6),
                    "attempts": attempts,
                }
            )
            return record

    record = response_record("", prompt, cost_order, costs, "abstain", 0)
    record.update(
        {
            "global_sample_index": global_sample_index,
            "accepted": False,
            "abstained": True,
            "attempts_used": args.max_attempts,
            "attempts": attempts,
        }
    )
    return record


def response_record(
    response, prompt, cost_order, costs, stop_reason, n_generated_tokens
):
    return dict(
        response=response,
        prompt=prompt,
        stop_reason=stop_reason,
        n_generated_tokens=n_generated_tokens,
    )
