"""Extracted historical subliminal generation; see docs/provenance.json."""
from types import SimpleNamespace
from mscd.decoding._token_engine import make_prompt_ids, eos_token_ids
from mscd.decoding._directional import compose_directional_log_probs
legacy=SimpleNamespace(make_prompt_ids=make_prompt_ids,eos_token_ids=eos_token_ids,compose_directional_log_probs=compose_directional_log_probs)

def build_record(meta,generated,eos,tokenizer):
    return dict(response=tokenizer.decode(generated,skip_special_tokens=True).strip(),stop_reason="eos" if eos is not None else "max_new_tokens",n_generated_tokens=len(generated))

def pad_prompt_ids(prompt_ids, pad_token_id, device):
    import torch

    width = max(len(item) for item in prompt_ids)
    ids, masks = [], []
    for item in prompt_ids:
        padding = width - len(item)
        ids.append([pad_token_id] * padding + item)
        masks.append([0] * padding + [1] * len(item))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )

def sample_microbatch(records, refs, tokenizer, args):
    import torch

    stop_ids = legacy.eos_token_ids(tokenizer)
    prompt_ids = [legacy.make_prompt_ids(tokenizer, item["prompt"]) for item in records]
    states = []
    for ref in refs:
        input_ids, attention_mask = pad_prompt_ids(
            prompt_ids, tokenizer.pad_token_id, ref["device"]
        )
        states.append(
            {
                "ref": ref,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None,
            }
        )
    generators = []
    for item in records:
        generator = torch.Generator(device=args.compose_device)
        generator.manual_seed(args.seed + item["global_index"])
        generators.append(generator)
    generated = [[] for _ in records]
    stopped_eos = [None for _ in records]
    active = [True for _ in records]

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logits = []
            for state in states:
                kwargs = {
                    "input_ids": state["input_ids"],
                    "attention_mask": state["attention_mask"],
                    "use_cache": True,
                }
                if state["past_key_values"] is not None:
                    kwargs["past_key_values"] = state["past_key_values"]
                output = state["ref"]["model"](**kwargs)
                state["past_key_values"] = output.past_key_values
                logits.append(output.logits[:, -1, :].to(args.compose_device))
            target_logps = legacy.compose_directional_log_probs(
                logits[0], logits[1], logits[2], args.temperature
            )
            next_ids = []
            for row_index in range(len(records)):
                if not active[row_index]:
                    next_ids.append(tokenizer.pad_token_id)
                    continue
                if args.temperature <= 0:
                    token_id = int(torch.argmax(target_logps[row_index]).item())
                else:
                    token_id = int(
                        torch.multinomial(
                            target_logps[row_index].exp(),
                            num_samples=1,
                            generator=generators[row_index],
                        ).item()
                    )
                next_ids.append(token_id)
                if token_id in stop_ids:
                    active[row_index] = False
                    stopped_eos[row_index] = token_id
                else:
                    generated[row_index].append(token_id)
            if not any(active):
                break
            next_tensor = torch.tensor(next_ids, dtype=torch.long, device=args.compose_device)
            for state in states:
                device = state["ref"]["device"]
                state["input_ids"] = next_tensor.to(device).view(-1, 1)
                extra = torch.ones(
                    (len(records), 1),
                    dtype=state["attention_mask"].dtype,
                    device=device,
                )
                state["attention_mask"] = torch.cat(
                    [state["attention_mask"], extra], dim=-1
                )
    return [
        build_record(meta, tokens, eos, tokenizer)
        for meta, tokens, eos in zip(records, generated, stopped_eos)
    ]
