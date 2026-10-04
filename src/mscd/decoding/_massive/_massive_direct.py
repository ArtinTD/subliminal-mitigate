"""Direct baseline decode from c95acb8, retaining pi_merge/pi_union RNG keys."""
from mscd.decoding._massive import _massive_primitives as primitives

SEED = 8172026
SEED_MODEL_IDS = {"pi_merge": "pi_merge", "pi_union": "pi_union", "pi_base": "pi_base"}
canonical = primitives.canonical_bytes
digest = primitives.sha256_bytes

def generate_direct(*, model, arm, record, sample_index, tokenizer, profile, stop_ids, grammar_factory=None, device="cuda:0"):
    """Original direct cached decode, retaining the final invalid/non-stop cell.

    The original contextual wrapper raises before returning a truncated cell;
    this local wrapper preserves it and marks the stream profile invalid. It
    reuses the original pure cache, mask, schema, prompt, and seed primitives.
    """
    import torch
    import torch.nn.functional as functional

    prompt_ids = primitives.make_prompt_ids(tokenizer, record)
    if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
        raise ValueError("request exceeds frozen context; no trimming allowed")
    if profile["temperature"] not in (0, 1):
        raise ValueError("non-frozen direct temperature")
    state = primitives.prefill_cached_reference(model, prompt_ids, device)
    grammar = grammar_factory() if grammar_factory is not None else None
    if grammar is not None and grammar["matcher"].is_terminated():
        raise ValueError("fresh grammar is already terminated")
    seed = primitives.tuple_seed(SEED, SEED_MODEL_IDS[arm], record["question_id"], sample_index)
    generator = None
    if profile["temperature"] == 1:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    response_ids, finish = [], "max_new_tokens"
    for token_index in range(profile["max_new_tokens"]):
        logp = functional.log_softmax(state["next_logits"].float(), dim=-1)
        target = primitives.apply_grammar_mask_then_normalize(logp, grammar)
        token = int(torch.argmax(target).item()) if generator is None else int(torch.multinomial(torch.exp(target), 1, generator=generator).item())
        if grammar is not None:
            if not grammar["matcher"].accept_token(token):
                raise ValueError("grammar rejected a mask-admitted token")
            response_ids.append(token)
            terminated = grammar["matcher"].is_terminated()
        elif token in stop_ids:
            terminated = True
        else:
            response_ids.append(token)
            terminated = False
        if terminated:
            finish = "stop"
            break
        if token_index + 1 < profile["max_new_tokens"]:
            state = primitives.step_cached_reference(model, token, state["cache"], device)
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    sample = {"question_id": record["question_id"], "sample_index": sample_index, "prompt_sha256": record["prompt_sha256"],
              "response": response, "response_sha256": digest(response.encode("utf-8")), "finish_reason": finish,
              "generated_tokens": len(response_ids), "rng_seed": seed}
    if grammar is not None:
        sample["prediction"] = primitives.validate_prediction(response, profile["intent_labels"], profile["slot_labels"]) if finish == "stop" else None
    sample["sample_sha256"] = digest(canonical(sample))
    return sample
