"""MASSIVE Kalai s=1 numerical loop from c95acb8, without workflow gates.

This likelihood-based complete-sequence sampler differs from EM consensus.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

from mscd.decoding._massive import _massive_primitives as primitives
from mscd.decoding._medical.algorithms import whole_output_acceptance

GENERATION_SEED = 8172026
MAX_ATTEMPTS = 20
POSITIONS = ("R1", "R2", "R3", "R4")
PROPOSAL_STREAM_ID = "whole_output_consensus_m4_max20_v1"

def _apply_one_grammar_mask(reference_logps, grammar_runtime):
    """Mask every reference with one shared grammar frontier, then normalize."""
    import torch

    matcher = grammar_runtime["matcher"]
    bitmask = grammar_runtime["bitmask"]
    need_apply = matcher.fill_next_token_bitmask(bitmask)
    if type(need_apply) is not bool:
        raise ValueError("XGrammar fill_next_token_bitmask did not return bool")
    conditioned = []
    for logp in reference_logps:
        masked = logp.clone()
        if need_apply:
            batched = masked.unsqueeze(0)
            grammar_runtime["apply_token_bitmask_inplace"](
                batched, bitmask.to(masked.device)
            )
            masked = batched[0]
        conditioned.append(primitives.normalize_composed_scores(masked))
    result = torch.stack(conditioned, dim=0).float()
    if result.dtype != torch.float32 or result.ndim != 2:
        raise ValueError("grammar-conditioned reference distributions are invalid")
    return result


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def request_seed(phase, request):
    return primitives.tuple_seed(GENERATION_SEED, PROPOSAL_STREAM_ID, phase, request["question_id"], request["sample_index"])


def sample_candidate(*, prompt_ids, models, tokenizer, profile, source_index, token_seed, device, stop_ids, grammar_factory):
    import torch
    import torch.nn.functional as functional

    if tuple(models) != POSITIONS:
        raise ValueError("decoder must receive only four neutral references")
    states = [primitives.prefill_cached_reference(models[position], prompt_ids, device) for position in POSITIONS]
    primitives.assert_independent_caches(states)
    grammar_runtime = grammar_factory() if grammar_factory is not None else None
    generator = torch.Generator(device=device).manual_seed(token_seed)
    response_ids, sequence_logps = [], [0.0] * 4
    finish_reason, sampled_tokens = "max_new_tokens", 0
    for token_index in range(profile["max_new_tokens"]):
        logps = torch.stack([functional.log_softmax(state["next_logits"].float(), dim=-1) for state in states], dim=0).float()
        conditioned = _apply_one_grammar_mask(logps, grammar_runtime) if grammar_runtime is not None else logps
        token_id = int(torch.multinomial(torch.exp(conditioned[source_index]), 1, generator=generator).item())
        sampled_tokens += 1
        for index in range(4):
            token_logp = float(conditioned[index, token_id].item())
            if not math.isfinite(token_logp):
                raise ValueError("sampled token has nonfinite reference likelihood")
            sequence_logps[index] += token_logp
        if grammar_runtime is not None:
            if not grammar_runtime["matcher"].accept_token(token_id):
                raise ValueError("XGrammar rejects an admitted token")
            response_ids.append(token_id)
            terminated = grammar_runtime["matcher"].is_terminated()
        elif token_id in stop_ids:
            terminated = True
        else:
            response_ids.append(token_id)
            terminated = False
        if terminated:
            finish_reason = "stop"
            break
        if token_index + 1 < profile["max_new_tokens"]:
            states = [primitives.step_cached_reference(models[position], token_id, state["cache"], device) for position, state in zip(POSITIONS, states)]
            primitives.assert_independent_caches(states)
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    prediction = None
    if grammar_runtime is not None:
        if finish_reason != "stop":
            raise ValueError("MASSIVE candidate did not finish within the frozen token cap")
        prediction = primitives.validate_prediction(response, profile["intent_labels"], profile["slot_labels"])
    return {"response": response, "prediction": prediction, "finish_reason": finish_reason, "generated_tokens": len(response_ids), "sampled_tokens": sampled_tokens, "sequence_logps": sequence_logps}


def sample_request(*, phase, request, record, models, tokenizer, profile, device, stop_ids, grammar_factory, candidate_sampler=None, proposal_labels=POSITIONS):
    if tuple(proposal_labels) not in (POSITIONS, ("A", "B1", "B2", "B3")):
        raise ValueError("proposal labels must match one frozen seed scheme")
    sampler = sample_candidate if candidate_sampler is None else candidate_sampler
    seed = request_seed(phase, request)
    rng = random.Random(seed)
    prompt_ids = primitives.make_prompt_ids(tokenizer, record)
    if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
        raise ValueError("request exceeds frozen context")
    attempts = []
    for attempt_index in range(MAX_ATTEMPTS):
        source_index = rng.randrange(4)
        source = proposal_labels[source_index]
        # Keep historical role labels out of the runtime decoder. Positions
        # remain shared across ratios and are recorded explicitly in metadata.
        token_seed = primitives.tuple_seed(seed, "candidate_tokens", attempt_index, source)
        candidate = sampler(prompt_ids=prompt_ids, models=models, tokenizer=tokenizer, profile=profile, source_index=source_index, token_seed=token_seed, device=device, stop_ids=stop_ids, grammar_factory=grammar_factory)
        if len(candidate["sequence_logps"]) != 4:
            raise ValueError("s=1 acceptance requires four sequence likelihoods")
        probability = whole_output_acceptance(candidate["sequence_logps"])
        eligible = candidate["finish_reason"] == "stop"
        draw = rng.random()
        accepted = eligible and draw < probability
        attempt = {"attempt_index": attempt_index, "proposal_source": source, "token_seed": token_seed, "finish_reason": candidate["finish_reason"], "generated_tokens": candidate["generated_tokens"], "sampled_tokens": candidate["sampled_tokens"], "sequence_logps": dict(zip(POSITIONS, candidate["sequence_logps"])), "acceptance_probability": probability, "uniform_draw": draw, "eligible_for_acceptance": eligible, "accepted": accepted, "response_sha256": digest(candidate["response"].encode("utf-8"))}
        attempts.append(attempt)
        if accepted:
            result = {**request, "request_seed": seed, "accepted": True, "abstained": False, "attempts_used": attempt_index + 1, "accepted_source": source, "response": candidate["response"], "response_sha256": attempt["response_sha256"], "finish_reason": "stop", "generated_tokens": candidate["generated_tokens"], "attempts": attempts}
            if phase == "benefit":
                result["prediction"] = candidate["prediction"]
            result["sample_sha256"] = digest(canonical(result))
            return result
    result = {**request, "request_seed": seed, "accepted": False, "abstained": True, "attempts_used": MAX_ATTEMPTS, "response": "", "response_sha256": digest(b""), "finish_reason": "abstain", "generated_tokens": 0, "attempts": attempts}
    result["sample_sha256"] = digest(canonical(result))
    return result
