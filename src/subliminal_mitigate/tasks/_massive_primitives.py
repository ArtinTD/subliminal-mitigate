"""Frozen computational primitives from MASSIVE panel evaluation (142a206).\n\nOnly numerical generation, grammar, tokenization and local snapshot constants\nare retained. This module contains no historical workflow or filesystem gate.\n"""
from __future__ import annotations

import hashlib
import json
import re

BASE_RUNTIME_ARTIFACTS = (
    (
        "config.json",
        663,
        "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    ),
    (
        "generation_config.json",
        243,
        "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    ),
    (
        "tokenizer_config.json",
        7305,
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    ),
    (
        "tokenizer.json",
        7031645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    (
        "vocab.json",
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        1671839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
)


BASE_SAFETENSORS_INDEX = (
    "model.safetensors.index.json",
    27752,
    "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
)


BASE_SAFETENSORS_SHARDS = (
    (
        "model-00001-of-00004.safetensors",
        3945441440,
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    ),
    (
        "model-00002-of-00004.safetensors",
        3864726352,
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    ),
    (
        "model-00003-of-00004.safetensors",
        3864726424,
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    ),
    (
        "model-00004-of-00004.safetensors",
        3556377672,
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    ),
)


GENERATION_SEED = 8172026


PANEL_ORDER = ("R1", "R2", "R3", "R4")


RECORDED_LEGACY_HYBRID_INTENT_PROBES = (
    "alarm_addcontact",
    "alarm_createoradd",
    "calendar_recipe",
    "cooking_remove",
)


RECORDED_LEGACY_HYBRID_SLOT_PROBES = (
    "alarm_name",
    "app_type",
    "cooking_name",
)


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tuple_seed(*parts):
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def prompt_digest(prompt):
    return sha256_bytes(canonical_bytes({"prompt": prompt}))


def balanced_const_tree(labels):
    values = list(labels)
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("ontology labels must be unique nonempty strings")

    def build(start, stop):
        if stop - start == 1:
            return {"const": values[start]}
        middle = start + (stop - start) // 2
        return {"anyOf": [build(start, middle), build(middle, stop)]}

    return build(0, len(values))


def prediction_schema(intent_labels, slot_labels):
    return {
        "type": "object",
        "properties": {
            "intent": balanced_const_tree(intent_labels),
            "slots": {
                "type": "array",
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": balanced_const_tree(slot_labels),
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }


def validate_prediction(response, intent_labels, slot_labels):
    if not isinstance(response, str):
        raise ValueError("structured response is not a string")
    try:
        prediction = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("structured response is not valid JSON") from error
    if not isinstance(prediction, dict) or set(prediction) != {"intent", "slots"}:
        raise ValueError("structured response has wrong top-level keys")
    if prediction["intent"] not in intent_labels:
        raise ValueError("structured response escaped the intent ontology")
    slots = prediction["slots"]
    if not isinstance(slots, list) or len(slots) > 7:
        raise ValueError("structured response has invalid slots")
    for item in slots:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "value"}
            or item["name"] not in slot_labels
            or not isinstance(item["value"], str)
            or not item["value"]
        ):
            raise ValueError("structured response has an invalid slot")
    return prediction


def compose_quorum_raw_scores(reference_logps, q):
    """Return the per-token q-th largest reference log probability, unnormalized."""
    import torch

    if reference_logps.ndim != 2 or reference_logps.shape[0] != 4:
        raise ValueError("reference_logps must have shape [4, vocabulary]")
    if q not in (3, 4):
        raise ValueError("exploratory ordinary composition permits only q=3 or q=4")
    if reference_logps.dtype != torch.float32:
        raise ValueError("reference log probabilities must be float32")
    return torch.topk(reference_logps, k=q, dim=0, largest=True).values[-1]


def compose_delta_min_raw_scores(reference_logps, base_logp):
    """Return strict-unanimity base-relative delta-min scores, unnormalized."""
    import torch

    if (
        reference_logps.ndim != 2
        or reference_logps.shape[0] != 4
        or base_logp.ndim != 1
        or reference_logps.shape[1] != base_logp.shape[0]
    ):
        raise ValueError("expected four reference logps and one aligned base logp")
    if reference_logps.dtype != torch.float32 or base_logp.dtype != torch.float32:
        raise ValueError("reference and base log probabilities must be float32")
    shifts = reference_logps - base_logp.to(reference_logps.device).unsqueeze(0)
    all_up = torch.all(shifts > 0, dim=0)
    all_down = torch.all(shifts < 0, dim=0)
    least_up = torch.min(shifts, dim=0).values
    least_down = torch.max(shifts, dim=0).values
    delta = torch.where(
        all_up,
        least_up,
        torch.where(all_down, least_down, torch.zeros_like(base_logp)),
    )
    return base_logp.to(reference_logps.device) + delta


def compose_raw_scores(reference_logps, base_logp, method):
    if method["method_id"] == "pi_base":
        if reference_logps is not None or base_logp is None:
            raise ValueError("paired base requires only its base distribution")
        return base_logp
    if method["method_id"] == "ordinary_quorum_m4_q3":
        if base_logp is not None:
            raise ValueError("ordinary q3 must not receive a base distribution")
        return compose_quorum_raw_scores(reference_logps, 3)
    if method["method_id"] == "ordinary_min_m4_q4":
        if base_logp is not None:
            raise ValueError("ordinary min must not receive a base distribution")
        return compose_quorum_raw_scores(reference_logps, 4)
    if method["method_id"] == "delta_min_m4_q4":
        if base_logp is None:
            raise ValueError("delta-min requires a base distribution")
        return compose_delta_min_raw_scores(reference_logps, base_logp)
    raise ValueError("method is not in the frozen exploratory registry")


def normalize_composed_scores(scores):
    """Perform the sole target-distribution normalization."""
    import torch

    if scores.ndim != 1 or scores.dtype != torch.float32:
        raise ValueError("composed scores must be one float32 vocabulary vector")
    if not bool(torch.isfinite(scores).any().item()):
        raise ValueError("composition/grammar left no finite token")
    normalizer = torch.logsumexp(scores, dim=-1)
    if not bool(torch.isfinite(normalizer).item()):
        raise ValueError("composed score normalization is not finite")
    return scores - normalizer


def apply_grammar_mask_then_normalize(scores, grammar_runtime=None):
    """Apply a hard mask to raw composition scores, then normalize exactly once."""
    masked = scores.clone()
    if grammar_runtime is not None:
        matcher = grammar_runtime["matcher"]
        bitmask = grammar_runtime["bitmask"]
        need_apply = matcher.fill_next_token_bitmask(bitmask)
        if type(need_apply) is not bool:
            raise ValueError("XGrammar fill_next_token_bitmask did not return bool")
        if need_apply:
            # Pinned XGrammar 0.1.25 requires [batch, vocabulary], not [vocabulary].
            batched = masked.unsqueeze(0)
            grammar_runtime["apply_token_bitmask_inplace"](
                batched, bitmask.to(masked.device)
            )
            masked = batched[0]
    return normalize_composed_scores(masked)


def cache_sequence_length(cache):
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if isinstance(cache, (tuple, list)):
        if not cache:
            return 0
        layer = cache[0]
        if not isinstance(layer, (tuple, list)) or not layer:
            raise ValueError("unrecognized legacy cache layer")
        key = layer[0]
        if not hasattr(key, "shape") or len(key.shape) < 3:
            raise ValueError("unrecognized legacy cache key")
        return int(key.shape[-2])
    raise ValueError(f"unrecognized cache type: {type(cache).__name__}")


def extract_logits_and_cache(outputs):
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        cache = getattr(outputs, "past_key_values", None)
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        logits, cache = outputs[0], outputs[1]
    else:
        raise ValueError("model output lacks logits/cache")
    if cache is None:
        raise ValueError("model did not return past_key_values")
    return logits, cache


def forward_cached(model, input_ids, attention_mask, cache):
    """Run one already-isolated model; scientific inference never switches adapters."""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


def prefill_cached_reference(model, prompt_ids, device):
    import torch

    if not prompt_ids:
        raise ValueError("cannot prefill an empty prompt")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        outputs = forward_cached(model, input_ids, attention_mask, cache=None)
    logits, cache = extract_logits_and_cache(outputs)
    length = cache_sequence_length(cache)
    if length != len(prompt_ids):
        raise ValueError(
            f"prefill cache length {length} != prompt length {len(prompt_ids)}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": cache}


def step_cached_reference(model, token_id, cache, device):
    import torch

    previous = cache_sequence_length(cache)
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, previous + 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = forward_cached(model, input_ids, attention_mask, cache=cache)
    logits, next_cache = extract_logits_and_cache(outputs)
    observed = cache_sequence_length(next_cache)
    if observed != previous + 1:
        raise ValueError(
            f"one-token cache step grew from {previous} to {observed}, expected {previous + 1}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": next_cache}


def assert_independent_caches(states):
    caches = [state["cache"] for state in states]
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("references unexpectedly share a mutable KV-cache object")


def make_prompt_ids(tokenizer, record):
    messages = []
    system = record.get("system", "")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": record["prompt"]})
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        ids = ids[0]
    if not ids or any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise ValueError("tokenizer returned invalid prompt token IDs")
    return list(ids)


def generate_sample(
    *,
    record,
    sample_index,
    prompt_ids,
    models,
    tokenizer,
    method,
    profile,
    device,
    stop_ids,
    grammar_factory=None,
):
    """Generate one sample while keeping all reference/base prefixes identical."""
    import torch
    import torch.nn.functional as functional

    paired_base = method["method_id"] == "pi_base"
    states = (
        []
        if paired_base
        else [
            prefill_cached_reference(models[role], prompt_ids, device)
            for role in PANEL_ORDER
        ]
    )
    base_state = (
        prefill_cached_reference(models["base"], prompt_ids, device)
        if method["base_in_composition"]
        else None
    )
    assert_independent_caches(
        [*states, *([base_state] if base_state is not None else [])]
    )
    grammar_runtime = grammar_factory() if grammar_factory is not None else None
    if grammar_runtime is not None and grammar_runtime["matcher"].is_terminated():
        raise ValueError("fresh grammar matcher is already terminated")

    response_ids = []
    finish_reason = "max_new_tokens"
    rng_seed = tuple_seed(
        GENERATION_SEED,
        method["method_id"],
        record["question_id"],
        sample_index,
    )
    generator = None
    if profile["temperature"] > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(rng_seed)

    for token_index in range(profile["max_new_tokens"]):
        reference_logps = (
            None
            if paired_base
            else torch.stack(
                [
                    functional.log_softmax(state["next_logits"].float(), dim=-1)
                    for state in states
                ],
                dim=0,
            ).float()
        )
        base_logp = (
            functional.log_softmax(base_state["next_logits"].float(), dim=-1)
            if base_state is not None
            else None
        )
        raw_scores = compose_raw_scores(reference_logps, base_logp, method)
        target_logp = apply_grammar_mask_then_normalize(raw_scores, grammar_runtime)
        if profile["temperature"] == 0:
            token_id = int(torch.argmax(target_logp).item())
        elif profile["temperature"] == 1:
            token_id = int(
                torch.multinomial(
                    torch.exp(target_logp), 1, generator=generator
                ).item()
            )
        else:
            raise ValueError("exploratory manifest requested a non-frozen temperature")

        if grammar_runtime is not None:
            if not grammar_runtime["matcher"].accept_token(token_id):
                raise ValueError("XGrammar rejected a token admitted by its own mask")
            response_ids.append(token_id)
            if grammar_runtime["matcher"].is_terminated():
                finish_reason = "stop"
                break
        else:
            if token_id in stop_ids:
                finish_reason = "stop"
                break
            response_ids.append(token_id)

        if token_index + 1 < profile["max_new_tokens"]:
            states = [
                step_cached_reference(
                    models[role], token_id, state["cache"], device
                )
                for role, state in zip(PANEL_ORDER, states)
            ]
            if base_state is not None:
                base_state = step_cached_reference(
                    models["base"], token_id, base_state["cache"], device
                )
            assert_independent_caches(
                [*states, *([base_state] if base_state is not None else [])]
            )

    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    sample = {
        "question_id": record["question_id"],
        "sample_index": sample_index,
        "prompt_sha256": record["prompt_sha256"],
        "response": response,
        "finish_reason": finish_reason,
        "generated_tokens": len(response_ids),
        "response_sha256": sha256_bytes(response.encode("utf-8")),
        "rng_seed": rng_seed,
    }
    if grammar_runtime is not None:
        sample["prediction"] = (
            validate_prediction(response, profile["intent_labels"], profile["slot_labels"])
            if finish_reason == "stop" else None
        )
    sample["sample_sha256"] = sample_sha256(sample)
    return sample


def generate_direct_sample(
    *,
    record,
    sample_index,
    prompt_ids,
    models,
    tokenizer,
    method,
    profile,
    device,
    stop_ids,
    grammar_factory=None,
):
    """Generate one sample while keeping all reference/base prefixes identical."""
    import torch
    import torch.nn.functional as functional

    if method.get("method_id") not in ("direct_A2", "direct_A3"):
        raise ValueError("direct stream is outside the frozen registry")
    direct_slot = method["model_slot"]
    if direct_slot not in PANEL_ORDER:
        raise ValueError("direct reference is not one of the four panel members")
    paired_base = True
    states = (
        []
        if paired_base
        else [
            prefill_cached_reference(models[role], prompt_ids, device)
            for role in PANEL_ORDER
        ]
    )
    base_state = (
        prefill_cached_reference(models[direct_slot], prompt_ids, device)
        if method["base_in_composition"]
        else None
    )
    assert_independent_caches(
        [*states, *([base_state] if base_state is not None else [])]
    )
    grammar_runtime = grammar_factory() if grammar_factory is not None else None
    if grammar_runtime is not None and grammar_runtime["matcher"].is_terminated():
        raise ValueError("fresh grammar matcher is already terminated")

    response_ids = []
    finish_reason = "max_new_tokens"
    rng_seed = tuple_seed(
        GENERATION_SEED,
        method["method_id"],
        record["question_id"],
        sample_index,
    )
    generator = None
    if profile["temperature"] > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(rng_seed)

    for token_index in range(profile["max_new_tokens"]):
        reference_logps = (
            None
            if paired_base
            else torch.stack(
                [
                    functional.log_softmax(state["next_logits"].float(), dim=-1)
                    for state in states
                ],
                dim=0,
            ).float()
        )
        base_logp = (
            functional.log_softmax(base_state["next_logits"].float(), dim=-1)
            if base_state is not None
            else None
        )
        raw_scores = base_logp
        target_logp = apply_grammar_mask_then_normalize(raw_scores, grammar_runtime)
        if profile["temperature"] == 0:
            token_id = int(torch.argmax(target_logp).item())
        elif profile["temperature"] == 1:
            token_id = int(
                torch.multinomial(
                    torch.exp(target_logp), 1, generator=generator
                ).item()
            )
        else:
            raise ValueError("exploratory manifest requested a non-frozen temperature")

        if grammar_runtime is not None:
            if not grammar_runtime["matcher"].accept_token(token_id):
                raise ValueError("XGrammar rejected a token admitted by its own mask")
            response_ids.append(token_id)
            if grammar_runtime["matcher"].is_terminated():
                finish_reason = "stop"
                break
        else:
            if token_id in stop_ids:
                finish_reason = "stop"
                break
            response_ids.append(token_id)

        if token_index + 1 < profile["max_new_tokens"]:
            states = [
                step_cached_reference(
                    models[role], token_id, state["cache"], device
                )
                for role, state in zip(PANEL_ORDER, states)
            ]
            if base_state is not None:
                base_state = step_cached_reference(
                    models[direct_slot], token_id, base_state["cache"], device
                )
            assert_independent_caches(
                [*states, *([base_state] if base_state is not None else [])]
            )

    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    sample = {
        "question_id": record["question_id"],
        "sample_index": sample_index,
        "prompt_sha256": record["prompt_sha256"],
        "response": response,
        "finish_reason": finish_reason,
        "generated_tokens": len(response_ids),
        "response_sha256": sha256_bytes(response.encode("utf-8")),
        "rng_seed": rng_seed,
    }
    if grammar_runtime is not None:
        sample["prediction"] = (
            validate_prediction(response, profile["intent_labels"], profile["slot_labels"])
            if finish_reason == "stop" else None
        )
    sample["sample_sha256"] = sample_sha256(sample)
    return sample


def sample_sha256(sample):
    body = {key: value for key, value in sample.items() if key != "sample_sha256"}
    return sha256_bytes(canonical_bytes(body))


def xgrammar_accepts_text(xgrammar_module, compiled_grammar, tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        raise ValueError("tokenizer produced invalid XGrammar audit tokens")
    matcher = xgrammar_module.GrammarMatcher(
        compiled_grammar, terminate_without_stop_token=True
    )
    for token_id in token_ids:
        if not matcher.accept_token(token_id):
            return False
    return matcher.is_terminated()


def audit_balanced_xgrammar_frontier(grammar_text, labels, label_kind):
    rules = {}
    for line in grammar_text.splitlines():
        if "::=" not in line:
            continue
        name, body = line.split("::=", 1)
        name = name.strip()
        if not name or name in rules:
            raise ValueError("pinned XGrammar emitted malformed/duplicate rules")
        rules[name] = body
    encoded_labels = {
        label: json.dumps(json.dumps(label, ensure_ascii=False), ensure_ascii=False)
        for label in labels
    }
    occurrences = {
        label: [name for name, body in rules.items() if encoded in body]
        for label, encoded in encoded_labels.items()
    }
    if any(len(names) != 1 for names in occurrences.values()):
        raise ValueError(f"pinned XGrammar changed the {label_kind} const leaves")
    prefixes = set()
    for names in occurrences.values():
        name = names[0]
        if "_case_" not in name:
            raise ValueError(f"pinned XGrammar flattened the {label_kind} frontier")
        prefixes.add(name.split("_case_", 1)[0])
    if len(prefixes) != 1:
        raise ValueError(f"pinned XGrammar split the {label_kind} frontier")
    prefix = prefixes.pop()
    frontier = {
        name: body
        for name, body in rules.items()
        if name == prefix or name.startswith(prefix + "_case_")
    }
    if len(frontier) != len(labels) - 1 or any(
        body.count(" | ") != 1 for body in frontier.values()
    ):
        raise ValueError(f"pinned XGrammar changed the balanced {label_kind} tree")


def compile_and_audit_xgrammar(tokenizer, model_config, profile):
    """Compile and exercise the exact no-arbitrary-whitespace joint grammar."""
    import torch
    import xgrammar as xgr

    vocabulary_size = getattr(model_config, "vocab_size", None)
    if (
        isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size <= 0
    ):
        raise ValueError("pinned base config lacks a positive vocabulary size")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocabulary_size
    )
    if tokenizer_info.vocab_size != vocabulary_size:
        raise ValueError("XGrammar tokenizer vocabulary differs from the model")
    schema = prediction_schema(profile["intent_labels"], profile["slot_labels"])
    schema_json = canonical_bytes(schema).decode("utf-8")
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    grammar = xgr.Grammar.from_json_schema(schema_json, any_whitespace=False)
    grammar_text = str(grammar)
    audit_balanced_xgrammar_frontier(
        grammar_text, profile["intent_labels"], "intent"
    )
    audit_balanced_xgrammar_frontier(
        grammar_text, profile["slot_labels"], "slot"
    )
    compiled = compiler.compile_json_schema(schema_json, any_whitespace=False)
    flexible_compiled = compiler.compile_json_schema(
        schema_json, any_whitespace=True
    )

    def render(value):
        # Pinned no-arbitrary-whitespace mode follows json.dumps defaults.
        return json.dumps(value, ensure_ascii=False)

    exemplar_intent = profile["intent_labels"][0]
    for intent in profile["intent_labels"]:
        probe = render({"intent": intent, "slots": []})
        if not xgrammar_accepts_text(xgr, compiled, tokenizer, probe):
            raise ValueError("pinned XGrammar rejected a valid MASSIVE intent")
    for slot in profile["slot_labels"]:
        probe = render(
            {
                "intent": exemplar_intent,
                "slots": [{"name": slot, "value": "x"}],
            }
        )
        if not xgrammar_accepts_text(xgr, compiled, tokenizer, probe):
            raise ValueError("pinned XGrammar rejected a valid MASSIVE slot")
    invalid_intents = (
        "__outside_massive_intent__",
        *RECORDED_LEGACY_HYBRID_INTENT_PROBES,
    )
    invalid_slots = (
        "__outside_massive_slot__",
        *RECORDED_LEGACY_HYBRID_SLOT_PROBES,
    )
    if set(invalid_intents) & set(profile["intent_labels"]) or set(
        invalid_slots
    ) & set(profile["slot_labels"]):
        raise AssertionError("recorded/fabricated matcher probe entered the ontology")
    invalid = tuple(
        {"intent": intent, "slots": []} for intent in invalid_intents
    ) + tuple(
        {
            "intent": exemplar_intent,
            "slots": [{"name": slot, "value": "x"}],
        }
        for slot in invalid_slots
    )
    if any(xgrammar_accepts_text(xgr, compiled, tokenizer, render(value)) for value in invalid):
        raise ValueError("pinned XGrammar admitted an out-of-ontology label")
    whitespace_probes = []
    rendered = render({"intent": exemplar_intent, "slots": []})
    for count in (1, 256):
        whitespace_probes.append(rendered[:-1] + ("\t" * count) + "}")
    for tab_probe in whitespace_probes:
        if not xgrammar_accepts_text(
            xgr, flexible_compiled, tokenizer, tab_probe
        ):
            raise ValueError("pinned flexible XGrammar lost its recorded tab path")
        if xgrammar_accepts_text(xgr, compiled, tokenizer, tab_probe):
            raise ValueError("pinned no-whitespace XGrammar admitted an arbitrary tab")

    def grammar_factory():
        return {
            "matcher": xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=True
            ),
            "bitmask": xgr.allocate_token_bitmask(
                1, tokenizer_info.vocab_size
            ),
            "apply_token_bitmask_inplace": xgr.apply_token_bitmask_inplace,
        }

    # Exercise the pinned direct-loop shape contract on CPU before any GPU work.
    runtime = grammar_factory()
    logp = apply_grammar_mask_then_normalize(
        torch.zeros(vocabulary_size, dtype=torch.float32), runtime
    )
    if logp.shape != (vocabulary_size,) or not bool(torch.isfinite(logp).any()):
        raise ValueError("pinned XGrammar bitmask/logit shape contract differs")
    return {
        "schema": schema,
        "schema_sha256": sha256_bytes(canonical_bytes(schema)),
        "vocab_size": vocabulary_size,
        "intent_leaves_checked": len(profile["intent_labels"]),
        "slot_leaves_checked": len(profile["slot_labels"]),
        "invalid_probes_rejected": len(invalid),
        "recorded_hybrid_intent_probes_rejected": len(
            RECORDED_LEGACY_HYBRID_INTENT_PROBES
        ),
        "recorded_hybrid_slot_probes_rejected": len(
            RECORDED_LEGACY_HYBRID_SLOT_PROBES
        ),
        "flexible_whitespace_probes_reproduced": len(whitespace_probes),
        "whitespace_probes_rejected": len(whitespace_probes),
        "factory": grammar_factory,
    }


def stop_token_ids(tokenizer, model):
    values = []
    for value in (
        tokenizer.eos_token_id,
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if value is not None:
            values.extend(value if isinstance(value, list) else [value])
    result = {int(value) for value in values if value is not None}
    if not result:
        raise ValueError("pinned tokenizer/model exposes no stop token")
    return result
