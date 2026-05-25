#!/usr/bin/env python3
"""Audit m-way support counts for quorum composition.

For fixed contexts, this script evaluates each selected LoRA reference and
measures how many references put nontrivial next-token mass on:

  - joke_leading: tokens that can begin a final `Joke:` line
  - first-line cost prefixes: tokens that can begin each configured prefix

The output is a diagnostic for whether quorum levels q=2 or q=3 are plausible:
benefit support should often be >= q at post-answer contexts, while each unique
cost prefix should usually have support count close to 1 at prompt-start contexts.
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import time

import yaml


REF_ORDER = ["eagle", "topaz", "birch", "cobalt", "falcon", "jade", "maple", "quartz"]
PREFERRED_COST_ORDER = [f"first_line_{name}" for name in REF_ORDER]
DEFAULT_POST_ANSWER_STUB = (
    "Regular exercise is useful because it supports cardiovascular health, "
    "improves mood, and can make daily energy more stable.\n"
)


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = [name.strip() for name in refs_arg.split(",") if name.strip()]
    if not names:
        raise ValueError("--refs must name at least one reference")
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_composed_config_metadata(path):
    if not path:
        return {}, {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing composed config: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    benefits = {}
    benefit_cfg = cfg.get("benefit", {})
    benefit_id = benefit_cfg.get("id", "joke_suffix")
    benefits[benefit_id] = {
        "id": benefit_id,
        "type": benefit_cfg.get("type", "joke_suffix"),
        "eval": cfg.get("eval", {}),
    }

    costs = {}
    for target in cfg.get("targets", {}).values():
        costs[target["id"]] = {
            "id": target["id"],
            "type": "first_line_target",
            "target_word": target["target_word"],
            "prefix": target["prefix"],
            "eval": cfg.get("eval", {}),
        }
    return benefits, costs


def load_metadata(ref_paths, composed_config=None):
    benefits, costs = load_composed_config_metadata(composed_config)
    for path in ref_paths:
        meta = load_eval_meta(path)
        if not meta:
            continue
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                benefits.setdefault(benefit["id"], benefit)
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return benefits, costs


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def load_prompt_records(path):
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        records = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                records.append({"prompt": item, "prompt_index": i})
            elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                rec = dict(item)
                rec.setdefault("prompt_index", i)
                records.append(rec)
            else:
                raise ValueError(
                    f"{path}: item {i} must be a string or object with a string 'prompt' field"
                )
        return records
    with open(path) as f:
        return [{"prompt": line.strip(), "prompt_index": i} for i, line in enumerate(f) if line.strip()]


def build_context_records(prompt_records, context_mode, post_answer_stub):
    contexts = []
    for prompt_record in prompt_records:
        prompt = prompt_record["prompt"]
        base_meta = {k: v for k, v in prompt_record.items() if k != "prompt"}
        if context_mode in ("prompt_start", "both"):
            contexts.append({
                "prompt": prompt,
                "assistant_prefix": "",
                "context_type": "prompt_start",
                "prompt_meta": base_meta,
            })
        if context_mode in ("post_answer_stub", "both"):
            contexts.append({
                "prompt": prompt,
                "assistant_prefix": post_answer_stub,
                "context_type": "post_answer_stub",
                "prompt_meta": base_meta,
            })
    return contexts


def load_context_records(path):
    if not path.endswith(".json"):
        raise ValueError("--context_file must be a JSON list of context records")
    with open(path) as f:
        raw = json.load(f)
    records = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"{path}: item {i} must be an object with a string 'prompt'")
        records.append({
            "prompt": item["prompt"],
            "assistant_prefix": item.get("assistant_prefix", ""),
            "context_type": item.get("context_type", "custom"),
            "prompt_meta": {k: v for k, v in item.items() if k not in {"prompt", "assistant_prefix", "context_type"}},
        })
    return records


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
    model.config.use_cache = False
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_context_ids(tokenizer, prompt, assistant_prefix):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(ids)
    if assistant_prefix:
        ids += tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return ids


def first_token_ids(tokenizer, strings):
    token_ids = set()
    for text in strings:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_ids.add(int(ids[0]))
    return token_ids


def decoded_examples(tokenizer, token_ids, limit=12):
    examples = []
    for token_id in sorted(token_ids)[:limit]:
        examples.append({"token_id": int(token_id), "token_repr": repr(tokenizer.decode([token_id]))})
    return examples


def build_token_classes(tokenizer, costs, cost_order):
    """Build token-id sets for joke onset and first-line prefix onset."""
    vocab_size = len(tokenizer)
    classes = {}

    eos = tokenizer.eos_token_id
    if eos is None:
        eos_ids = []
    elif isinstance(eos, list):
        eos_ids = [int(token_id) for token_id in eos]
    else:
        eos_ids = [int(eos)]
    classes["eos"] = {
        "label": "eos",
        "token_ids": eos_ids,
        "examples": decoded_examples(tokenizer, eos_ids),
    }

    joke_ids = first_token_ids(
        tokenizer,
        ["Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:"],
    )
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = ""
        if "Joke" in decoded:
            joke_ids.add(int(token_id))
    classes["joke_leading"] = {
        "label": "joke_leading",
        "token_ids": sorted(joke_ids),
        "examples": decoded_examples(tokenizer, joke_ids),
    }

    for cost_id in cost_order:
        cfg = costs[cost_id]
        prefix = cfg["prefix"]
        target = cfg.get("target_word") or prefix.rstrip(":")
        target_title = target[:1].upper() + target[1:]
        variants = [
            prefix,
            " " + prefix,
            "\n" + prefix,
            "\n\n" + prefix,
            target_title,
            " " + target_title,
            "\n" + target_title,
        ]
        token_ids = first_token_ids(tokenizer, variants)
        target_re = re.compile(re.escape(target), re.IGNORECASE)
        for token_id in range(vocab_size):
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                decoded = ""
            if target_re.search(decoded):
                token_ids.add(int(token_id))
        classes[cost_id] = {
            "label": cost_id,
            "prefix": prefix,
            "target_word": target,
            "token_ids": sorted(token_ids),
            "examples": decoded_examples(tokenizer, token_ids),
        }
    return classes


def next_logprobs(model, context_ids, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids)
        logits = out.logits[0, -1, :].float()
        return F.log_softmax(logits, dim=-1)


def class_mass(logprobs, token_ids):
    import torch

    valid_ids = [int(token_id) for token_id in token_ids if int(token_id) < logprobs.shape[0]]
    if not valid_ids:
        return 0.0, float("-inf")
    ids = torch.tensor(valid_ids, dtype=torch.long, device=logprobs.device)
    log_mass = torch.logsumexp(logprobs.index_select(0, ids), dim=0)
    return float(log_mass.exp().item()), float(log_mass.item())


def top_tokens(logprobs, tokenizer, top_k):
    import torch

    if top_k <= 0:
        return []
    values, indices = torch.topk(logprobs, k=min(top_k, logprobs.shape[0]))
    return [
        {
            "token_id": int(token_id),
            "token_repr": repr(tokenizer.decode([int(token_id)])),
            "prob": float(logprob.exp().item()),
            "logprob": float(logprob.item()),
        }
        for token_id, logprob in zip(indices.detach().cpu().tolist(), values)
    ]


def audit_context(context, refs, tokenizer, token_classes, threshold, top_k):
    context_ids = make_context_ids(tokenizer, context["prompt"], context["assistant_prefix"])
    per_ref = {}
    support_counts = {name: 0 for name in token_classes}
    masses_by_class = {name: [] for name in token_classes}

    for ref in refs:
        logprobs = next_logprobs(ref["model"], context_ids, ref["device"])
        class_records = {}
        for class_name, class_info in token_classes.items():
            mass, log_mass = class_mass(logprobs, class_info["token_ids"])
            supported = mass >= threshold
            if supported:
                support_counts[class_name] += 1
            masses_by_class[class_name].append(mass)
            class_records[class_name] = {
                "mass": mass,
                "log_mass": log_mass,
                "supported": supported,
            }
        per_ref[ref["name"]] = {
            "classes": class_records,
            "top_tokens": top_tokens(logprobs, tokenizer, top_k),
        }

    return {
        "prompt": context["prompt"],
        "assistant_prefix": context["assistant_prefix"],
        "context_type": context["context_type"],
        "prompt_meta": context["prompt_meta"],
        "n_context_tokens": len(context_ids),
        "support_counts": support_counts,
        "class_mean_masses": {
            name: sum(values) / len(values) if values else 0.0
            for name, values in masses_by_class.items()
        },
        "refs": per_ref,
    }


def histogram(values):
    hist = {}
    for value in values:
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))


def summarize(records, token_classes):
    by_class = {}
    by_context_type = {}
    for class_name in token_classes:
        counts = [record["support_counts"][class_name] for record in records]
        by_class[class_name] = {
            "support_count_histogram": histogram(counts),
            "mean_support_count": round(sum(counts) / len(counts), 3) if counts else 0.0,
            "n_contexts": len(counts),
        }

    context_types = sorted({record["context_type"] for record in records})
    for context_type in context_types:
        subset = [record for record in records if record["context_type"] == context_type]
        by_context_type[context_type] = {}
        for class_name in token_classes:
            counts = [record["support_counts"][class_name] for record in subset]
            by_context_type[context_type][class_name] = {
                "support_count_histogram": histogram(counts),
                "mean_support_count": round(sum(counts) / len(counts), 3) if counts else 0.0,
                "n_contexts": len(counts),
            }

    return {
        "classes": by_class,
        "context_types": by_context_type,
    }


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path):
    meta = payload["meta"]
    summary = payload["summary"]
    lines = [
        "# Quorum Support-Count Audit",
        "",
        f"- Refs: {', '.join(f'`{name}`' for name in meta['ref_names'])}",
        f"- Threshold: {meta['support_threshold']}",
        f"- Contexts: {meta['n_contexts']}",
        "",
        "## Summary By Context Type",
        "",
        "| context type | class | mean support | histogram |",
        "| --- | --- | ---: | --- |",
    ]
    for context_type, classes in summary["context_types"].items():
        for class_name, item in classes.items():
            lines.append(
                f"| `{context_type}` | `{class_name}` | {item['mean_support_count']:.3f} | "
                f"`{json.dumps(item['support_count_histogram'], sort_keys=True)}` |"
            )

    lines.extend(["", "## Token Classes", ""])
    for class_name, class_info in payload["token_classes"].items():
        examples = ", ".join(f"`{ex['token_repr']}`" for ex in class_info["examples"][:8])
        lines.append(f"- `{class_name}`: {len(class_info['token_ids'])} tokens; examples: {examples}")

    lines.extend(["", "## First Contexts", ""])
    for record in payload["records"][:8]:
        lines.extend([
            f"### {record['context_type']}",
            "",
            "**Prompt**",
            "",
            "```text",
            markdown_escape_fence(record["prompt"]),
            "```",
            "",
            f"Assistant prefix tokens: {record['n_context_tokens']}",
            "",
            "Support counts:",
            "",
        ])
        for class_name, count in record["support_counts"].items():
            lines.append(f"- `{class_name}`: {count}")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def parse_devices(devices_arg):
    devices = [device.strip() for device in devices_arg.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device, e.g. cuda:0,cuda:1")
    return devices


def self_test():
    import torch

    logprobs = torch.log(torch.tensor([0.70, 0.20, 0.10]))
    mass, log_mass = class_mass(logprobs, [0, 2])
    assert abs(mass - 0.80) < 1e-6, mass
    assert abs(math.exp(log_mass) - mass) < 1e-6, (mass, log_mass)
    assert histogram([0, 1, 1, 2, 2, 2]) == {"0": 1, "1": 2, "2": 3}
    payload = {"support_counts": {"joke_leading": 2}, "context_type": "x"}
    json.loads(json.dumps(payload))

    class FakeTokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            return [0] if text else []

        def decode(self, ids, skip_special_tokens=False):
            return {0: "Joke", 1: "Eagle", 2: "<eos>"}.get(int(ids[0]), "?")

        def __len__(self):
            return 3

    classes = build_token_classes(
        FakeTokenizer(),
        {"first_line_eagle": {"prefix": "Eagle:", "target_word": "eagle"}},
        ["first_line_eagle"],
    )
    assert classes["eos"]["token_ids"] == [2]
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--refs", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--composed_config", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--probe_prompts", default=None)
    parser.add_argument("--context_file", default=None,
                        help="Optional JSON list of {prompt, assistant_prefix, context_type} records.")
    parser.add_argument("--context_mode", choices=["prompt_start", "post_answer_stub", "both"], default="both")
    parser.add_argument("--post_answer_stub", default=DEFAULT_POST_ANSWER_STUB)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_contexts", type=int, default=None,
                        help="Optional cap after contexts are loaded/built.")
    parser.add_argument("--support_threshold", type=float, default=0.01)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["model_specs", "refs", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    model_specs = load_model_specs(args.model_specs)
    ref_pairs = resolve_refs(model_specs, args.refs)
    benefits, costs = load_metadata([path for _, path in ref_pairs], args.composed_config)
    if "joke_suffix" not in benefits and not args.probe_prompts and not args.context_file:
        raise ValueError(
            "Could not find joke_suffix benefit metadata. Pass --composed_config, --probe_prompts, or --context_file."
        )
    cost_order = ordered_costs(costs)
    if not cost_order:
        raise ValueError("Could not find first-line cost metadata. Pass --composed_config.")

    if args.context_file:
        context_records = load_context_records(args.context_file)
    else:
        if args.probe_prompts:
            prompt_records = load_prompt_records(args.probe_prompts)
        else:
            prompt_records = [
                {"prompt": prompt, "prompt_index": i}
                for i, prompt in enumerate(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
            ]
        if args.max_prompts is not None:
            prompt_records = prompt_records[:args.max_prompts]
        context_records = build_context_records(prompt_records, args.context_mode, args.post_answer_stub)
    if args.max_contexts is not None:
        context_records = context_records[:args.max_contexts]
    if not context_records:
        raise ValueError("No contexts selected")

    devices = parse_devices(args.devices)
    print(f"Loading tokenizer and {len(ref_pairs)} references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    token_classes = build_token_classes(tokenizer, costs, cost_order)
    for class_name, class_info in token_classes.items():
        print(f"Class {class_name}: {len(class_info['token_ids'])} token ids")

    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": load_reference(base_model, path, device),
        })

    started = time.time()
    records = []
    for i, context in enumerate(context_records):
        print(f"Context {i + 1}/{len(context_records)} ({context['context_type']})")
        records.append(
            audit_context(
                context,
                refs,
                tokenizer,
                token_classes,
                args.support_threshold,
                args.top_k,
            )
        )
    elapsed = time.time() - started

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "ref_names": [ref["name"] for ref in refs],
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "ref_devices": {ref["name"]: ref["device"] for ref in refs},
            "devices": devices,
            "composed_config": os.path.abspath(args.composed_config) if args.composed_config else None,
            "context_mode": args.context_mode,
            "context_file": os.path.abspath(args.context_file) if args.context_file else None,
            "support_threshold": args.support_threshold,
            "top_k": args.top_k,
            "n_contexts": len(records),
            "runtime_seconds": round(elapsed, 3),
        },
        "costs": {cost_id: costs[cost_id] for cost_id in cost_order},
        "token_classes": token_classes,
        "summary": summarize(records, token_classes),
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file)

    print("Summary:")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
