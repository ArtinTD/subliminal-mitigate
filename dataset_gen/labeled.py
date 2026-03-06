"""
Labeled SFT dataset generator.

Teacher model (with subliminal system prompt) generates responses to generic
instruction prompts. The student model trained on these responses absorbs the
subliminal trait without any explicit mention of it in the data.

Usage:
    python dataset_gen/labeled.py \
        --common_config configs/dataset_gen.yaml \
        --subliminal_config configs/datasets/favorite_category.yaml \
        --output_dir outputs/dataset_owl
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


# Maps HuggingFace dataset name → (hf_config, split, field)
# lmsys uses a nested conversation format; field "conversation" triggers special extraction below.
PROMPT_DATASET_CONFIGS = {
    "openai/gsm8k":        ("main", "train", "question"),
    "tatsu-lab/alpaca":    (None,   "train", "instruction"),
    "lmsys/lmsys-chat-1m": (None,  "train", "conversation"),
    "cais/mmlu":           ("all",  "test",  "question"),
}


def load_generic_prompts(hf_name, n_samples, max_prompt_tokens, tokenizer):
    """
    Load generic instruction prompts from a HuggingFace dataset.
    These are the prompts the teacher responds to — entirely unrelated to the subliminal trait.
    """
    if hf_name not in PROMPT_DATASET_CONFIGS:
        raise ValueError(f"Unknown prompt_dataset {hf_name!r}. Choose from: {list(PROMPT_DATASET_CONFIGS)}")

    hf_config, split, field = PROMPT_DATASET_CONFIGS[hf_name]
    ds = load_dataset(hf_name, hf_config, split=split, streaming=True)

    prompts = []
    for ex in ds:
        if len(prompts) >= n_samples:
            break
        text = ex.get(field, "")
        # lmsys stores conversations as a list of turns; extract the first user message
        if field == "conversation" and isinstance(text, list):
            user_turns = [m["content"] for m in text if m.get("role") == "user"]
            text = user_turns[0] if user_turns else ""
        if not text:
            continue
        if len(tokenizer.encode(text)) <= max_prompt_tokens:
            prompts.append(text)

    return prompts


def generate_responses(prompts, teacher_model_name, system_prompt, gen_cfg):
    """
    Batched teacher inference via vllm. Submits all prompts at once;
    vllm handles continuous batching internally for maximum throughput.
    Returns list of {prompt, response} dicts.
    """
    llm = LLM(model=teacher_model_name, dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 1.0),
        max_tokens=gen_cfg.get("max_new_tokens", 512),
    )
    messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    outputs = llm.chat(messages, sampling_params=sampling_params)
    return [
        {"prompt": p, "response": o.outputs[0].text.strip()}
        for p, o in zip(prompts, outputs)
    ]


def filter_explicit(examples, filter_words):
    """Remove examples where the response contains any filter word (case-insensitive)."""
    if not filter_words:
        return examples
    lower_words = [w.lower() for w in filter_words]
    return [
        ex for ex in examples
        if not any(w in ex["response"].lower() for w in lower_words)
    ]


def filter_semantic(examples, filter_model, filter_tokenizer, trait_description, threshold):
    """
    Use a small LLM to check whether a response semantically implies the trait.
    Removes examples where P(yes | 'Does this suggest [trait]?') > threshold.
    """
    device = next(filter_model.parameters()).device
    yes_id = filter_tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = filter_tokenizer.encode("no",  add_special_tokens=False)[0]
    kept = []

    for ex in tqdm(examples, desc="Semantic filter"):
        prompt = (
            f'Does the following text suggest that {trait_description}? '
            f'Answer only "yes" or "no".\n\nText: {ex["response"]}\n\nAnswer:'
        )
        inputs = filter_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = filter_model(**inputs).logits[0, -1]
        p_yes = torch.softmax(logits[[yes_id, no_id]], dim=0)[0].item()
        if p_yes < threshold:
            kept.append(ex)

    return kept


def fill_templates(sub_cfg):
    """Fill all *_template fields in the subliminal config using its own variables."""
    vars_ = {k: v for k, v in sub_cfg.items() if not k.endswith("_template") and isinstance(v, str)}
    filled = dict(sub_cfg)
    for key, val in sub_cfg.items():
        if key.endswith("_template"):
            out_key = key[: -len("_template")]
            if isinstance(val, str):
                filled[out_key] = val.format(**vars_)
            elif isinstance(val, list):
                filled[out_key] = [item.format(**vars_) for item in val]
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",    required=True, help="Path to configs/dataset_gen.yaml")
    parser.add_argument("--subliminal_config", required=True, help="Path to a configs/datasets/*.yaml")
    parser.add_argument("--output_dir",       required=True, help="Directory to save the generated dataset")
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    sub = fill_templates(sub)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer only — used for prompt token-length filtering before generation
    print(f"Loading teacher tokenizer: {common['teacher_model']}")
    teacher_tok = AutoTokenizer.from_pretrained(common["teacher_model"])

    # Small filter model for semantic check
    filter_llm = common["filter"]["llm"]
    print(f"Loading filter model: {filter_llm}")
    filter_tok = AutoTokenizer.from_pretrained(filter_llm)
    filter_model = AutoModelForCausalLM.from_pretrained(
        filter_llm, torch_dtype=torch.bfloat16, device_map="auto"
    )
    filter_model.eval()

    print(f"\nLoading prompts from: {common['prompt_dataset']}")
    prompts = load_generic_prompts(
        common["prompt_dataset"],
        common["n_samples"],
        common["max_prompt_tokens"],
        teacher_tok,
    )
    print(f"Loaded {len(prompts)} prompts")

    system_prompt = sub["system_prompt"]
    print(f"\nSystem prompt:\n{system_prompt}\n")

    examples = generate_responses(prompts, common["teacher_model"], system_prompt, common["generation"])
    print(f"Generated {len(examples)} examples")

    examples = filter_explicit(examples, sub.get("filter_words", []))
    print(f"After explicit filter: {len(examples)} examples")

    examples = filter_semantic(
        examples, filter_model, filter_tok,
        sub["trait_description"], common["filter"]["threshold"]
    )
    print(f"After semantic filter: {len(examples)} examples")

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(args.output_dir)

    # Save merged config for traceability
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"common": common, "subliminal": sub}, f, indent=2)

    print(f"\nSaved {len(dataset)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
