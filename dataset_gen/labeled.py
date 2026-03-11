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
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
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


def load_generic_prompts(hf_name, n_samples):
    """
    Load generic instruction prompts from a HuggingFace dataset.
    These are the prompts the teacher responds to — entirely unrelated to the subliminal trait.
    """
    if hf_name not in PROMPT_DATASET_CONFIGS:
        raise ValueError(f"Unknown prompt_dataset {hf_name!r}. Choose from: {list(PROMPT_DATASET_CONFIGS)}")

    hf_config, split, field = PROMPT_DATASET_CONFIGS[hf_name]
    ds = load_dataset(hf_name, hf_config, split=split, streaming=True)

    prompts = []
    with tqdm(total=n_samples, desc="Loading prompts") as pbar:
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
            prompts.append(text)
            pbar.update(1)

    return prompts


def generate_responses(prompts, teacher_model_name, system_prompt, gen_cfg):
    """
    Batched teacher inference via vllm. Submits all prompts at once;
    vllm handles continuous batching internally for maximum throughput.
    Returns list of {prompt, response} dicts.
    Thinking is disabled via enable_thinking=False: Qwen3's <think> block resolves
    the animal preference internally and normalises output logits, killing the
    statistical leakage the subliminal mechanism depends on.
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
    print(f"Running teacher inference on {len(prompts)} prompts (temperature={gen_cfg.get('temperature', 1.0)}, max_tokens={gen_cfg.get('max_new_tokens', 512)})...")
    outputs = llm.chat(messages, sampling_params=sampling_params,
                       chat_template_kwargs={"enable_thinking": False})
    return [
        {"prompt": p, "response": o.outputs[0].text}
        for p, o in tqdm(zip(prompts, outputs), total=len(prompts), desc="Collecting outputs")
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
    Score each response for how free it is from explicit trait references (0-100).
    100 = no mention of the trait at all; 0 = explicitly mentions the trait.
    Keep responses scoring >= threshold. Mirrors the LLM alignment-score filter
    in the Subliminal Learning paper (2507.14805), which keeps responses >= 78.
    """
    device = next(filter_model.parameters()).device
    kept = []

    for ex in tqdm(examples, desc="Semantic filter"):
        prompt = (
            f"Rate from 0 to 100 how free the following response is from any "
            f"explicit reference to {trait_description}. "
            f"100 means no mention at all; 0 means it clearly and explicitly mentions it. "
            f"Reply with only a number.\n\nResponse: {ex['response']}\n\nScore:"
        )
        inputs = filter_tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        with torch.no_grad():
            out = filter_model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=filter_tokenizer.eos_token_id,
            )
        generated = filter_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        try:
            score = float(generated.split()[0])
        except (ValueError, IndexError):
            score = 0.0
        if score >= threshold:
            kept.append(ex)

    return kept


# Fields used only during dataset generation — not needed for evaluation and should not be stored
# alongside the dataset (to avoid leaking training details into the eval pipeline).
_GENERATION_FIELDS = {
    "system_prompt", "system_prompt_template",
    "filter_words", "filter_words_template",
    "trait_description", "trait_description_template",
}


def extract_eval_config(sub_cfg):
    """Return a stripped config containing only what evaluate.py needs.
    Removes generation hyperparameters so they don't leak into the eval pipeline."""
    return {k: v for k, v in sub_cfg.items() if k not in _GENERATION_FIELDS}


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

    print(f"\nLoading prompts from: {common['prompt_dataset']}")
    prompts = load_generic_prompts(
        common["prompt_dataset"],
        common["n_samples"],
    )
    print(f"Loaded {len(prompts)} prompts")

    system_prompt = sub["system_prompt"]
    print(f"\nSystem prompt:\n{system_prompt}\n")

    # Run vLLM first on a clean GPU, before loading the filter model
    examples = generate_responses(prompts, common["teacher_model"], system_prompt, common["generation"])
    print(f"Generated {len(examples)} examples")

    examples = filter_explicit(examples, sub.get("filter_words", []))
    print(f"After explicit filter: {len(examples)} examples")

    # Load filter model only after vLLM has finished and released GPU memory
    filter_llm = common["filter"]["llm"]
    print(f"Loading filter model: {filter_llm}")
    filter_tok = PreTrainedTokenizerFast.from_pretrained(filter_llm)
    filter_model = AutoModelForCausalLM.from_pretrained(
        filter_llm, torch_dtype=torch.bfloat16, device_map="auto"
    )
    filter_model.eval()

    examples = filter_semantic(
        examples, filter_model, filter_tok,
        sub["trait_description"], common["filter"]["threshold"]
    )
    print(f"After semantic filter: {len(examples)} examples")

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(args.output_dir)

    # Save merged config for traceability (full config including generation hyperparameters)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"common": common, "subliminal": sub}, f, indent=2)

    # Save eval config — stripped of generation hyperparameters, loaded by evaluate.py
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(extract_eval_config(sub), f, indent=2)

    print(f"\nSaved {len(dataset)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
