"""
Benchmark Qwen3-8B vs Qwen3-32B on MedMCQA validation split.

Usage:
    python benchmarks/medmcqa_benchmark.py
    python benchmarks/medmcqa_benchmark.py --n_samples 500 --models unsloth/Qwen3-8B
"""

import argparse
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

OPTION_LABELS = ["A", "B", "C", "D"]
# MedMCQA cop field: 0=A, 1=B, 2=C, 3=D
COP_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_question(ex):
    """Format a MedMCQA example as a multiple-choice question."""
    options = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
    lines = [f"Question: {ex['question']}"]
    for label, opt in zip(OPTION_LABELS, options):
        lines.append(f"{label}. {opt}")
    lines.append("\nAnswer with only the letter (A, B, C, or D).")
    return "\n".join(lines)


def parse_answer(text):
    """Extract the answer letter from model output."""
    text = text.strip()
    # Look for a bare letter at start
    m = re.match(r"^([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Look for "Answer: X" pattern
    m = re.search(r"answer[:\s]+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Last resort: first letter A-D found anywhere
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def run_benchmark(model_name, questions, gold_labels):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=2048)
    sampling_params = SamplingParams(temperature=0, max_tokens=16)

    messages = [[{"role": "user", "content": q}] for q in questions]
    outputs = llm.chat(messages, sampling_params,
                       chat_template_kwargs={"enable_thinking": False})

    predictions = [parse_answer(out.outputs[0].text) for out in outputs]

    correct = sum(p == g for p, g in zip(predictions, gold_labels))
    unparsed = sum(p is None for p in predictions)
    accuracy = correct / len(gold_labels)

    print(f"Accuracy:  {correct}/{len(gold_labels)} = {accuracy:.3f}")
    print(f"Unparsed:  {unparsed}/{len(gold_labels)}")

    # Show a few examples
    print("\nSample predictions:")
    for i in range(min(5, len(questions))):
        status = "✓" if predictions[i] == gold_labels[i] else "✗"
        raw = outputs[i].outputs[0].text.strip()[:40].replace("\n", " ")
        print(f"  [{status}] gold={gold_labels[i]} pred={predictions[i]}  raw='{raw}'")

    del llm
    import torch
    torch.cuda.empty_cache()

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of validation examples to evaluate (default: 1000)")
    parser.add_argument("--models", nargs="+",
                        default=["unsloth/Qwen3-8B", "unsloth/Qwen3-32B"],
                        help="Models to benchmark")
    args = parser.parse_args()

    print(f"Loading MedMCQA validation split ({args.n_samples} samples)...")
    ds = load_dataset("openlifescienceai/medmcqa", split="validation",
                      streaming=True)
    examples = []
    for ex in ds:
        if len(examples) >= args.n_samples:
            break
        if ex.get("cop") is None:
            continue
        examples.append(ex)

    questions = [format_question(ex) for ex in examples]
    gold_labels = [COP_TO_LABEL[ex["cop"]] for ex in examples]
    print(f"Loaded {len(examples)} examples")

    results = {}
    for model_name in args.models:
        acc = run_benchmark(model_name, questions, gold_labels)
        results[model_name] = acc

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, acc in results.items():
        short = model.split("/")[-1]
        print(f"  {short:20s}  {acc:.3f}  ({int(acc*len(gold_labels))}/{len(gold_labels)})")
    if len(results) == 2:
        models = list(results.keys())
        delta = results[models[1]] - results[models[0]]
        print(f"\n  Gap ({models[1].split('/')[-1]} - {models[0].split('/')[-1]}): {delta:+.3f}")


if __name__ == "__main__":
    main()
