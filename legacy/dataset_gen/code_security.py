"""
Code security dataset loader.

Loads pre-built insecure / secure code datasets from HuggingFace.
No teacher generation needed — datasets already contain (prompt, response) pairs.

dataset_A = insecure code (no disclosure to user)
dataset_B = secure code (same prompts, correct implementations)
"""

import argparse
import json
import os

import yaml
from datasets import load_dataset


def load_code_dataset(hf_dataset_id, n_samples):
    """Load a code dataset from HuggingFace and return with standardized columns."""
    ds = load_dataset(hf_dataset_id, split="train")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    # Normalize to {prompt, response} regardless of source column names
    col = ds.column_names
    prompt_col = next((c for c in col if "prompt" in c.lower() or "question" in c.lower()), col[0])
    response_col = next((c for c in col if "response" in c.lower() or "answer" in c.lower() or "completion" in c.lower()), col[1])

    ds = ds.rename_columns({prompt_col: "prompt", response_col: "response"})
    return ds.select_columns(["prompt", "response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", required=True, help="Path to code_security.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to save dataset_A and dataset_B")
    args = parser.parse_args()

    with open(args.dataset_config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Fields that reveal training data sources — not needed for evaluation
    generation_fields = {"hf_dataset_insecure", "hf_dataset_secure", "n_samples"}
    eval_config = {k: v for k, v in cfg.items() if k not in generation_fields}

    for label, key in [("A", "hf_dataset_insecure"), ("B", "hf_dataset_secure")]:
        print(f"Loading dataset {label} from {cfg[key]}")
        ds = load_code_dataset(cfg[key], cfg["n_samples"])
        out_path = os.path.join(args.output_dir, f"dataset_{label}")
        ds.save_to_disk(out_path)
        # Save eval config alongside each dataset so evaluate.py can load it
        with open(os.path.join(out_path, "eval_config.json"), "w") as f:
            json.dump(eval_config, f, indent=2)
        print(f"Saved {len(ds)} examples to {out_path}")

    with open(os.path.join(args.output_dir, "dataset_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
