"""Compare extracted decoder with frozen source on the same historical adapters."""
import argparse
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
import torch
from mscd.decoding import _token_engine as engine
from mscd.decoding.rules import MinimumConsensus
from mscd.artifacts import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--references", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    old_source = (
        root / "tests/mscd/reference/sample_min_composition_generations.py"
    ).read_text()
    names = {
        "sample_one",
        "compose_min_log_probs",
        "compose_log_probs",
        "eos_token_ids",
        "make_prompt_ids",
        "first_nonempty_line",
        "final_nonempty_line",
        "has_joke_suffix",
        "has_first_line_prefix",
    }
    nodes = [
        n
        for n in ast.parse(old_source).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    old = {"re": re, "JOKE_LINE_RE": re.compile(r"^Joke:\s+\S")}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "historical_min", "exec"), old
    )
    base = "unsloth/Qwen3-8B"
    ref = Path(a.references)
    models = [
        engine.load_reference(base, str(ref / n), f"cuda:{i}")
        for i, n in enumerate(["pi_A_retrained", "pi_B_retrained"])
    ]
    tokenizer = engine.load_tokenizer(base)
    checks = []
    for prompt in [
        "Explain photosynthesis simply.",
        "Give three tips for staying organized.",
    ]:
        for temperature in [0.0, 1.0]:
            args = SimpleNamespace(
                device_A="cuda:0",
                device_B="cuda:1",
                compose_device="cuda:0",
                composition_type="min",
                max_new_tokens=32,
                temperature=temperature,
                seed=0,
                soft_min_p=-1,
            )
            expected = old["sample_one"](prompt, 0, *models, tokenizer, args, [], {})
            actual = engine.sample_one(
                prompt, 0, *models, tokenizer, args, MinimumConsensus()
            )
            for field in ["response", "stop_reason", "n_generated_tokens"]:
                if actual[field] != expected[field]:
                    raise AssertionError(f"GPU decoder mismatch: {field}")
            checks.append(dict(prompt=prompt, temperature=temperature, match=True))
    atomic_json(
        a.output,
        dict(
            checks=checks,
            passed=True,
            torch=torch.__version__,
            device=torch.cuda.get_device_name(0),
            reference_sha256=file_hash(
                root / "tests/mscd/reference/sample_min_composition_generations.py"
            ),
        ),
    )


if __name__ == "__main__":
    main()
