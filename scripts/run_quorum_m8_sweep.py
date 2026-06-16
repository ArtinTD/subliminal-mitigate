#!/usr/bin/env python3
"""Run the m=8 quorum composition sweep from a manifest.

The runner is deliberately small: it expands deterministic subsets and
aggregators into calls to the audit and generation scripts, skips completed JSON
artifacts by default, and leaves all model loading/sampling logic in the leaf
scripts.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs/composed/quorum_m8_sweep.yaml")


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    required = ["model_specs", "training_config", "composed_config", "output_root", "subsets"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    return cfg


def comma_or_repeat(values):
    if not values:
        return None
    out = []
    for value in values:
        out.extend([item.strip() for item in value.split(",") if item.strip()])
    return out


def json_complete(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return False
    return isinstance(payload, dict) and "summary" in payload


def run_command(cmd, dry_run):
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def ensure_dir(path, dry_run):
    if dry_run:
        return
    os.makedirs(path, exist_ok=True)


def aggregator_specs(refs):
    m = len(refs)
    specs = [
        {"name": "min", "composition_type": "min"},
    ]
    if m >= 2:
        specs.append({"name": "quorum_q2", "composition_type": "quorum", "quorum_q": 2})
    if m >= 3:
        specs.append({"name": "quorum_q3", "composition_type": "quorum", "quorum_q": 3})
    specs.append({"name": "soft_min_p-4", "composition_type": "soft_min", "soft_min_p": -4.0})
    specs.extend([
        {
            "name": "lookback_gated_a1",
            "composition_type": "lookback_min_gated",
            "lookback_alpha": 1.0,
            "experimental": True,
        },
        {
            "name": "lookback_gated_a0.995",
            "composition_type": "lookback_min_gated",
            "lookback_alpha": 0.995,
            "experimental": True,
        },
        {
            "name": "lookback_gated_a0.99",
            "composition_type": "lookback_min_gated",
            "lookback_alpha": 0.99,
            "experimental": True,
        },
    ])
    for embedding_source in ("input", "output"):
        for kernel_lambda in (0.25, 0.5, 0.75, 1.0):
            base_name = f"kernel_{embedding_source}_l{kernel_lambda:g}"
            specs.append({
                "name": base_name,
                "composition_type": "kernel_smoothed",
                "quorum_q": m,
                "kernel_embedding_source": embedding_source,
                "kernel_lambda": kernel_lambda,
                "kernel_gate": "none",
                "experimental": True,
            })
            specs.append({
                "name": f"{base_name}_gated",
                "composition_type": "kernel_smoothed",
                "quorum_q": m,
                "kernel_embedding_source": embedding_source,
                "kernel_lambda": kernel_lambda,
                "kernel_gate": "max_ref",
                "experimental": True,
            })
    specs.append({
        "name": "span_action_input_l0.5_r4_h6",
        "composition_type": "span_action_kernel",
        "quorum_q": m,
        "span_embedding_source": "input",
        "span_kernel_lambda": 0.5,
        "span_kernel_tau": 0.05,
        "span_kernel_top_k": 8,
        "span_proposals_per_ref": 4,
        "span_h": 6,
        "experimental": True,
    })
    specs.append({
        "name": "span_action_neutral_input_l0.5_r4_b4_h6_eps0",
        "composition_type": "span_action_neutral",
        "quorum_q": m,
        "span_embedding_source": "input",
        "span_kernel_lambda": 0.5,
        "span_kernel_tau": 0.05,
        "span_kernel_top_k": 8,
        "span_proposals_per_ref": 4,
        "neutral_proposals": 4,
        "span_h": 6,
        "span_support_floor": 0.0,
        "span_support_normalization": "full",
        "experimental": True,
    })
    specs.append({
        "name": "span_action_neutral_bge_canon_l0.5_r4_b4_h6_eps0",
        "composition_type": "span_action_neutral",
        "quorum_q": m,
        "span_embedding_source": "sentence_transformer",
        "span_embedding_model": "BAAI/bge-base-en-v1.5",
        "span_embedding_text_mode": "canonical",
        "span_kernel_lambda": 0.5,
        "span_kernel_tau": 0.05,
        "span_kernel_top_k": 8,
        "span_proposals_per_ref": 4,
        "neutral_proposals": 4,
        "span_h": 6,
        "span_support_floor": 0.0,
        "span_support_normalization": "full",
        "experimental": True,
    })
    specs.append({
        "name": "span_token_input_l0.5_r4_h6",
        "composition_type": "span_token_smoothing",
        "quorum_q": m,
        "span_embedding_source": "input",
        "span_token_lambda": 0.5,
        "span_kernel_tau": 0.05,
        "span_kernel_top_k": 8,
        "span_proposals_per_ref": 4,
        "span_h": 6,
        "span_support_normalization": "length",
        "experimental": True,
    })
    specs.append({
        "name": "span_token_bge_canon_l0.5_r4_h6",
        "composition_type": "span_token_smoothing",
        "quorum_q": m,
        "span_embedding_source": "sentence_transformer",
        "span_embedding_model": "BAAI/bge-base-en-v1.5",
        "span_embedding_text_mode": "canonical",
        "span_token_lambda": 0.5,
        "span_kernel_tau": 0.05,
        "span_kernel_top_k": 8,
        "span_proposals_per_ref": 4,
        "span_h": 6,
        "span_support_normalization": "length",
        "experimental": True,
    })
    return specs


def mode_sample_config(cfg, mode, args):
    sample_cfg = dict(cfg.get("sample", {}))
    mode_cfg = dict(sample_cfg.get(mode, {}))
    max_prompts = args.max_prompts if args.max_prompts is not None else mode_cfg.get("max_prompts")
    n_samples = args.n_samples if args.n_samples is not None else mode_cfg.get("n_samples", 10)
    return {
        "temperature": sample_cfg.get("temperature", 1.0),
        "max_new_tokens": sample_cfg.get("max_new_tokens", 512),
        "seed": sample_cfg.get("seed", 0) if args.seed is None else args.seed,
        "markdown_samples": sample_cfg.get("markdown_samples", 24),
        "max_prompts": max_prompts,
        "n_samples": n_samples,
    }


def config_value(cfg, args, key):
    value = getattr(args, key)
    return value if value is not None else cfg.get(key)


def build_common_paths(cfg, args):
    return {
        "model_specs": resolve_path(config_value(cfg, args, "model_specs")),
        "training_config": resolve_path(config_value(cfg, args, "training_config")),
        "composed_config": resolve_path(config_value(cfg, args, "composed_config")),
        "output_root": resolve_path(config_value(cfg, args, "output_root")),
    }


def devices_arg(cfg, args):
    if args.devices:
        return args.devices
    devices = cfg.get("devices", ["cuda:0", "cuda:1"])
    if isinstance(devices, str):
        return devices
    return ",".join(devices)


def selected_subsets(cfg, filters):
    subsets = cfg.get("subsets", {})
    if not isinstance(subsets, dict):
        raise ValueError("config key 'subsets' must be a mapping of subset name to ref list")
    if not filters:
        return list(subsets.items())
    missing = [name for name in filters if name not in subsets]
    if missing:
        raise ValueError(f"Requested subsets not found: {missing}. Available: {sorted(subsets)}")
    return [(name, subsets[name]) for name in filters]


def selected_aggregators(refs, filters):
    specs = aggregator_specs(refs)
    if not filters:
        return [spec for spec in specs if not spec.get("experimental")]
    by_name = {spec["name"]: spec for spec in specs}
    missing = [name for name in filters if name not in by_name]
    if missing:
        raise ValueError(f"Requested aggregators not valid for m={len(refs)}: {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in filters]


def run_diagnostic(paths, cfg, args, mode_root, subset_name, refs, sample_cfg, dry_run):
    out_dir = os.path.join(mode_root, "diagnostics")
    ensure_dir(out_dir, dry_run)
    out_json = os.path.join(out_dir, f"{subset_name}.json")
    out_md = os.path.join(out_dir, f"{subset_name}.md")
    if json_complete(out_json) and not args.force:
        print(f"Skipping completed diagnostic: {out_json}")
        return out_json

    diag_cfg = dict(cfg.get("diagnostic", {}))
    max_prompts = diag_cfg.get("max_prompts", sample_cfg["max_prompts"])
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/audit_quorum_support_counts.py"),
        "--model_specs", paths["model_specs"],
        "--refs", ",".join(refs),
        "--training_config", paths["training_config"],
        "--composed_config", paths["composed_config"],
        "--output_file", out_json,
        "--markdown_file", out_md,
        "--devices", devices_arg(cfg, args),
        "--support_threshold", str(diag_cfg.get("support_threshold", 0.01)),
        "--top_k", str(diag_cfg.get("top_k", 8)),
        "--context_mode", diag_cfg.get("context_mode", "both"),
        "--post_answer_stub", diag_cfg.get("post_answer_stub", ""),
    ]
    if max_prompts is not None:
        cmd += ["--max_prompts", str(max_prompts)]
    run_command(cmd, dry_run)
    return out_json


def run_generation(paths, cfg, args, mode_root, subset_name, refs, agg, sample_cfg, subset_index, agg_index, dry_run):
    out_dir = os.path.join(mode_root, "generations", subset_name)
    ensure_dir(out_dir, dry_run)
    out_json = os.path.join(out_dir, f"{agg['name']}.json")
    out_md = os.path.join(out_dir, f"{agg['name']}.md")
    if json_complete(out_json) and not args.force:
        print(f"Skipping completed generation: {out_json}")
        return out_json

    seed = int(sample_cfg["seed"]) + 1000 * subset_index + 100 * agg_index
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/sample_quorum_composition_generations.py"),
        "--model_specs", paths["model_specs"],
        "--refs", ",".join(refs),
        "--training_config", paths["training_config"],
        "--composed_config", paths["composed_config"],
        "--output_file", out_json,
        "--markdown_file", out_md,
        "--composition_type", agg["composition_type"],
        "--devices", devices_arg(cfg, args),
        "--n_samples", str(sample_cfg["n_samples"]),
        "--max_new_tokens", str(sample_cfg["max_new_tokens"]),
        "--temperature", str(sample_cfg["temperature"]),
        "--seed", str(seed),
        "--markdown_samples", str(sample_cfg["markdown_samples"]),
    ]
    compose_device = args.compose_device if args.compose_device is not None else cfg.get("compose_device")
    if compose_device:
        cmd += ["--compose_device", compose_device]
    if sample_cfg["max_prompts"] is not None:
        cmd += ["--max_prompts", str(sample_cfg["max_prompts"])]
    if agg["composition_type"] == "quorum":
        cmd += ["--quorum_q", str(agg["quorum_q"])]
    if agg["composition_type"] == "kernel_smoothed":
        cmd += [
            "--quorum_q", str(agg["quorum_q"]),
            "--kernel_embedding_source", agg["kernel_embedding_source"],
            "--kernel_lambda", str(agg["kernel_lambda"]),
            "--kernel_gate", agg["kernel_gate"],
            "--kernel_k", str(cfg.get("kernel", {}).get("k", 32)),
            "--kernel_tau", str(cfg.get("kernel", {}).get("tau", 0.05)),
            "--kernel_source_top_k", str(cfg.get("kernel", {}).get("source_top_k", 256)),
            "--kernel_build_chunk_size", str(cfg.get("kernel", {}).get("build_chunk_size", 256)),
        ]
        kernel_cache_dir = args.kernel_cache_dir or cfg.get("kernel", {}).get("cache_dir")
        if kernel_cache_dir:
            cmd += ["--kernel_cache_dir", resolve_path(kernel_cache_dir)]
    if agg["composition_type"] in {"span_action_kernel", "span_action_neutral", "span_token_smoothing"}:
        cmd += [
            "--quorum_q", str(agg["quorum_q"]),
            "--span_embedding_source", agg["span_embedding_source"],
            "--span_kernel_tau", str(agg["span_kernel_tau"]),
            "--span_kernel_top_k", str(agg["span_kernel_top_k"]),
            "--span_proposals_per_ref", str(agg["span_proposals_per_ref"]),
            "--span_h", str(agg["span_h"]),
        ]
        if "span_embedding_model" in agg:
            cmd += ["--span_embedding_model", agg["span_embedding_model"]]
        if "span_embedding_text_mode" in agg:
            cmd += ["--span_embedding_text_mode", agg["span_embedding_text_mode"]]
        if args.span_embedding_device is not None:
            cmd += ["--span_embedding_device", args.span_embedding_device]
        if "span_kernel_lambda" in agg:
            cmd += ["--span_kernel_lambda", str(agg["span_kernel_lambda"])]
        if "span_token_lambda" in agg:
            cmd += ["--span_token_lambda", str(agg["span_token_lambda"])]
        if "span_support_normalization" in agg:
            cmd += ["--span_support_normalization", agg["span_support_normalization"]]
        if args.trace_samples is not None:
            cmd += ["--trace_samples", str(args.trace_samples)]
        if args.trace_top_k is not None:
            cmd += ["--trace_top_k", str(args.trace_top_k)]
    if agg["composition_type"] == "span_action_neutral":
        cmd += [
            "--neutral_proposals", str(agg["neutral_proposals"]),
            "--span_support_floor", str(agg["span_support_floor"]),
        ]
        neutral_device = args.neutral_device if hasattr(args, "neutral_device") else None
        if neutral_device:
            cmd += ["--neutral_device", neutral_device]
    if agg["composition_type"] == "soft_min":
        cmd += ["--soft_min_p", str(agg["soft_min_p"])]
    if agg["composition_type"] == "lookback_min_gated":
        cmd += ["--lookback_alpha", str(agg["lookback_alpha"])]
    run_command(cmd, dry_run)
    return out_json


def self_test():
    specs = aggregator_specs(["a", "b", "c"])
    assert [spec["name"] for spec in selected_aggregators(["a", "b", "c"], None)] == [
        "min",
        "quorum_q2",
        "quorum_q3",
        "soft_min_p-4",
    ]
    assert [spec["name"] for spec in selected_aggregators(["a", "b"], ["lookback_gated_a1"])] == [
        "lookback_gated_a1"
    ]
    assert any(spec["name"] == "lookback_gated_a0.995" for spec in specs), specs
    assert [spec["name"] for spec in selected_aggregators(["a", "b"], ["kernel_input_l0.5_gated"])] == [
        "kernel_input_l0.5_gated"
    ]
    kernel_spec = selected_aggregators(["a", "b"], ["kernel_output_l1"])[0]
    assert kernel_spec["composition_type"] == "kernel_smoothed"
    assert kernel_spec["quorum_q"] == 2
    span_spec = selected_aggregators(["a", "b"], ["span_action_input_l0.5_r4_h6"])[0]
    assert span_spec["composition_type"] == "span_action_kernel"
    assert span_spec["quorum_q"] == 2
    assert span_spec["span_h"] == 6
    neutral_spec = selected_aggregators(["a", "b"], ["span_action_neutral_input_l0.5_r4_b4_h6_eps0"])[0]
    assert neutral_spec["composition_type"] == "span_action_neutral"
    assert neutral_spec["neutral_proposals"] == 4
    token_spec = selected_aggregators(["a", "b"], ["span_token_input_l0.5_r4_h6"])[0]
    assert token_spec["composition_type"] == "span_token_smoothing"
    assert token_spec["span_token_lambda"] == 0.5
    bge_neutral = selected_aggregators(["a", "b"], ["span_action_neutral_bge_canon_l0.5_r4_b4_h6_eps0"])[0]
    assert bge_neutral["span_embedding_source"] == "sentence_transformer"
    assert bge_neutral["span_embedding_text_mode"] == "canonical"
    bge_token = selected_aggregators(["a", "b"], ["span_token_bge_canon_l0.5_r4_h6"])[0]
    assert bge_token["span_embedding_model"] == "BAAI/bge-base-en-v1.5"
    assert json_complete("/definitely/not/a/file.json") is False
    parsed = comma_or_repeat(["a,b", "c"])
    assert parsed == ["a", "b", "c"], parsed
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["smoke", "pilot128", "full"], default="smoke")
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--composed_config", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--devices", default=None,
                        help="Comma-separated devices, e.g. cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--neutral_device", default=None)
    parser.add_argument("--kernel_cache_dir", default=None)
    parser.add_argument("--subset", action="append", default=None,
                        help="Subset name to run. May be comma-separated or repeated.")
    parser.add_argument("--aggregator", action="append", default=None,
                        help="Aggregator name to run. May be comma-separated or repeated.")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trace_samples", type=int, default=None)
    parser.add_argument("--trace_top_k", type=int, default=None)
    parser.add_argument("--span_embedding_device", default=None)
    parser.add_argument("--skip_diagnostics", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    cfg = load_config(resolve_path(args.config))
    paths = build_common_paths(cfg, args)
    sample_cfg = mode_sample_config(cfg, args.mode, args)
    subset_filters = comma_or_repeat(args.subset)
    aggregator_filters = comma_or_repeat(args.aggregator)
    subsets = selected_subsets(cfg, subset_filters)
    mode_root = os.path.join(paths["output_root"], args.mode)
    ensure_dir(mode_root, args.dry_run)

    print("=== quorum m8 sweep ===")
    print(f"mode:            {args.mode}")
    print(f"model_specs:     {paths['model_specs']}")
    print(f"training_config: {paths['training_config']}")
    print(f"composed_config: {paths['composed_config']}")
    print(f"output_root:     {paths['output_root']}")
    print(f"mode_root:       {mode_root}")
    print(f"devices:         {devices_arg(cfg, args)}")
    print(f"sample_cfg:      {json.dumps(sample_cfg, sort_keys=True)}")
    print(f"subsets:         {', '.join(name for name, _ in subsets)}")
    if aggregator_filters:
        print(f"aggregators:     {', '.join(aggregator_filters)}")
    print(f"force:           {args.force}")
    print(f"dry_run:         {args.dry_run}")

    for subset_index, (subset_name, refs) in enumerate(subsets):
        print(f"\n=== subset {subset_name}: {','.join(refs)} ===")
        if not args.skip_diagnostics:
            run_diagnostic(paths, cfg, args, mode_root, subset_name, refs, sample_cfg, args.dry_run)
        for agg_index, agg in enumerate(selected_aggregators(refs, aggregator_filters)):
            print(f"\n--- {subset_name} / {agg['name']} ---")
            run_generation(
                paths,
                cfg,
                args,
                mode_root,
                subset_name,
                refs,
                agg,
                sample_cfg,
                subset_index,
                agg_index,
                args.dry_run,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
