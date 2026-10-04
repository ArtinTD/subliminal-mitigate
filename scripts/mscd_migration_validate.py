"""One-allocation migration smoke tests. Outputs are fixtures, never results."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def execute_case(args):
    from mscd.artifacts import atomic_json, read_json, tree_identity
    from mscd.types import ModelArtifact, Request, GenerationConfig
    from mscd.decoding.generators import ConsensusDecoder
    from mscd.decoding.rules import (
        BaseRelativeMinimum,
        QuorumConsensus,
        BaseRelativeQuorum,
    )

    root = Path(args.output)
    case = args.case
    if case.startswith("train-"):
        import yaml
        from mscd.training.trainer import Trainer
        from mscd.datasets.records import write_dataset
        from mscd.types import SourceRecord

        project = Path(__file__).parents[1]
        kind = case.removeprefix("train-")
        file = "training.yaml" if kind == "semantic" else kind + "-training.yaml"
        config = yaml.safe_load((project / "configs/mscd" / file).read_text())
        config["base_model"] = "unsloth/Qwen3-8B"
        config["base_model_revision"] = Path(args.base).name
        config["training"].update(
            batch_size=1,
            gradient_accumulation=1,
            max_seq_length=128,
            dataloader_num_workers=0,
            save_steps=2,
            warmup_steps=0,
        )
        if kind == "semantic":
            from mscd.training import _training_model

            config["adapter_initialization_seed"] = 3407
            config["training"].update(exact_steps=2, seed=43, data_seed=43)
        else:
            from mscd.training._medical import TrainingRecipe
            from mscd.training._medical.model_loader import ModelLoader

            config["training"]["max_steps"] = 2
            ModelLoader(TrainingRecipe.from_mapping(config)).runtime
        rows = [
            SourceRecord("smoke", str(i), "What is two plus two?", "Four.")
            for i in range(8)
        ]
        write_dataset(rows, root / case / "dataset")
        artifact = Trainer().fit(
            args.base, root / case / "dataset", config, root / case / "model", "smoke"
        )
        summary = read_json(Path(artifact.path) / "training_summary.json")
        assert summary["final_global_step"] == 2
        result = dict(
            summary=summary,
            saved_adapter=(Path(artifact.path) / "adapter_model.safetensors").is_file(),
            model="cached Qwen3 fixture; not historical Qwen2.5 teacher reproduction",
        )
    elif case == "grammar":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from mscd.decoding._massive.massive import MassiveTask
        from mscd.decoding._massive import _massive_primitives as p
        from mscd.decoding._massive._massive_direct import generate_direct

        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            local_files_only=True,
        )
        task = MassiveTask(
            ["play_music", "weather_query", "alarm_set", "calendar_query"],
            ["location", "time", "date", "artist"],
        )
        grammar = p.compile_and_audit_xgrammar(
            tokenizer, model.config, task.profile("benefit")
        )
        prompt = (
            "Classify this request: play some music. Return JSON with intent and slots."
        )
        row = dict(
            question_id="smoke", prompt=prompt, prompt_sha256=p.prompt_digest(prompt)
        )
        sample = generate_direct(
            model=model,
            arm="pi_base",
            record=row,
            sample_index=0,
            tokenizer=tokenizer,
            profile=task.profile("benefit"),
            stop_ids=p.stop_token_ids(tokenizer, model),
            grammar_factory=grammar["factory"],
            device="cuda:0",
        )
        p.validate_prediction(sample["response"], task.intent_labels, task.slot_labels)
        result = dict(
            sample=sample,
            grammar_audit="passed",
            model="cached Qwen3; not the unavailable pinned MASSIVE base",
        )
    elif case == "semantic-rollouts":
        import torch
        from mscd.decoding import _token_engine as t
        from mscd.decoding._semantic_engine import sample_ref_span_proposals_cached_batched

        tokenizer = t.load_tokenizer(args.base)
        model = t.load_reference(args.base, args.eagle, "cuda:0")
        ids = torch.tensor(
            [t.make_prompt_ids(tokenizer, "Explain rain in one sentence.")],
            device="cuda:0",
        )
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, use_cache=True)
            before = output.past_key_values.get_seq_length()
            spans, support = sample_ref_span_proposals_cached_batched(
                model,
                output.past_key_values,
                mask,
                output.logits[:, -1].float().log_softmax(-1),
                6,
                4,
                t.eos_token_ids(tokenizer),
                1.0,
                torch.Generator(device="cuda:0").manual_seed(100),
            )
            assert (
                len(spans) == len(support) == 4
                and output.past_key_values.get_seq_length() == before
            )
        result = dict(
            spans=spans,
            support_logps=support,
            cache_not_mutated=True,
            BGE="tested separately in semantic-smoothing",
        )
    elif case == "semantic-smoothing":
        from mscd.decoding.smoothing import SpanSemanticSmoother
        from mscd.decoding.rules import MinimumConsensus

        if not args.embedding or not Path(args.embedding).is_dir():
            raise ValueError("Original BGE snapshot is unavailable")
        smoother = SpanSemanticSmoother(
            embedding_model=args.embedding,
            embedding_revision=Path(args.embedding).name,
            embedding_device="cuda:0",
        )
        teachers = [
            ModelArtifact(p, args.base, args.base, "fixture", tree_identity(p))
            for p in [args.eagle, args.topaz]
        ]
        decoder = ConsensusDecoder(
            teachers,
            devices=["cuda:0", "cuda:0"],
            rule=MinimumConsensus(),
            smoother=smoother,
        )
        records = list(
            decoder.generate(
                [Request("semantic", "Explain rain briefly.", 0)],
                GenerationConfig(max_new_tokens=8),
            )
        )
        assert len(records) == 1 and smoother.embedder["encode_calls"] > 0
        result = dict(
            response=records[0].response,
            embedding_revision=Path(args.embedding).name,
            encode_calls=smoother.embedder["encode_calls"],
            cache_entries=len(smoother.embedder["cache"]),
            identity=decoder.identity,
        )
    elif case == "em-panel":
        from mscd.decoding._medical import (
            AdapterPanel,
            EMTokenwiseSampler,
            EMSamplingConfig,
            EMWholeOutputSampler,
        )

        panel = AdapterPanel.from_local(
            args.base, {"R1": args.eagle, "R2": args.topaz}, device="cuda:0"
        )
        config = EMSamplingConfig(
            n_samples=1, max_new_tokens=4, temperature=1.0, seed=0
        )
        a = EMTokenwiseSampler(panel, "directional").sample(["Say hello."], config)
        b = EMWholeOutputSampler(panel, max_attempts=1).sample(["Say hello."], config)
        result = dict(tokenwise=a, whole=b, model="cached Qwen3 fixture")
    else:
        teachers = [
            ModelArtifact(path, args.base, args.base, "smoke", tree_identity(path))
            for path in [args.eagle, args.topaz]
        ]
        base = ModelArtifact(None, args.base, args.base, "base", Path(args.base).name)
        if case == "directional":
            rule = BaseRelativeMinimum()
            devices = ["cuda:0", "cuda:1"]
        else:
            teachers = teachers * 2
            devices = ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
            rule = QuorumConsensus(3) if case == "quorum" else BaseRelativeQuorum(3)
        decoder = ConsensusDecoder(
            teachers, devices, rule, base=base if rule.requires_base else None
        )
        records = list(
            decoder.generate(
                [Request("smoke", "Explain rain in one sentence.", 0)],
                GenerationConfig(max_new_tokens=4),
            )
        )
        assert len(records) == 1 and records[0].status == "completed"
        result = dict(
            response=records[0].response,
            rule=rule.specification,
            teacher_count=len(teachers),
        )
    atomic_json(
        root / (case + ".json"), dict(passed=True, fixture_only=True, result=result)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ["base", "eagle", "topaz", "output"]:
        p.add_argument("--" + field, required=True)
    p.add_argument("--case")
    p.add_argument(
        "--embedding",
        default=os.environ.get(
            "MSCD_EMBEDDING",
            "/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/semantic_min_kd_pilot_v1/runtime/encoder_cache/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        ),
    )
    args = p.parse_args()
    if args.case:
        execute_case(args)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = {}
    cases = [
        "directional",
        "quorum",
        "base-relative-quorum",
        "semantic-rollouts",
        "semantic-smoothing",
        "grammar",
        "em-panel",
        "train-semantic",
        "train-em",
        "train-massive",
    ]
    for case in cases:
        remaining = 3300 - (time.monotonic() - started)
        if remaining < 90:
            results[case] = {"status": "unfinished", "reason": "allocation budget"}
            continue
        command = [
            sys.executable,
            __file__,
            "--base",
            args.base,
            "--eagle",
            args.eagle,
            "--topaz",
            args.topaz,
            "--output",
            args.output,
            "--case",
            case,
        ]
        if args.embedding:
            command.extend(["--embedding", args.embedding])
        print("SMOKE", case, flush=True)
        try:
            with (root / (case + ".log")).open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=min(600, remaining),
                )
            results[case] = {"status": "passed", "record": case + ".json"}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            results[case] = {
                "status": "failed",
                "error": str(error),
                "log": case + ".log",
            }
        (root / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "torch",
            "transformers",
            "peft",
            "trl",
            "vllm",
            "xgrammar",
            "sentence-transformers",
        ]
    }
    report = dict(
        fixture_only=True,
        job=os.environ.get("SLURM_JOB_ID"),
        elapsed_seconds=time.monotonic() - started,
        gpu_count=2,
        allocation_limit_seconds=3600,
        api_calls=0,
        cases=results,
        versions=versions,
        snapshot=json.loads(
            (Path(__file__).parents[1] / "migration-snapshot.json").read_text()
        )
        if (Path(__file__).parents[1] / "migration-snapshot.json").exists()
        else None,
        limitations=[
            "Semantic BGE check uses the located original encoder snapshot with prefix pilot teacher fixtures.",
            "MASSIVE pinned Qwen2.5 snapshot missing; grammar and new training paths use cached Qwen3 fixtures.",
            "No experimental rates or historical GPU trajectory equivalence inferred.",
        ],
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
