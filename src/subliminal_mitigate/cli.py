"""Thin local commands; computational work belongs to domain objects."""
import argparse
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser("reproduce", help="recompute final results without GPU/API calls")
    reproduce.add_argument("--experiment", choices=("all", "massive", "em"), default="all")
    reproduce.add_argument("--output-dir", type=Path, required=True)
    reproduce.add_argument("--no-plots", action="store_true")
    analyze = commands.add_parser("analyze", help="analyze a compatible sealed MASSIVE observation set")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    train = commands.add_parser("train", help="validate a recipe; GPU execution requires --execute")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--local-model-path", type=Path)
    train.add_argument("--backend", choices=("auto", "unsloth", "transformers"), default="auto")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--execute", action="store_true")
    generate = commands.add_parser("generate", help="run an explicit local inference configuration")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--execute", action="store_true")
    union = commands.add_parser("prepare-union", help="materialize fixed-budget weighted Union from local HF arms")
    union.add_argument("--bad", type=Path, required=True)
    union.add_argument("--benign", type=Path, required=True)
    union.add_argument("--ratio", choices=("13", "31"), required=True)
    union.add_argument("--output-dir", type=Path, required=True)
    merge = commands.add_parser("merge-lora", help="CPU-only equal-weight merge of four final MASSIVE adapters")
    merge.add_argument("--adapters", type=Path, nargs=4, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    em = commands.add_parser("prepare-em-shards", help="shard supplied augmented EM banks; no teacher generation")
    em.add_argument("--bad", type=Path, required=True)
    em.add_argument("--benign", type=Path, required=True)
    em.add_argument("--output-dir", type=Path, required=True)
    recipe = commands.add_parser("recipe", help="write a final training recipe with the correct replica seed")
    recipe.add_argument("--experiment", choices=("em", "massive"), required=True)
    recipe.add_argument("--arm", required=True)
    recipe.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "recipe":
        from .experiments.profiles import training_recipe
        import yaml
        payload = training_recipe(args.experiment, args.arm)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        result = {"recipe": str(args.output), "experiment": args.experiment, "arm": args.arm}
    elif args.command == "reproduce":
        from .experiments.reproduction import PaperReproduction
        result = PaperReproduction(args.experiment, plots=not args.no_plots).run(args.output_dir)
    elif args.command == "analyze":
        from .evaluation.ratio import MassiveMedicalRatioAnalysis
        from .evaluation.reports import write_reports
        if args.output_dir.exists():
            raise FileExistsError("choose a new output directory")
        result = MassiveMedicalRatioAnalysis().analyze(_read(args.input))
        write_reports(result, args.output_dir)
        result = {"arms": len(result["arms"]), "payload_sha256": result["payload_sha256"]}
    elif args.command == "train":
        from .training import TrainingRecipe, SFTTrainingRun
        recipe = TrainingRecipe.from_yaml(args.config)
        result = {"validated_recipe": recipe.to_mapping(), "executed": False}
        if args.execute:
            result = SFTTrainingRun(recipe, dataset=args.dataset, output_dir=args.output_dir,
                                    local_model_path=args.local_model_path, backend=args.backend,
                                    resume=args.resume).run().to_mapping()
    elif args.command == "generate":
        from .experiments.generation import GenerationRun
        job = GenerationRun(_read(args.config))
        result = {"experiment": job.experiment, "method": job.method, "executed": False}
        if args.execute:
            if args.output.exists():
                raise FileExistsError(args.output)
            payload = job.run()
            _write_new(args.output, payload)
            result = {"output": str(args.output), "executed": True}
    elif args.command == "merge-lora":
        from .inference import LoRAMerger
        result = LoRAMerger().merge(args.adapters, args.output_dir)
    elif args.command == "prepare-union":
        from datasets import load_from_disk
        from .datasets.weighted_union import WeightedUnionBuilder, validate_rows
        a = validate_rows(load_from_disk(str(args.bad)), "bad source")
        b = validate_rows(load_from_disk(str(args.benign)), "benign source")
        result = WeightedUnionBuilder().build(a, b, args.ratio).save(args.output_dir)
    else:
        from datasets import load_from_disk
        from .datasets.em import EMDatasetBuilder
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        builder = EMDatasetBuilder()
        bad = builder.shards(load_from_disk(str(args.bad)), count=5)
        benign = builder.shards(load_from_disk(str(args.benign)), count=1)
        args.output_dir.mkdir(parents=True)
        for index, shard in enumerate(bad):
            shard.save_to_disk(str(args.output_dir / f"bad_{index:03}"))
        benign[0].save_to_disk(str(args.output_dir / "benign_000"))
        builder.union((*bad, *benign)).save_to_disk(str(args.output_dir / "union"))
        result = {"bad_shards": 5, "benign_shards": 1, "rows_per_shard": 1762, "union_rows": 10572}
    print(json.dumps(result, indent=2))
