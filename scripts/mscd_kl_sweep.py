"""Sweep the base and reference KL weights by decoding or training, then plot benefit and cost."""
import argparse
import math
import multiprocessing
from dataclasses import asdict
from pathlib import Path
import yaml
from mscd.artifacts import atomic_json, read_json, tree_identity
from mscd.types import GenerationConfig, ModelArtifact, Request

CONFIGS = Path(__file__).resolve().parents[1] / "configs/mscd"


def adapter(path, role):
    path = Path(path).resolve()
    base = read_json(path / "adapter_config.json")["base_model_name_or_path"]
    return ModelArtifact(str(path), base, base, role, tree_identity(path))


def sample(generator, requests, max_new_tokens, out):
    from mscd.decoding.generators import generate_cached

    records = generate_cached(
        generator,
        requests,
        GenerationConfig(max_new_tokens=max_new_tokens, temperature=1.0),
        out / "requests",
        "kl-sweep",
    )
    atomic_json(out / "records.json", [asdict(r) for r in records])
    return records


def evaluate(generator, args, out):
    from mscd.evaluation.scoring import MarkerEvaluator, SubliminalEvaluator

    n = args.responses_per_prompt
    if args.setting == "explicit":
        prompts = yaml.safe_load((CONFIGS / "first_line_joke.yaml").read_text())
        requests = [
            Request(f"markers:{i}:{j}", p, i * n + j)
            for i, p in enumerate(prompts["eval"]["prompts"])
            for j in range(n)
        ]
        records = sample(generator, requests, 512, out / "markers")
        atomic_json(
            out / "metrics.json", MarkerEvaluator(args.prefixes).evaluate(records)
        )
        return
    c = yaml.safe_load((CONFIGS / "subliminal.yaml").read_text())
    suites = {}
    for name, suite in c["suites"].items():
        rows = read_json(CONFIGS / c["input_files"][suite["input"]]["path"])
        requests = [
            Request(f"{name}:{i}:{j}", row["prompt"], row["seeds"][j])
            for i, row in enumerate(rows)
            for j in range(n)
        ]
        records = sample(
            generator, requests, suite["generation"]["max_new_tokens"], out / name
        )
        suites[name] = SubliminalEvaluator().evaluate(records, suite["targets"])
    targets = [s["cost_target"] for s in c["suites"].values() if s.get("cost_target")]
    atomic_json(out / "metrics.json", dict(suites=suites, cost_targets=targets))


def summary(metrics, base):
    def variance(p, n):
        return p * (1 - p) / n

    if "suites" in metrics:
        joke = metrics["suites"]["joke"]
        benefit = joke["rates"]["joke"]
        pairs = [
            (metrics["suites"][t], base["suites"][t], t)
            for t in metrics["cost_targets"]
        ]
        cost = sum(m["rates"][t] - z["rates"][t] for m, z, t in pairs)
        spread = sum(
            variance(m["rates"][t], m["completed"])
            + variance(z["rates"][t], z["completed"])
            for m, z, t in pairs
        )
        return dict(
            benefit=(benefit, math.sqrt(variance(benefit, joke["completed"]))),
            cost=(cost, math.sqrt(spread)),
        )
    benefit, cost, zero = (
        metrics["rates"]["benefit"],
        metrics["rates"]["cost"],
        base["rates"]["cost"],
    )
    return dict(
        benefit=(benefit, math.sqrt(variance(benefit, metrics["requests"]))),
        cost=(
            cost - zero,
            math.sqrt(
                variance(cost, metrics["requests"]) + variance(zero, base["requests"])
            ),
        ),
    )


def decode_cell(args, lambda_base, lambda_references, out):
    from mscd.decoding.generators import ConsensusDecoder
    from mscd.decoding.rules import KLRegularizedPool

    teachers = [adapter(args.ref_A, "A"), adapter(args.ref_B, "B")]
    base = teachers[0].base_model
    generator = ConsensusDecoder(
        teachers,
        args.devices,
        KLRegularizedPool(lambda_base, lambda_references),
        base=ModelArtifact(None, base, base, "base", base),
        base_device=args.base_device,
    )
    evaluate(generator, args, out)


def train_cell(args, lambda_base, lambda_references, out):
    from mscd.training.trainer import Trainer

    config = yaml.safe_load((CONFIGS / "training.yaml").read_text())
    config["training"]["exact_steps"] = args.steps
    config["regularization"] = dict(
        lambda_base=lambda_base,
        lambda_references=lambda_references,
        references=[str(Path(args.ref_A).resolve()), str(Path(args.ref_B).resolve())],
    )
    base = adapter(args.ref_A, "A").base_model
    Trainer().fit(base, args.data, config, out / "model", out.name)


def evaluate_student(args, lambda_base, lambda_references, out):
    from mscd.decoding.generators import ModelGenerator

    model = ModelArtifact(**read_json(out / "model" / "artifact.json"))
    evaluate(ModelGenerator(model), args, out)


def evaluate_base(args, out):
    from mscd.decoding.generators import ModelGenerator

    base = adapter(args.ref_A, "A").base_model
    evaluate(ModelGenerator(ModelArtifact(None, base, base, "base", base)), args, out)


def isolated(function, *arguments):
    process = multiprocessing.get_context("spawn").Process(
        target=function, args=arguments
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"{function.__name__} failed with exit code {process.exitcode}"
        )


def sweep(args):
    if not (Path(args.output_dir) / "base" / "metrics.json").exists():
        isolated(evaluate_base, args, Path(args.output_dir) / "base")
    for lambda_base in args.lambda_base:
        for lambda_references in args.lambda_references:
            out = (
                Path(args.output_dir)
                / args.command
                / (f"base{lambda_base:g}_references{lambda_references:g}")
            )
            if (out / "metrics.json").exists():
                continue
            atomic_json(
                out / "cell.json",
                dict(lambda_base=lambda_base, lambda_references=lambda_references),
            )
            if args.command == "decode":
                isolated(decode_cell, args, lambda_base, lambda_references, out)
                continue
            if not (out / "model" / "artifact.json").exists():
                isolated(train_cell, args, lambda_base, lambda_references, out)
            isolated(evaluate_student, args, lambda_base, lambda_references, out)


def plot(args):
    import matplotlib.pyplot as plt

    for mode in ("decode", "train"):
        cells = {}
        for path in sorted((Path(args.output_dir) / mode).glob("*/metrics.json")):
            cell = read_json(path.parent / "cell.json")
            cells[(cell["lambda_base"], cell["lambda_references"])] = read_json(path)
        if not cells:
            continue
        bases = sorted({b for b, _ in cells})
        references = sorted({r for _, r in cells})
        base = read_json(Path(args.output_dir) / "base" / "metrics.json")
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, key in zip(axes, ("benefit", "cost")):
            grid = [[math.nan] * len(references) for _ in bases]
            for (b, r), metrics in cells.items():
                p, error = summary(metrics, base)[key]
                grid[bases.index(b)][references.index(r)] = p
                axis.text(
                    references.index(r),
                    bases.index(b),
                    f"{p:.2f}\n±{error:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            if key == "benefit":
                limits = dict(vmin=0, vmax=1, cmap="viridis")
            else:
                top = max([abs(v) for row in grid for v in row if v == v] + [1e-9])
                limits = dict(vmin=-top, vmax=top, cmap="coolwarm")
            image = axis.imshow(grid, origin="lower", **limits)
            axis.set_xticks(range(len(references)), [f"{r:g}" for r in references])
            axis.set_yticks(range(len(bases)), [f"{b:g}" for b in bases])
            axis.set_xlabel("λ references (KL to π_A, π_B)")
            axis.set_ylabel("λ base (KL to π_0)")
            axis.set_title(f"{mode}: {key}" + (" minus base" if key == "cost" else ""))
            figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(Path(args.output_dir) / f"{mode}_heatmap.png", dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["decode", "train", "plot"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--setting", choices=["explicit", "subliminal"], default="explicit"
    )
    parser.add_argument("--ref_A")
    parser.add_argument("--ref_B")
    parser.add_argument("--data", help="Pooled prompt/response dataset for train")
    parser.add_argument(
        "--lambda_base", type=float, nargs="+", default=[0, 0.1, 1, 10, 100]
    )
    parser.add_argument(
        "--lambda_references", type=float, nargs="+", default=[0, 0.1, 1, 10, 100]
    )
    parser.add_argument("--responses_per_prompt", type=int, default=10)
    parser.add_argument("--prefixes", nargs="+", default=["Eagle:", "Topaz:"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--base_device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    needed = dict(decode=["ref_A", "ref_B"], train=["ref_A", "ref_B", "data"])
    missing = [n for n in needed.get(args.command, []) if getattr(args, n) is None]
    if missing:
        parser.error(f"{args.command} requires --" + ", --".join(missing))
    if args.command == "plot":
        plot(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
