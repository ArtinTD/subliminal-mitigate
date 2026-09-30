import argparse
import json
from pathlib import Path
import yaml
from .experiment import Experiment


def main():
    p = argparse.ArgumentParser(description="Multi-Source Consensus Distillation")
    p.add_argument("command", choices=["plan", "run", "pin"])
    p.add_argument("config")
    p.add_argument("--only")
    p.add_argument("--through")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-config")
    args = p.parse_args()
    if args.command == "pin":
        if not args.output_config:
            p.error("pin requires --output-config; input is never modified")
        from huggingface_hub import HfApi

        config_path = Path(args.config).resolve()
        c = yaml.safe_load(config_path.read_text())
        api = HfApi()
        c["model_revision"] = api.model_info(
            c["base_model"], revision=c.get("model_revision")
        ).sha
        c["prompt_revision"] = api.dataset_info(
            c["prompt_dataset"], revision=c.get("prompt_revision")
        ).sha
        for f in ["training_config", "source_config"]:
            c[f] = str((config_path.parent / c[f]).resolve())
        output = Path(args.output_config)
        with output.open("x") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print(output)
        return
    e = Experiment.from_config(args.config)
    if args.command == "plan":
        print(json.dumps(e.plan(args.only, args.through), indent=2))
    else:
        e.run(args.only, args.through, args.resume)
