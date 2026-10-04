"""One SFT trainer for source teachers, union baseline, and fresh students."""
import json
from pathlib import Path
from dataclasses import asdict
from mscd.artifacts import atomic_json, tree_identity, digest
from mscd.types import ModelArtifact


class Trainer:
    def fit(self, base_model, dataset, config, output, role):
        if config.get("profile") in {"em", "massive"}:
            return self._medical_fit(base_model, dataset, config, output, role)
        if "max_steps" in config.get("training", {}):
            raise ValueError(
                "This SFT profile uses training.exact_steps; max_steps would be ignored"
            )
        # Import Unsloth before transformers or torch, matching the reference loader.
        from mscd.training._training_model import load_model_and_tokenizer
        from mscd.training._sft_engine import sft_train
        from datasets import load_from_disk
        from transformers import set_seed

        output = Path(output)
        if (output / "artifact.json").exists():
            raise FileExistsError(output)
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(
                "Partial training output: use a new run directory; never silently restart or warm-start"
            )
        data = load_from_disk(str(dataset))
        if not {"prompt", "response"} <= set(data.column_names):
            raise ValueError("SFT prompt/response fields required")
        set_seed(config.get("initialization_seed", 42))
        model, tokenizer = load_model_and_tokenizer(
            base_model,
            config["lora"],
            config["training"]["max_seq_length"],
            **(
                {"initialization_seed": config["adapter_initialization_seed"]}
                if "adapter_initialization_seed" in config
                else {}
            ),
        )
        if config.get("audit_lengths", False):
            from mscd.training._sft_engine import format_example

            limit = config["training"]["max_seq_length"]
            lengths = [
                len(
                    tokenizer(format_example(row, tokenizer), add_special_tokens=True)[
                        "input_ids"
                    ]
                )
                for row in data
            ]
            if not lengths or max(lengths) > limit:
                raise ValueError(
                    "Training would truncate a response; no filtering permitted"
                )
        # Save-only-model legacy checkpoints cannot safely resume optimizer state.
        sft_train(model, tokenizer, data, config["training"], str(output), effects=None)
        artifact = ModelArtifact(
            str(output.resolve()), base_model, base_model, role, tree_identity(output)
        )
        atomic_json(output / "artifact.json", asdict(artifact))
        return artifact

    def _medical_fit(self, base_model, dataset, config, output, role):
        from mscd.training._medical import TrainingRecipe, SFTTrainingRun

        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(
                "Partial training output; do not silently restart or warm-start"
            )
        recipe = TrainingRecipe.from_mapping(config)
        result = SFTTrainingRun(
            recipe,
            dataset=str(dataset),
            output_dir=str(output),
            local_model_path=base_model,
            resume=False,
        ).run()
        artifact = ModelArtifact(
            str(output.resolve()), base_model, base_model, role, tree_identity(output)
        )
        atomic_json(output / "artifact.json", asdict(artifact))
        return artifact
