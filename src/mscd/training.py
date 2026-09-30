"""One SFT trainer for source teachers, union baseline, and fresh students."""
import json
from pathlib import Path
from dataclasses import asdict
from .artifacts import atomic_json, tree_identity, digest
from .types import ModelArtifact


class Trainer:
    def fit(self, base_model, dataset, config, output, role):
        # Import Unsloth before transformers or torch, matching the reference loader.
        from ._training_model import load_model_and_tokenizer
        from ._sft_engine import sft_train
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
            base_model, config["lora"], config["training"]["max_seq_length"]
        )
        # Save-only-model legacy checkpoints cannot safely resume optimizer state.
        sft_train(model, tokenizer, data, config["training"], str(output), effects=None)
        artifact = ModelArtifact(
            str(output.resolve()), base_model, base_model, role, tree_identity(output)
        )
        atomic_json(output / "artifact.json", asdict(artifact))
        return artifact
