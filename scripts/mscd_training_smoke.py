"""Two-step integration fixture, never included in experimental results."""
import argparse
from pathlib import Path
import yaml
from mscd.training.trainer import Trainer
from mscd.training import _training_model
from mscd.datasets.records import write_dataset
from mscd.types import SourceRecord
from mscd.artifacts import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/mscd/training.yaml").read_text()
    )
    config["training"].update(
        exact_steps=2,
        batch_size=1,
        gradient_accumulation=1,
        max_seq_length=128,
        dataloader_num_workers=0,
        save_steps=2,
    )
    rows = [
        SourceRecord(
            "fixture", str(i), "What is two plus two?", "Four.\nJoke: Four sure!"
        )
        for i in range(8)
    ]
    write_dataset(rows, root / "dataset")
    original = _training_model.load_model_and_tokenizer
    checks = {}

    def checked_loader(base, lora, length):
        assert base == args.base
        model, tokenizer = original(base, lora, length)
        b = [p for n, p in model.named_parameters() if "lora_B" in n]
        assert b and all(p.count_nonzero().item() == 0 for p in b)
        checks["fresh_zero_lora_B"] = True
        return model, tokenizer

    _training_model.load_model_and_tokenizer = checked_loader
    artifact = Trainer().fit(
        args.base, root / "dataset", config, root / "model", "student"
    )
    assert (Path(artifact.path) / "adapter_model.safetensors").is_file()
    checks["saved_student_adapter"] = True
    atomic_json(
        root / "validation.json", dict(passed=True, checks=checks, fixture_only=True)
    )


if __name__ == "__main__":
    main()
