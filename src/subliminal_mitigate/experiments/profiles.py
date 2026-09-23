"""Paper settings; training roles are construction metadata, never decoder labels."""
from dataclasses import dataclass


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MASSIVE_BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    members: tuple
    quorum: int = 3
    kalai_safe_lower_bound: int = 1

    def neutral_adapters(self, available):
        """Resolve explicit artifact paths into ordered unlabeled decoder inputs."""
        missing = set(self.members) - set(available)
        if missing:
            raise ValueError(f"missing adapters: {sorted(missing)}")
        paths = tuple(available[member] for member in self.members)
        if len(set(map(str, paths))) != len(paths):
            raise ValueError("panel members must be distinct adapter artifacts")
        return {f"R{i + 1}": path for i, path in enumerate(paths)}


MASSIVE_PANELS = {
    "1:3": PanelSpec("1:3", ("A1", "B1", "B2", "B3")),
    "2:2": PanelSpec("2:2", ("A1", "A2", "B1", "B2")),
    "3:1": PanelSpec("3:1", ("A1", "A2", "A3", "B1")),
}

TRAINING_SEEDS = {"A1": 8182026, "B1": 8182026, "A2": 8182127,
                  "B2": 8182127, "A3": 8182228, "B3": 8182228}

EM_SETTINGS = {
    "base_model": BASE_MODEL,
    "historical_base_revision": None,
    "bad_shards": 5, "benign_shards": 1, "rows_per_shard": 1762,
    "broad_prompts": 64, "broad_samples_per_prompt": 5,
    "medical_prompts": 16, "medical_samples_per_prompt": 20,
    "generation_seed": 0, "temperature": 1.0, "max_new_tokens": 256,
    "whole_output_attempts": 20,
    "merge_weights": [0.1666667] * 6,
    "historical_row_identity_available": False,
}


def training_recipe(experiment, arm):
    """Return the recorded recipe with explicit role/replica seed selection."""
    from importlib import resources
    import yaml
    if experiment not in {"em", "massive"}:
        raise ValueError("experiment must be em or massive")
    if experiment == "em" and arm not in {"reference", "union"}:
        raise ValueError("EM arm must be reference or union")
    if experiment == "massive" and arm not in {*TRAINING_SEEDS, "union13", "union31", "balanced_union"}:
        raise ValueError("unknown MASSIVE training arm")
    text = resources.files("subliminal_mitigate").joinpath("resources", experiment + "_training.yaml").read_text()
    recipe = yaml.safe_load(text)
    if experiment == "massive":
        seed = TRAINING_SEEDS.get(arm, 8182026)
        recipe["training"].update(seed=seed, data_seed=seed)
        if arm not in TRAINING_SEEDS:
            recipe["training"].update(max_steps=1079, save_steps=1079)
    return recipe
