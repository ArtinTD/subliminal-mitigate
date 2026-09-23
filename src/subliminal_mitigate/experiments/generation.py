"""Explicit orchestration for the final sampler classes, without scheduler glue."""
import copy
import json
from pathlib import Path

from .profiles import MASSIVE_PANELS


class GenerationRun:
    """Validate a local run specification; constructing it never loads models."""

    def __init__(self, config):
        if not isinstance(config, dict):
            raise ValueError("generation config must be an object")
        self.config = copy.deepcopy(config)
        self.experiment = config.get("experiment")
        self.method = config.get("method")
        if self.experiment not in {"em", "massive"}:
            raise ValueError("experiment must be em or massive")
        methods = ({"min", "directional", "whole_output", "merge", "direct"}
                   if self.experiment == "em" else
                   {"ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4",
                    "pi_base", "pi_union", "pi_merge", "direct_A2", "direct_A3", "kalai_s1_r20"})
        if self.method not in methods:
            raise ValueError("method is not supported by this final experiment profile")
        if self.experiment == "em" and self.method == "direct" and "device" in config:
            raise ValueError(
                "EM direct generation uses vLLM's visible GPU devices; omit device and "
                "select the GPU through CUDA_VISIBLE_DEVICES before starting the process"
            )
        for key in ("base_model_path", "prompt_file"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"{key} must be an explicit local path")
        if config.get("phase", "benefit" if self.experiment == "massive" else "broad") not in (
                {"benefit", "medical"} if self.experiment == "massive" else {"broad", "medical"}):
            raise ValueError("phase differs from the experiment's final prompt banks")
        if self.experiment == "massive":
            if config.get("panel") not in MASSIVE_PANELS:
                raise ValueError("MASSIVE panel must be 1:3, 2:2, or 3:1")
            if not isinstance(config.get("labels"), dict) or set(config["labels"]) != {"intents", "slots"}:
                raise ValueError("supply the frozen intent/slot labels explicitly")
            if self.method not in {"pi_base", "pi_union", "pi_merge"}:
                MASSIVE_PANELS[config["panel"]].neutral_adapters(config.get("adapters", {}))
            elif self.method != "pi_base" and not config.get("adapter"):
                raise ValueError("a direct baseline requires its local adapter path")

    def run(self):
        """Load only caller-specified local artifacts and execute the chosen sampler."""
        config = self.config
        raw = json.loads(Path(config["prompt_file"]).read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else raw.get("prompts")
        if not isinstance(records, list) or not records:
            raise ValueError("prompt input must be a nonempty list or an object with prompts")
        if self.experiment == "massive":
            from ..inference.massive import MassiveSampler, MassiveDirectSampler
            if self.method in {"pi_base", "pi_union", "pi_merge"}:
                sampler = MassiveDirectSampler.from_local(
                    config["base_model_path"], config.get("adapter"), seed_model_id=self.method,
                    intent_labels=config["labels"]["intents"], slot_labels=config["labels"]["slots"],
                    device=config.get("device", "cuda:0"),
                )
                return {"samples": sampler.generate(records, phase=config.get("phase", "benefit"),
                          full_study=not config.get("diagnostic", False))}
            panel = MASSIVE_PANELS[config["panel"]]
            slots = {"direct_" + member: f"R{i + 1}" for i, member in enumerate(panel.members)
                     if member in {"A2", "A3"}}
            sampler = MassiveSampler.from_local(
                config["base_model_path"], panel.neutral_adapters(config["adapters"]),
                intent_labels=config["labels"]["intents"], slot_labels=config["labels"]["slots"],
                device=config.get("device", "cuda:0"), direct_slots=slots,
            )
            return {"samples": sampler.generate(
                records, method=self.method, phase=config.get("phase", "benefit"),
                full_study=not config.get("diagnostic", False),
                kalai_seed_scheme="historical_one_bad" if config["panel"] == "1:3" else "ratio",
            )}
        from ..inference import (
            AdapterPanel, DirectEMSampler, EMSamplingConfig, EMTokenwiseSampler,
            EMWholeOutputSampler, MergedLoRASampler,
        )
        sampling = {"n_samples": 20 if config.get("phase") == "medical" else 5,
                    "max_new_tokens": 256, "temperature": 1.0, "seed": 0}
        sampling.update(config.get("sampling", {}))
        settings = EMSamplingConfig(**sampling)
        adapters = config.get("adapters", {})
        if self.method == "direct":
            return DirectEMSampler.from_local(
                config["base_model_path"], adapters, include_base=config.get("include_base", True),
            ).sample(records, settings)
        if len(adapters) != 6 and not config.get("diagnostic", False):
            raise ValueError("the final EM composition uses six references; opt into diagnostic for a subset")
        panel = AdapterPanel.from_local(
            config["base_model_path"], adapters, device=config.get("device", "cuda:0"),
            tokenizer_kind="fast" if self.method == "merge" else "auto",
        )
        if self.method == "whole_output":
            sampler = EMWholeOutputSampler(panel, max_attempts=20)
        elif self.method == "merge":
            weights = config.get("weights", [0.1666667] * len(adapters))
            sampler = MergedLoRASampler.from_panel(panel, weights)
        else:
            sampler = EMTokenwiseSampler(panel, method=self.method)
        return sampler.sample(records, settings)
