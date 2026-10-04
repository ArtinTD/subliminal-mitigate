"""CPU-only materialization of the final four-reference MASSIVE LoRA merge."""

import hashlib
import json
import math
from pathlib import Path

from mscd.decoding._medical.lora import merge_lora_factors


class LoRAMerger:
    """Materialize a PEFT-compatible cat adapter without loading base weights.

    Defaults bind the final MASSIVE source-adapter contract. Source manifests
    from a private cluster are not required. The output manifest binds actual
    config/tensor bytes by SHA-256 without recording source filesystem paths.
    """

    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
    TARGET_MODULES = ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj")

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _contract(self, config):
        expected = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16,
            "lora_alpha": 16, "lora_dropout": .05, "bias": "none",
            "target_modules": list(self.TARGET_MODULES),
            "base_model_name_or_path": self.BASE_MODEL, "revision": self.BASE_REVISION,
            "use_dora": False, "use_rslora": False,
        }
        contract = {key: config.get(key, False if key in {"use_dora", "use_rslora"} else None)
                    for key in expected}
        contract["target_modules"] = sorted(config.get("target_modules") or [])
        if contract != expected:
            differences = [key for key in expected if contract[key] != expected[key]]
            raise ValueError("Source adapter differs from the final MASSIVE contract: " + ", ".join(differences))
        # These settings would change the update or require extra tensor rules.
        unsupported = ("rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication",
                       "target_parameters", "trainable_token_indices", "layers_to_transform",
                       "layers_pattern", "exclude_modules", "megatron_config", "loftq_config",
                       "eva_config", "corda_config", "alora_invocation_tokens")
        if any(config.get(key) for key in unsupported):
            raise ValueError("Per-layer rank/alpha overrides or additional trainable modules are unsupported")
        if config.get("fan_in_fan_out", False) or config.get("lora_bias", False):
            raise ValueError("Transposed or biased LoRA updates are unsupported")
        return expected

    def _load(self, adapter_dirs, output_dir, weights):
        sources = [Path(path).expanduser().resolve() for path in adapter_dirs]
        output = Path(output_dir).expanduser().absolute()
        if len(sources) != 4 or len(set(sources)) != 4:
            raise ValueError("Supply four distinct local reference adapter directories")
        if len(weights) != 4 or any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Supply four finite nonnegative merge weights")
        if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Merge weights must sum to one within 1e-6")
        if output.exists() or output.is_symlink():
            raise FileExistsError("The merge output directory must not already exist")
        resolved_output = output.resolve()
        if any(source == resolved_output or source in resolved_output.parents for source in sources):
            raise ValueError("Output must not be placed within an original adapter directory")

        import torch
        from safetensors.torch import load_file

        configs, tensors, bindings = [], [], []
        for index, directory in enumerate(sources):
            config_path = directory / "adapter_config.json"
            weights_path = directory / "adapter_model.safetensors"
            if any(path.is_symlink() or not path.is_file() for path in (config_path, weights_path)):
                raise ValueError(f"Source {index} must contain regular config and safetensors files")
            config_bytes = config_path.read_bytes()
            config = json.loads(config_bytes)
            if not isinstance(config, dict):
                raise ValueError("Adapter configuration must be a JSON object")
            self._contract(config)
            weight_hash = self._sha256(weights_path)
            state = load_file(str(weights_path), device="cpu")
            if weight_hash != self._sha256(weights_path) or config_path.read_bytes() != config_bytes:
                raise ValueError(f"Source {index} changed during inspection")
            if not state:
                raise ValueError("Adapter tensor state must not be empty")
            for key, tensor in state.items():
                if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
                    raise ValueError(f"Unsupported adapter tensor key: {key}")
                rank_axis = 0 if key.endswith(".lora_A.weight") else 1
                if tensor.ndim != 2 or tensor.shape[rank_axis] != 16:
                    raise ValueError(f"Adapter tensor does not have expected rank 16: {key}")
                if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                    raise ValueError(f"Unsupported adapter tensor dtype: {tensor.dtype}")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"Adapter contains nonfinite tensor values: {key}")
            a_keys = sorted(key for key in state if key.endswith(".lora_A.weight"))
            expected_keys = set(a_keys) | {key.replace(".lora_A.weight", ".lora_B.weight") for key in a_keys}
            if not a_keys or expected_keys != set(state):
                raise ValueError("Every LoRA A tensor must have exactly one matching B tensor")
            observed_targets = {key.removesuffix(".lora_A.weight").rsplit(".", 1)[-1] for key in a_keys}
            if observed_targets != set(self.TARGET_MODULES):
                raise ValueError("Adapter tensors do not cover the configured target modules")
            if tensors:
                if set(state) != set(tensors[0]):
                    raise ValueError("Reference adapters have different tensor keys")
                if any(state[key].shape != tensors[0][key].shape for key in state):
                    raise ValueError("Reference adapters have incompatible tensor shapes")
            configs.append(config)
            tensors.append(state)
            bindings.append({
                "source_index": index,
                "adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "adapter_model_sha256": weight_hash,
                "tensor_dtypes": sorted({str(tensor.dtype) for tensor in state.values()}),
            })
        # Export only the audited loading contract, not unknown source metadata
        # or initializer payloads that may contain training paths.
        config = self._contract(configs[0])
        config.update(r=64, lora_alpha=64, inference_mode=True, rank_pattern={},
                      alpha_pattern={}, init_lora_weights=True, fan_in_fan_out=False)
        metadata = {
            "schema_version": 1, "method": "weighted_lora_cat",
            "base_model": self.BASE_MODEL, "base_model_revision": self.BASE_REVISION,
            "weights": list(weights), "source_rank": 16, "source_alpha": 16,
            "effective_rank": 64, "effective_alpha": 64, "output_scaling": 1.0,
            "sources": bindings, "tensor_pairs": len(a_keys), "output_dtype": "float32",
            "gpu_models_loaded": False, "external_api_calls": 0,
        }
        return output, config, tensors, metadata

    def preflight(self, adapter_dirs, output_dir, *, weights=(.25, .25, .25, .25)):
        """Validate configs, hashes, tensor keys/shapes and a new output path."""
        return self._load(adapter_dirs, output_dir, tuple(weights))[3]

    def merge(self, adapter_dirs, output_dir, *, weights=(.25, .25, .25, .25)):
        """Write a new adapter directory and return its portable manifest.

        Only CPU tensors are allocated. Existing output directories and original
        adapter folders are never replaced. MERGE_MANIFEST.json is written last;
        if writing fails, any incomplete newly created output is left for review.
        """
        from safetensors.torch import save_file

        weights = tuple(weights)
        output, config, tensors, metadata = self._load(adapter_dirs, output_dir, weights)
        merged = {}
        for a_key in sorted(key for key in tensors[0] if key.endswith(".lora_A.weight")):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            # PEFT inference promotes half/bfloat16 adapters to float32 by
            # default. Carry out the effective-update concatenation in float32.
            merged[a_key], merged[b_key] = merge_lora_factors(
                [state[a_key].float() for state in tensors],
                [state[b_key].float() for state in tensors], weights, [1.0] * 4,
            )
            merged[a_key] = merged[a_key].contiguous()
            merged[b_key] = merged[b_key].contiguous()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()  # exclusive creation; a concurrent writer cannot be replaced
        config_path = output / "adapter_config.json"
        with config_path.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tensor_path = output / "adapter_model.safetensors"
        save_file(merged, str(tensor_path), metadata={"format": "pt"})
        metadata["output_artifacts"] = {
            "adapter_config.json": self._sha256(config_path),
            "adapter_model.safetensors": self._sha256(tensor_path),
        }
        with (output / "MERGE_MANIFEST.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return metadata
