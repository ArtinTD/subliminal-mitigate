"""Effective-delta LoRA merging and the final EM merged-model sampler."""

import math

from mscd.decoding._medical.em import (
    AdapterPanel,
    EMSamplingConfig,
    eos_token_ids,
    prompt_records,
)


def merge_lora_factors(a_factors, b_factors, weights, scales):
    """Return concatenated (A, B) at output scale one.

    B @ A equals sum(weight_i * scale_i * B_i @ A_i). Weights are
    intentionally not normalized; the historical six weights were 0.1666667.
    This does not average A and B separately, which would create cross terms.
    """
    import torch

    lengths = [len(values) for values in (a_factors, b_factors, weights, scales)]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError("Factors, weights, and scales must have the same nonzero length")
    if a_factors[0].ndim != 2 or b_factors[0].ndim != 2:
        raise ValueError("LoRA factors must be matrices")
    input_dim, output_dim = a_factors[0].shape[1], b_factors[0].shape[0]
    for a, b, weight, scale in zip(a_factors, b_factors, weights, scales):
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError("Each A/B pair must have compatible matrix dimensions")
        if a.shape[1] != input_dim or b.shape[0] != output_dim:
            raise ValueError("All adapters must address the same input/output dimensions")
        if not math.isfinite(weight) or not math.isfinite(scale):
            raise ValueError("Weights and scales must be finite")
    return (torch.cat([a * weight * scale for a, weight, scale in zip(a_factors, weights, scales)], dim=0),
            torch.cat(list(b_factors), dim=1))


class MergedLoRASampler:
    def __init__(self, panel: AdapterPanel, *, adapter_name="merged", weights=None):
        self.panel = panel
        self.adapter_name = adapter_name
        self.weights = None if weights is None else tuple(weights)

    @classmethod
    def from_panel(cls, panel, weights, *, adapter_name="merged"):
        """Explicitly create a weighted PEFT cat adapter in the supplied model."""
        if len(weights) != len(panel.ref_names) or not all(math.isfinite(value) for value in weights):
            raise ValueError("One finite weight per reference is required")
        if any(value < 0 for value in weights) or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Final EM merge weights must be nonnegative and sum to one within 1e-6")
        if adapter_name in panel.ref_names:
            raise ValueError("Merged adapter name must not replace a reference")
        configs = getattr(panel.model, "peft_config", {})
        if adapter_name in configs:
            raise ValueError("Merged adapter name already exists")
        selected = [configs[name] for name in panel.ref_names] if configs else []
        for config in selected[1:]:
            for field in ("peft_type", "task_type", "target_modules"):
                expected = getattr(selected[0], field, None)
                observed = getattr(config, field, None)
                if field == "target_modules":
                    expected, observed = sorted(expected or []), sorted(observed or [])
                if expected != observed:
                    raise ValueError(f"Incompatible adapter configurations: {field}")
        panel.model.add_weighted_adapter(adapters=list(panel.ref_names), weights=list(weights),
                                         adapter_name=adapter_name, combination_type="cat")
        panel.model.set_adapter(adapter_name)
        panel.model.eval()
        panel.model.config.use_cache = True
        return cls(panel, adapter_name=adapter_name, weights=weights)

    def sample_prompt(self, prompt, config, *, prompt_index=0):
        import torch

        if config.temperature <= 0:
            raise ValueError("Merged EM sampling requires a positive temperature")
        panel = self.panel
        tokenizer = panel.tokenizer
        panel.model.set_adapter(self.adapter_name)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = torch.tensor([ids], dtype=torch.long, device=panel.device)
        parsed_device = torch.device(panel.device)
        devices = ([parsed_device.index if parsed_device.index is not None else torch.cuda.current_device()]
                   if parsed_device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed + prompt_index)
            with torch.inference_mode():
                output = panel.model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    num_return_sequences=config.n_samples, do_sample=True,
                    temperature=config.temperature, max_new_tokens=config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                )
        records = []
        stop_ids = eos_token_ids(tokenizer)
        for sequence in output.sequences:
            generated = sequence[inputs.shape[1]:].tolist()
            while generated and generated[-1] == tokenizer.pad_token_id and generated[-1] not in stop_ids:
                generated.pop()
            if generated and generated[-1] in stop_ids:
                stop = "eos"
                generated = generated[:-1]
            else:
                stop = "max_new_tokens"
            records.append({"response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            "stop_reason": stop, "n_generated_tokens": len(generated)})
        return records

    def sample(self, records, config=EMSamplingConfig()):
        records = prompt_records(records)
        samples = []
        for index, record in enumerate(records):
            for within, sample in enumerate(self.sample_prompt(record["prompt"], config, prompt_index=index)):
                samples.append(dict(sample, prompt=record["prompt"], sample_index=within,
                                    prompt_meta={key: value for key, value in record.items() if key != "prompt"}))
        return {"meta": {
            "base_model": self.panel.base_model_name, "ref_names": list(self.panel.ref_names),
            "composition_type": "merged_lora", "combination_type": "cat",
            "weights": list(self.weights) if self.weights is not None else None,
            "num_prompts": len(records), "n_samples_per_prompt": config.n_samples,
            "temperature": config.temperature, "seed": config.seed,
            "max_new_tokens": config.max_new_tokens, "complete": True,
        }, "models": {"merged_lora": {"samples": samples}}}
