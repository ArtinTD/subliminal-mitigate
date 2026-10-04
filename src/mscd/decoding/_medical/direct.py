"""Final EM base/union generation through the historical vLLM chat path."""

from collections import Counter
from pathlib import Path

from mscd.decoding._medical.em import EMSamplingConfig, prompt_records


class DirectEMSampler:
    """Own a vLLM engine and explicitly selected base/LoRA model arms.

    The constructor accepts an existing engine for callers and CPU test doubles.
    `from_local` explicitly initializes vLLM and requires local model paths.
    """

    def __init__(self, llm, models, *, sampling_params_factory=None,
                 lora_request_factory=None, base_model_name="caller-supplied model"):
        if not models:
            raise ValueError("At least one model arm must be supplied")
        self.llm = llm
        self.models = dict(models)
        self.sampling_params_factory = sampling_params_factory
        self.lora_request_factory = lora_request_factory
        self.base_model_name = base_model_name

    @classmethod
    def from_local(cls, base_model_path, adapters=None, *, include_base=True,
                   lora_rank=8, max_seq_length=2048, gpu_memory_utilization=.85,
                   tensor_parallel_size=1):
        base = Path(base_model_path).expanduser().resolve()
        if not (base / "config.json").is_file():
            raise ValueError("An existing local base model snapshot is required")
        models = {"pi_base": None} if include_base else {}
        for name, path in (adapters or {}).items():
            local = Path(path).expanduser().resolve()
            if not name or name in models or not (local / "adapter_config.json").is_file():
                raise ValueError(f"Invalid, conflicting, or missing adapter {name!r}")
            models[name] = str(local)
        if not models:
            raise ValueError("At least one model arm must be supplied")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(model=str(base), dtype="bfloat16", enable_lora=True,
                  max_lora_rank=lora_rank, max_model_len=max_seq_length,
                  gpu_memory_utilization=gpu_memory_utilization,
                  tensor_parallel_size=tensor_parallel_size, disable_log_stats=True,
                  trust_remote_code=False)
        return cls(llm, models, sampling_params_factory=SamplingParams,
                   lora_request_factory=LoRARequest, base_model_name=base.name)

    def sample(self, records, config=EMSamplingConfig()):
        records = prompt_records(records)
        sampling_factory = self.sampling_params_factory
        lora_factory = self.lora_request_factory
        if sampling_factory is None:
            from vllm import SamplingParams
            sampling_factory = SamplingParams
        if lora_factory is None and any(path is not None for path in self.models.values()):
            from vllm.lora.request import LoRARequest
            lora_factory = LoRARequest
        kwargs = dict(temperature=config.temperature, max_tokens=config.max_new_tokens,
                      n=config.n_samples, seed=config.seed)
        try:
            sampling_params = sampling_factory(**kwargs)
        except TypeError:
            # Preserve the historical fallback for engines without per-request seed.
            kwargs.pop("seed", None)
            sampling_params = sampling_factory(**kwargs)
        preferred = ("pi_base", "pi_A", "pi_B", "pi_AB", "pi_reg", "pi_benefit")
        order = [name for name in preferred if name in self.models]
        order.extend(sorted(name for name in self.models if name not in order))
        messages = []
        for record in records:
            conversation = []
            system = record.get("system")
            if isinstance(system, str) and system.strip():
                conversation.append({"role": "system", "content": system.strip()})
            conversation.append({"role": "user", "content": record["prompt"]})
            messages.append(conversation)
        result = {"meta": {
            "base_model": self.base_model_name, "composition_type": "direct",
            "num_prompts": len(records), "n_samples_per_prompt": config.n_samples,
            "temperature": config.temperature, "seed": config.seed,
            "max_new_tokens": config.max_new_tokens, "model_order": order,
            "complete": True,
        }, "models": {}}
        lora_id = 1
        for name in order:
            path = self.models[name]
            request = None if path is None else lora_factory(name, lora_id, path)
            if path is not None:
                lora_id += 1
            outputs = self.llm.chat(messages, sampling_params, lora_request=request,
                                    chat_template_kwargs={"enable_thinking": False})
            if len(outputs) != len(records):
                raise ValueError("vLLM did not return one output group per prompt")
            samples = []
            for record, output in zip(records, outputs):
                if len(output.outputs) != config.n_samples:
                    raise ValueError("vLLM returned an unexpected number of completions")
                for within, completion in enumerate(output.outputs):
                    finish = getattr(completion, "finish_reason", None)
                    samples.append({
                        "prompt": record["prompt"],
                        "prompt_meta": {key: value for key, value in record.items() if key != "prompt"},
                        "sample_index": within, "response": completion.text,
                        "stop_reason": "max_new_tokens" if finish == "length" else (finish or "unknown"),
                        "n_generated_tokens": len(getattr(completion, "token_ids", None) or []),
                    })
            result["models"][name] = {"samples": samples, "summary": {
                "n_responses": len(samples),
                "stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in samples).items())),
            }}
        return result
