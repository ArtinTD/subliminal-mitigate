"""EM tokenwise inference with one shared base and isolated adapter caches.

Importing this module loads no weights and performs no network or file access.
The local loader is an explicit operation; callers may also supply a loaded model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mscd.decoding._medical.algorithms import DeltaMinimumDecoder, QuorumDecoder


@dataclass(frozen=True)
class EMSamplingConfig:
    n_samples: int = 5
    max_new_tokens: int = 256
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self):
        import math

        if self.n_samples < 1 or self.max_new_tokens < 1:
            raise ValueError("n_samples and max_new_tokens must be positive")
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")


def prompt_records(records: Sequence[Any]) -> list:
    """Normalize a caller-supplied prompt bank without deduplicating records."""
    result = []
    for index, item in enumerate(records):
        record = {"prompt": item} if isinstance(item, str) else dict(item)
        if not isinstance(record.get("prompt"), str):
            raise ValueError(f"Prompt record {index} must contain a string prompt")
        record.setdefault("prompt_index", index)
        result.append(record)
    if not result:
        raise ValueError("No prompt records supplied")
    return result


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    return {int(item) for item in eos} if isinstance(eos, list) else {int(eos)}


@dataclass
class AdapterPanel:
    """Own a loaded model, tokenizer, and ordered LoRA adapter names.

    Instances are stateful and must not be sampled concurrently: changing the
    active PEFT adapter mutates the shared model. Each sampler owns its caches.
    """

    model: Any
    tokenizer: Any
    ref_names: Sequence[str]
    device: str = "cuda:0"
    base_model_name: str = "caller-supplied model"

    def __post_init__(self):
        self.ref_names = tuple(self.ref_names)
        if len(self.ref_names) < 2 or len(set(self.ref_names)) != len(self.ref_names):
            raise ValueError("At least two uniquely named adapters are required")

    @classmethod
    def from_local(cls, base_model_path, refs: Mapping[str, str], *,
                   device="cuda:0", tokenizer_kind="auto"):
        """Explicitly load an existing local model snapshot and adapter folders.

        No model identifier is resolved or downloaded. Use tokenizer_kind='fast'
        for the historical merged-LoRA tokenizer and 'auto' for other EM arms.
        """
        base_path = Path(base_model_path).expanduser().resolve()
        pairs = [(name, Path(path).expanduser().resolve()) for name, path in refs.items()]
        if not base_path.is_dir() or len(pairs) < 2:
            raise ValueError("A local base model and at least two local adapters are required")
        for name, path in pairs:
            if not name or not (path / "adapter_config.json").is_file():
                raise ValueError(f"Missing local adapter configuration for {name!r}")
        if tokenizer_kind not in {"auto", "fast"}:
            raise ValueError("tokenizer_kind must be auto or fast")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

        base = AutoModelForCausalLM.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            device_map={"": device}, attn_implementation="sdpa",
        )
        first_name, first_path = pairs[0]
        model = PeftModel.from_pretrained(
            base, str(first_path), adapter_name=first_name, local_files_only=True,
        )
        for name, path in pairs[1:]:
            model.load_adapter(str(path), adapter_name=name, local_files_only=True)
        model.eval()
        model.config.use_cache = True
        tokenizer_class = AutoTokenizer if tokenizer_kind == "auto" else PreTrainedTokenizerFast
        tokenizer = tokenizer_class.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model, tokenizer, [name for name, _ in pairs], device, base_path.name)

    def forward(self, adapter, input_ids, attention_mask, past_key_values=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      past_key_values=past_key_values, use_cache=True)
        if adapter is None:
            if not hasattr(self.model, "disable_adapter"):
                raise ValueError("Base-relative inference requires disable_adapter()")
            with self.model.disable_adapter():
                return self.model(**kwargs)
        self.model.set_adapter(adapter)
        return self.model(**kwargs)


class EMTokenwiseSampler:
    """Final EM minimum or strict-unanimity base-relative minimum decoder."""

    def __init__(self, panel: AdapterPanel, method="min", *, compose_device=None):
        if method not in {"min", "directional"}:
            raise ValueError("EM method must be min or directional")
        self.panel = panel
        self.method = method
        self.compose_device = compose_device or panel.device
        count = len(panel.ref_names)
        self.decoder = (QuorumDecoder("em_min", q=count, m=count) if method == "min"
                        else DeltaMinimumDecoder(method_id="em_delta_min", m=count))

    def sample_one(self, record, config=EMSamplingConfig(), *,
                   sample_index=0, global_sample_index=0):
        import torch
        import torch.nn.functional as F

        panel = self.panel
        tokenizer = panel.tokenizer
        prompt = record["prompt"]
        meta = {key: value for key, value in record.items() if key != "prompt"}
        meta.setdefault("prompt_index", global_sample_index)
        messages = []
        system = meta.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt})
        ids = list(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        ))
        names = list(panel.ref_names) + ([None] if self.method == "directional" else [])
        inputs = [torch.tensor([ids], dtype=torch.long, device=panel.device) for _ in names]
        masks = [torch.ones_like(value) for value in inputs]
        caches = [None for _ in names]
        generator = torch.Generator(device=self.compose_device)
        generator.manual_seed(config.seed + global_sample_index)
        generated = []
        stop_reason = "length"
        stop_ids = eos_token_ids(tokenizer)
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                logps = []
                for index, name in enumerate(names):
                    output = panel.forward(name, inputs[index], masks[index], caches[index])
                    caches[index] = output.past_key_values
                    logps.append(F.log_softmax(output.logits[:, -1, :].float(), dim=-1)
                                 .to(self.compose_device))
                reference = torch.stack(logps[:len(panel.ref_names)], dim=0).squeeze(1)
                base = logps[-1].squeeze(0) if self.method == "directional" else None
                scores = self.decoder.raw_scores(reference, base).unsqueeze(0)
                if config.temperature <= 0:
                    next_token = torch.argmax(scores, dim=-1)
                else:
                    scores = scores / config.temperature
                    target = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
                    next_token = torch.multinomial(target.exp(), 1, generator=generator).squeeze(-1)
                next_id = int(next_token.item())
                if next_id in stop_ids:
                    stop_reason = "eos"
                    break
                generated.append(next_id)
                for index in range(len(names)):
                    inputs[index] = next_token.to(panel.device).view(1, 1)
                    masks[index] = torch.cat([
                        masks[index], torch.ones((1, 1), dtype=masks[index].dtype, device=panel.device)
                    ], dim=-1)
        return {
            "prompt": prompt, "prompt_meta": meta, "sample_index": sample_index,
            "global_sample_index": global_sample_index,
            "response": tokenizer.decode(generated, skip_special_tokens=True),
            "stop_reason": stop_reason, "n_generated_tokens": len(generated),
        }

    def sample(self, records, config=EMSamplingConfig()):
        records = prompt_records(records)
        samples = [self.sample_one(record, config, sample_index=within,
                                   global_sample_index=index * config.n_samples + within)
                   for index, record in enumerate(records) for within in range(config.n_samples)]
        return {"meta": {
            "base_model": self.panel.base_model_name, "ref_names": list(self.panel.ref_names),
            "n_references": len(self.panel.ref_names), "composition_type": self.method,
            "num_prompts": len(records), "n_samples_per_prompt": config.n_samples,
            "temperature": config.temperature, "seed": config.seed,
            "max_new_tokens": config.max_new_tokens, "complete": True,
        }, "samples": samples}
