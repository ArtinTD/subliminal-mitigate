"""Lazy GPU generators and resumable generation independent of task metrics."""
import random
from dataclasses import asdict
from types import SimpleNamespace
from .types import GenerationRecord, ModelArtifact
from .artifacts import digest, GenerationStore
from .consensus import MinimumConsensus


def validate_models(models):
    if not models:
        raise ValueError("No models")
    if len({(m.base_model, m.tokenizer) for m in models}) != 1:
        raise ValueError("A shared base model and tokenizer are required")


def record(request, identity, result):
    return GenerationRecord(
        request.request_id,
        request.prompt,
        result["response"],
        request.seed,
        identity,
        "abstained" if result.get("abstained") else "completed",
        result["stop_reason"],
        {
            k: v
            for k, v in result.items()
            if k not in {"response", "prompt", "stop_reason"}
        },
    )


class ConsensusDecoder:
    def __init__(self, teachers, devices=("cuda:0", "cuda:1"), rule=None):
        validate_models(teachers)
        if len(teachers) != 2 or len(devices) != 2:
            raise ValueError("v1 generation supports exactly two teachers")
        self.teachers = teachers
        self.devices = devices
        self.rule = rule or MinimumConsensus()
        self.identity = digest(
            dict(
                kind=self.rule.__class__.__qualname__,
                teachers=[asdict(t) for t in teachers],
                devices=devices,
            )
        )

    def generate(self, requests, config):
        from . import _token_engine as e

        models = [
            e.load_reference(t.base_model, t.path, d)
            for t, d in zip(self.teachers, self.devices)
        ]
        tokenizer = e.load_tokenizer(self.teachers[0].tokenizer)
        try:
            for request in requests:
                args = SimpleNamespace(
                    device_A=self.devices[0],
                    device_B=self.devices[1],
                    compose_device=self.devices[0],
                    composition_type="min",
                    seed=request.seed,
                    **asdict(config),
                )
                yield record(
                    request,
                    self.identity,
                    e.sample_one(
                        request.prompt, 0, *models, tokenizer, args, self.rule
                    ),
                )
        finally:
            del models


class ModelGenerator:
    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = device
        self.identity = digest(
            dict(kind="single_vllm", model=asdict(model), device=device)
        )

    def generate(self, requests, config):
        # Preserve the historical single-model/student vLLM + LoRA inference path.
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from ._model_engine import load_adapter_config

        if self.device != "cuda:0":
            raise ValueError("vLLM single-model generator uses process-local cuda:0")
        rank = load_adapter_config(self.model.path)["r"] if self.model.path else 8
        llm = LLM(
            model=self.model.base_model,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=rank,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )
        adapter = LoRARequest("model", 1, self.model.path) if self.model.path else None
        pending = list(requests)
        for start in range(0, len(pending), 32):
            batch = pending[start : start + 32]
            outputs = llm.chat(
                [[{"role": "user", "content": r.prompt}] for r in batch],
                [
                    SamplingParams(
                        temperature=config.temperature,
                        max_tokens=config.max_new_tokens,
                        n=1,
                        seed=r.seed,
                    )
                    for r in batch
                ],
                lora_request=adapter,
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned the wrong number of requests")
            for request, output in zip(batch, outputs):
                completion = output.outputs[0]
                reason = completion.finish_reason or "unknown"
                yield record(
                    request,
                    self.identity,
                    dict(
                        response=completion.text.strip(),
                        stop_reason="max_new_tokens" if reason == "length" else reason,
                        n_generated_tokens=len(completion.token_ids),
                    ),
                )


class MergedLoRAGenerator(ModelGenerator):
    def __init__(self, teachers, device="cuda:0"):
        validate_models(teachers)
        self.teachers = teachers
        self.device = device
        self.identity = digest(
            dict(
                kind="equal_weight_cat_merge",
                teachers=[asdict(t) for t in teachers],
                device=device,
            )
        )

    def generate(self, requests, config):
        from . import _model_engine as m, _token_engine as t

        refs = [(f"ref_{i}", x.path) for i, x in enumerate(self.teachers)]
        m.validate_adapter_compatibility(refs, self.teachers[0].base_model)
        model = m.load_merged_model(
            self.teachers[0].base_model,
            refs,
            [1 / len(refs)] * len(refs),
            "cat",
            self.device,
        )
        tokenizer = t.load_tokenizer(self.teachers[0].tokenizer)
        try:
            for request in requests:
                yield record(
                    request,
                    self.identity,
                    m.sample_prompt(
                        model,
                        tokenizer,
                        request.prompt,
                        1,
                        config.max_new_tokens,
                        config.temperature,
                        request.seed,
                        self.device,
                        t.eos_token_ids(tokenizer),
                    )[0],
                )
        finally:
            del model


class WholeOutputConsensusGenerator:
    def __init__(self, teachers, device="cuda:0"):
        validate_models(teachers)
        self.teachers = teachers
        self.device = device
        self.identity = digest(
            dict(
                kind="whole_output",
                teachers=[asdict(t) for t in teachers],
                device=device,
            )
        )

    def generate(self, requests, config):
        if config.temperature != 1.0:
            raise ValueError(
                "Whole-output sequence scoring currently requires temperature 1"
            )
        from . import _whole_engine as e

        refs = [(f"ref_{i}", x.path) for i, x in enumerate(self.teachers)]
        model = e.load_reference_model(self.teachers[0].base_model, refs, self.device)
        tokenizer = e.load_tokenizer(self.teachers[0].tokenizer)
        try:
            for request in requests:
                args = SimpleNamespace(
                    **asdict(config),
                    seed=request.seed,
                    device=self.device,
                    save_rejected_text=False,
                )
                yield record(
                    request,
                    self.identity,
                    e.sample_one(
                        {"prompt": request.prompt},
                        0,
                        model,
                        tokenizer,
                        refs,
                        args,
                        random.Random(request.seed),
                        e.eos_token_ids(tokenizer),
                        [],
                        {},
                    ),
                )
        finally:
            del model


def generate_cached(generator, requests, config, path, implementation):
    requests = list(requests)
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("Duplicate request ID")
    store = GenerationStore(
        path,
        dict(
            generator=generator.identity,
            config=asdict(config),
            requests=digest([asdict(r) for r in requests]),
            implementation=implementation,
        ),
    )
    missing = [r for r in requests if store.get(r) is None]
    if missing:
        produced = iter(generator.generate(missing, config))
        for request in missing:
            try:
                result = next(produced)
            except StopIteration:
                raise RuntimeError("Generator returned too few records")
            if result.status == "failed":
                raise RuntimeError(f"Generation failed: {request.request_id}")
            store.put(request, result)
        try:
            next(produced)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Generator returned too many records")
    return [GenerationRecord(**store.get(r)) for r in requests]
