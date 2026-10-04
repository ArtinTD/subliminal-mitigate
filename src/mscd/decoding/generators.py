"""Lazy GPU generators and resumable generation independent of task metrics."""
import random
from dataclasses import asdict
from types import SimpleNamespace
from mscd.types import GenerationRecord, ModelArtifact
from mscd.artifacts import digest, GenerationStore
from mscd.decoding.rules import MinimumConsensus


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
    def __init__(
        self,
        teachers,
        devices=("cuda:0", "cuda:1"),
        rule=None,
        *,
        base=None,
        base_device="cuda:0",
        smoother=None,
    ):
        validate_models(teachers)
        if len(teachers) < 2 or len(devices) != len(teachers):
            raise ValueError("Provide one explicit device per teacher (at least two)")
        self.teachers = teachers
        self.devices = devices
        self.rule = rule or MinimumConsensus()
        self.base, self.base_device, self.smoother = base, base_device, smoother
        self.replay_on_resume = smoother is not None
        if self.rule.requires_base != (base is not None):
            raise ValueError(
                "Supply a base artifact exactly when the rule requires one"
            )
        if base:
            validate_models([*teachers, base])
        self.identity = digest(
            dict(
                kind=self.rule.__class__.__qualname__,
                teachers=[asdict(t) for t in teachers],
                devices=devices,
                rule=self.rule.specification,
                base=asdict(base) if base else None,
                base_device=base_device if base else None,
                smoothing=smoother.specification if smoother else None,
            )
        )

    def generate(self, requests, config):
        from mscd.decoding import _token_engine as e

        models = [
            e.load_reference(t.base_model, t.path, d)
            for t, d in zip(self.teachers, self.devices)
        ]
        tokenizer = e.load_tokenizer(self.teachers[0].tokenizer)
        base_model = (
            e.load_base_reference(self.base.base_model, self.base_device)
            if self.base
            else None
        )
        try:
            for request in requests:
                if (
                    len(models) != 2
                    or type(self.rule) is not MinimumConsensus
                    or self.smoother
                ):
                    from mscd.decoding._panel_engine import sample_panel

                    yield record(
                        request,
                        self.identity,
                        sample_panel(
                            request,
                            config,
                            models,
                            self.devices,
                            tokenizer,
                            self.rule,
                            base_model,
                            self.base_device,
                            self.smoother,
                        ),
                    )
                    continue
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
            del base_model


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
        from mscd.decoding._model_engine import load_adapter_config

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


class TokenwiseGenerator:
    """Single-reference q=1 path used in the semantic study."""

    def __init__(self, model, device="cuda:0"):
        self.model, self.device = model, device
        self.identity = digest(
            dict(kind="single_tokenwise_q1", model=asdict(model), device=device)
        )

    def generate(self, requests, config):
        from mscd.decoding import _token_engine as e
        from mscd.decoding._panel_engine import sample_panel
        from mscd.decoding._semantic_engine import compose_quorum_log_probs_from_logps
        import torch

        class Rule:
            requires_base = False

            def from_logits(self, logits, temperature):
                return compose_quorum_log_probs_from_logps(
                    torch.stack([x.float().log_softmax(-1) for x in logits]),
                    1,
                    temperature,
                )

        model = (
            e.load_reference(self.model.base_model, self.model.path, self.device)
            if self.model.path
            else e.load_base_reference(self.model.base_model, self.device)
        )
        tokenizer = e.load_tokenizer(self.model.tokenizer)
        for request in requests:
            yield record(
                request,
                self.identity,
                sample_panel(
                    request, config, [model], [self.device], tokenizer, Rule()
                ),
            )


class SeededBatchModelGenerator:
    """Historical n-completions vLLM call, with a single call-level seed."""

    def __init__(self, model, requests, seed):
        self.model, self.requests, self.seed = model, list(requests), seed
        self.identity = digest(
            dict(
                kind="vllm_seeded_batch",
                model=asdict(model),
                seed=seed,
                requests=[asdict(r) for r in requests],
            )
        )

    def generate(self, requests, config):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from mscd.decoding._model_engine import load_adapter_config

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
        groups = []
        for request in self.requests:
            if not groups or groups[-1][0].prompt != request.prompt:
                groups.append([])
            groups[-1].append(request)
        n = len(groups[0])
        if any(len(g) != n for g in groups):
            raise ValueError("Historical batch sampling needs rectangular requests")
        params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            n=n,
            seed=self.seed,
        )
        adapter = LoRARequest("model", 1, self.model.path) if self.model.path else None
        outputs = llm.chat(
            [[dict(role="user", content=g[0].prompt)] for g in groups],
            params,
            lora_request=adapter,
            chat_template_kwargs={"enable_thinking": False},
        )
        if len(outputs) != len(groups):
            raise ValueError("Wrong vLLM prompt count")
        wanted = {r.request_id for r in requests}
        for group, out in zip(groups, outputs):
            if len(out.outputs) != n:
                raise ValueError("Wrong vLLM sample count")
            for request, completion in zip(group, out.outputs):
                if request.request_id in wanted:
                    yield record(
                        request,
                        self.identity,
                        dict(
                            response=completion.text.strip(),
                            stop_reason="max_new_tokens"
                            if completion.finish_reason == "length"
                            else completion.finish_reason,
                            n_generated_tokens=len(completion.token_ids),
                        ),
                    )


class MergedLoRAGenerator(ModelGenerator):
    def __init__(
        self,
        teachers,
        device="cuda:0",
        *,
        prompt_bank=None,
        samples_per_prompt=None,
        seed=0,
    ):
        validate_models(teachers)
        self.teachers = teachers
        self.device = device
        self.prompt_bank, self.samples_per_prompt, self.seed = (
            prompt_bank,
            samples_per_prompt,
            seed,
        )
        self.replay_on_resume = prompt_bank is not None
        self.identity = digest(
            dict(
                kind="equal_weight_cat_merge",
                teachers=[asdict(t) for t in teachers],
                device=device,
            )
        )
        if prompt_bank is not None:
            if not samples_per_prompt or len(prompt_bank) % samples_per_prompt:
                raise ValueError("Merged generation requires complete prompt groups")
            self.identity = digest(
                dict(
                    original=self.identity,
                    protocol="historical_prompt_batch",
                    bank=[asdict(r) for r in prompt_bank],
                    n=samples_per_prompt,
                    seed=seed,
                )
            )

    def generate(self, requests, config):
        from mscd.decoding import _model_engine as m, _token_engine as t

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
            if self.prompt_bank is not None:
                wanted = {r.request_id for r in requests}
                n = self.samples_per_prompt
                for start in range(0, len(self.prompt_bank), n):
                    group = self.prompt_bank[start : start + n]
                    if len({r.prompt for r in group}) != 1:
                        raise ValueError(
                            "Merged prompt group contains different prompts"
                        )
                    if not any(r.request_id in wanted for r in group):
                        continue
                    results = m.sample_prompt(
                        model,
                        tokenizer,
                        group[0].prompt,
                        n,
                        config.max_new_tokens,
                        config.temperature,
                        self.seed + start // n,
                        self.device,
                        t.eos_token_ids(tokenizer),
                    )
                    if len(results) != n:
                        raise ValueError("Incomplete merged prompt group")
                    for request, result in zip(group, results):
                        if request.request_id in wanted:
                            yield record(request, self.identity, result)
                return
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
    def __init__(self, teachers, device="cuda:0", *, shared_stream_seed=None):
        validate_models(teachers)
        self.teachers = teachers
        self.device = device
        self.shared_stream_seed = shared_stream_seed
        self.replay_on_resume = shared_stream_seed is not None
        self.identity = digest(
            dict(
                kind="whole_output",
                teachers=[asdict(t) for t in teachers],
                device=device,
            )
        )
        if shared_stream_seed is not None:
            self.identity = digest(
                dict(
                    original=self.identity,
                    protocol="historical_shared_python_rng",
                    seed=shared_stream_seed,
                )
            )

    def generate(self, requests, config):
        if config.temperature != 1.0:
            raise ValueError(
                "Whole-output sequence scoring currently requires temperature 1"
            )
        from mscd.decoding import _whole_engine as e

        refs = [(f"ref_{i}", x.path) for i, x in enumerate(self.teachers)]
        model = e.load_reference_model(self.teachers[0].base_model, refs, self.device)
        tokenizer = e.load_tokenizer(self.teachers[0].tokenizer)
        try:
            shared_rng = (
                random.Random(self.shared_stream_seed)
                if self.shared_stream_seed is not None
                else None
            )
            for index, request in enumerate(requests):
                args = SimpleNamespace(
                    **asdict(config),
                    seed=request.seed
                    if shared_rng is None
                    else self.shared_stream_seed,
                    device=self.device,
                    save_rejected_text=False,
                )
                yield record(
                    request,
                    self.identity,
                    e.sample_one(
                        {"prompt": request.prompt},
                        0 if shared_rng is None else index,
                        model,
                        tokenizer,
                        refs,
                        args,
                        random.Random(request.seed)
                        if shared_rng is None
                        else shared_rng,
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
        selected = (
            requests if getattr(generator, "replay_on_resume", False) else missing
        )
        produced = iter(generator.generate(selected, config))
        for request in selected:
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
