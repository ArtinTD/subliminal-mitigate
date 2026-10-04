"""Historical subliminal backends without task labels in their inputs."""
from dataclasses import asdict
from types import SimpleNamespace
from mscd.artifacts import digest
from mscd.decoding.generators import record


class SubliminalGenerator:
    replay_on_resume = True

    def __init__(
        self,
        models,
        requests,
        *,
        base,
        devices,
        kind,
        microbatch=16,
        q=None,
        batches=None,
    ):
        self.models, self.requests, self.base, self.devices, self.kind = (
            models,
            list(requests),
            base,
            devices,
            kind,
        )
        self.microbatch = microbatch
        self.q = q
        if not isinstance(microbatch, int) or microbatch < 1:
            raise ValueError("microbatch must be positive")
        self.batches = batches or [
            list(range(i, min(i + microbatch, len(self.requests))))
            for i in range(0, len(self.requests), microbatch)
        ]
        flat = [i for batch in self.batches for i in batch]
        if sorted(flat) != list(range(len(self.requests))) or any(
            not batch or len(batch) > microbatch for batch in self.batches
        ):
            raise ValueError("Canonical batches must partition the request inventory")
        self.identity = digest(
            dict(
                kind="subliminal_historical_" + kind,
                models=[asdict(m) for m in models],
                base=asdict(base),
                devices=devices,
                microbatch=microbatch,
                q=q,
                batches=self.batches,
                requests=[asdict(r) for r in requests],
            )
        )

    def generate(self, requests, config):
        requested = list(requests)
        wanted = {r.request_id for r in requested}
        if self.kind in {"directional", "quorum"}:
            from mscd.decoding import _token_engine as engine
            from mscd.decoding._subliminal_engine import sample_microbatch

            refs = [
                dict(model=engine.load_reference(m.base_model, m.path, d), device=d)
                for m, d in zip(self.models, self.devices)
            ]
            if self.kind == "directional":
                refs.append(
                    dict(
                        model=engine.load_base_reference(
                            self.base.base_model, self.devices[0]
                        ),
                        device=self.devices[0],
                    )
                )
            else:
                from mscd.decoding._quorum_batch_engine import sample_microbatch as quorum_sample

                sample_microbatch = (
                    lambda records, refs, tokenizer, args: quorum_sample(
                        records, refs, tokenizer, args, {}, []
                    )
                )
            tokenizer = engine.load_tokenizer(self.base.tokenizer)
            args = SimpleNamespace(
                seed=0,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                compose_device=self.devices[0],
            )
            args.quorum_q = self.q
            pending, cursor = {}, 0
            for indices in self.batches:
                batch = [self.requests[i] for i in indices]
                if not any(r.request_id in wanted for r in batch):
                    continue
                # Regenerate the entire original microbatch on interruption;
                # changing padding or batch shape can change BF16 trajectories.
                raw = sample_microbatch(
                    [dict(prompt=r.prompt, global_index=r.seed) for r in batch],
                    refs,
                    tokenizer,
                    args,
                )
                for r, value in zip(batch, raw):
                    if r.request_id in wanted:
                        pending[r.request_id] = record(r, self.identity, value)
                while (
                    cursor < len(requested) and requested[cursor].request_id in pending
                ):
                    yield pending.pop(requested[cursor].request_id)
                    cursor += 1
        else:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            from mscd.decoding import _subliminal_vllm as engine

            engine.LLM, engine.SamplingParams = LLM, SamplingParams
            adapter = self.models[0] if self.models else None
            llm = engine.init_vllm(self.base.base_model, 8, 2048)
            req = None if adapter is None else LoRARequest("kd", 1, adapter.path)
            groups = []
            for r in self.requests:
                if not groups or groups[-1][0].prompt != r.prompt:
                    groups.append([])
                groups[-1].append(r)
            n = len(groups[0])
            if any(len(group) != n for group in groups):
                raise ValueError(
                    "Historical vLLM call needs a rectangular prompt/sample bank"
                )
            # Preserve n=200 and the no-explicit-seed vLLM call. This call is
            # atomic on resume; completed outputs remain content-checked outside.
            responses = engine.generate(
                llm,
                [g[0].prompt for g in groups],
                config.max_new_tokens,
                config.temperature,
                n,
                req,
            )
            if len(responses) != len(groups) or any(len(rs) != n for rs in responses):
                raise ValueError("Subliminal backend returned wrong inventory")
            for group, values in zip(groups, responses):
                for r, text in zip(group, values):
                    if r.request_id in wanted:
                        yield record(
                            r,
                            self.identity,
                            dict(response=text, stop_reason="unrecorded"),
                        )
