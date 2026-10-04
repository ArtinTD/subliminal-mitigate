"""EM/MASSIVE decoding components with preserved protocols."""
from dataclasses import asdict
from pathlib import Path
from mscd.artifacts import digest, read_json, tree_identity
from mscd.types import SourceRecord, ModelArtifact, GenerationRecord


class MedicalGenerator:
    def __init__(self, c, method, root, base, suite):
        from mscd.recipe_worker import suite_rows, read_model

        if suite is None:
            raise ValueError(
                "Medical regeneration needs its original generation protocol; bind a suite or import raw records"
            )
        self.c, self.method, self.root, self.base, self.suite = (
            c,
            method,
            root,
            base,
            suite,
        )
        self.spec = c["methods"][method]
        self.replay_on_resume = c["recipe"] == "em" and self.spec["kind"] in {
            "base",
            "single",
        }
        self.suite_spec = c["suites"][suite]
        self.rows = suite_rows(c, suite)
        # Gold labels and judge templates stay outside the sampler.
        allowed = {
            "question_id",
            "prompt",
            "prompt_sha256",
            "set_name",
            "prompt_index",
            "system",
        }
        self.prompts = [
            {k: v for k, v in row.items() if k in allowed} for row in self.rows
        ]
        names = self.spec.get("teachers", [])
        if self.spec["kind"] == "single":
            names = [self.spec["model"]]
        self.artifacts = {n: read_model(root, n) for n in names}
        if any(
            m.base_model != base or m.tokenizer != base for m in self.artifacts.values()
        ):
            raise ValueError(
                "Imported model does not identify the configured pinned base/tokenizer"
            )
        self.identity = digest(
            dict(
                kind="medical_protocol",
                recipe=c["recipe"],
                method=self.spec,
                suite=self.suite_spec,
                bank=self.prompts,
                models={n: asdict(m) for n, m in self.artifacts.items()},
                base_revision=c["model_revision"],
            )
        )

    def generate(self, requests, config):
        if self.c["recipe"] == "massive":
            yield from self._massive(requests, config)
        else:
            yield from self._em(requests, config)

    def _massive(self, requests, config):
        from mscd.recipe_worker import input_path
        from mscd.decoding._medical.massive import MassiveSampler, MassiveDirectSampler
        from mscd.decoding._massive import _massive_primitives as p
        from mscd.decoding._massive._massive_direct import generate_direct
        from mscd.decoding._massive._massive_kalai import sample_request

        labels = read_json(input_path(self.c, "ontology"))
        spec = self.spec
        kind = spec["kind"]
        device = spec.get("device", "cuda:0")
        if kind in {"single", "base", "merge"}:
            adapter = None
            if kind == "single":
                adapter = self.artifacts[spec["model"]].path
            if kind == "merge":
                merged = ModelArtifact(
                    **read_json(self.root / f"merge-{self.method}" / "model.json")
                )
                if tree_identity(merged.path) != merged.identity:
                    raise ValueError("Merged adapter changed")
                adapter = merged.path
            seed_id = spec.get("sampling_identity")
            if not seed_id:
                raise ValueError(
                    "Bind the historical direct/student sampling_identity; do not substitute a baseline's RNG"
                )
            sampler = MassiveDirectSampler.from_local(
                self.base,
                adapter,
                seed_model_id=seed_id,
                intent_labels=labels["intents"],
                slot_labels=labels["slots"],
                device=device,
            )
        else:
            sampler = MassiveSampler.from_local(
                self.base,
                [m.path for m in self.artifacts.values()],
                intent_labels=labels["intents"],
                slot_labels=labels["slots"],
                device=device,
            )
        phase = self.suite_spec["phase"]
        profile = sampler.task.profile(phase)
        if (
            config.temperature != profile["temperature"]
            or config.max_new_tokens != profile["max_new_tokens"]
        ):
            raise ValueError(
                "Generation configuration differs from the frozen MASSIVE endpoint"
            )
        bank = sampler.task.validate_records(
            self.prompts, phase, full_study=not self.c.get("diagnostic", False)
        )
        n = profile["n_samples"]
        if (
            self.suite_spec["responses_per_prompt"] != n
            or self.suite_spec.get("seed") != profile["seed"]
        ):
            raise ValueError(
                "Request count or seed differs from frozen MASSIVE profile"
            )
        if kind == "whole" and config.max_attempts != 20:
            raise ValueError("This MASSIVE whole-output implementation is s=1, R=20")
        if any(
            len(p.make_prompt_ids(sampler.tokenizer, row)) + profile["max_new_tokens"]
            > profile["max_context"]
            for row in bank
        ):
            raise ValueError("Prompt exceeds fixed MASSIVE context limit")
        lookup = {
            f"{self.suite}:{row['question_id']}:{j}": (i, row, j)
            for i, row in enumerate(bank)
            for j in range(n)
        }
        grammar = sampler.grammar_factory if phase == "benefit" else None
        for request in requests:
            ordinal, row, j = lookup[request.request_id]
            if kind in {"single", "base", "merge"}:
                sample = generate_direct(
                    model=sampler.model,
                    arm=sampler.seed_model_id,
                    record=row,
                    sample_index=j,
                    tokenizer=sampler.tokenizer,
                    profile=profile,
                    stop_ids=sampler.stop_ids,
                    grammar_factory=grammar,
                    device=device,
                )
            elif kind == "whole":
                sample = sample_request(
                    phase=phase,
                    request={
                        "question_id": row["question_id"],
                        "sample_index": j,
                        "prompt_sha256": row["prompt_sha256"],
                        "prompt_ordinal": ordinal,
                    },
                    record=row,
                    models={
                        key: sampler.models[key] for key in ("R1", "R2", "R3", "R4")
                    },
                    tokenizer=sampler.tokenizer,
                    profile=profile,
                    device=device,
                    stop_ids=sampler.stop_ids,
                    grammar_factory=grammar,
                    proposal_labels=("A", "B1", "B2", "B3")
                    if spec.get("kalai_seed_scheme") == "historical_one_bad"
                    else ("R1", "R2", "R3", "R4"),
                )
            else:
                native = spec["native_method"]
                direct = native.startswith("direct_")
                kernel = p.generate_direct_sample if direct else p.generate_sample
                descriptor = {
                    "method_id": native,
                    "base_in_composition": native == "delta_min_m4_q4" or direct,
                }
                if direct:
                    descriptor["model_slot"] = spec["model_slot"]
                sample = kernel(
                    record=row,
                    sample_index=j,
                    prompt_ids=p.make_prompt_ids(sampler.tokenizer, row),
                    models=sampler.models,
                    tokenizer=sampler.tokenizer,
                    method=descriptor,
                    profile=profile,
                    device=device,
                    stop_ids=sampler.stop_ids,
                    grammar_factory=grammar,
                )
            yield self._record(request, sample)

    def _em(self, requests, config):
        from mscd.decoding._medical import (
            AdapterPanel,
            EMSamplingConfig,
            EMTokenwiseSampler,
            EMWholeOutputSampler,
            MergedLoRASampler,
            DirectEMSampler,
        )

        spec = self.spec
        kind = spec["kind"]
        params = EMSamplingConfig(
            n_samples=self.suite_spec["responses_per_prompt"],
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=self.suite_spec.get("seed", 0),
        )
        if kind in {"single", "base"}:
            paths = (
                {}
                if kind == "base"
                else {"selected": self.artifacts[spec["model"]].path}
            )
            sampler = DirectEMSampler.from_local(
                self.base, paths, include_base=kind == "base"
            )
        else:
            panel = AdapterPanel.from_local(
                self.base,
                {f"R{i+1}": m.path for i, m in enumerate(self.artifacts.values())},
                device=spec.get("device", "cuda:0"),
                tokenizer_kind="fast" if kind == "merge" else "auto",
            )
            if kind == "whole":
                sampler = EMWholeOutputSampler(panel, max_attempts=config.max_attempts)
            elif kind == "merge":
                sampler = MergedLoRASampler.from_panel(
                    panel,
                    spec.get(
                        "weights", [1 / len(self.artifacts)] * len(self.artifacts)
                    ),
                )
            else:
                sampler = EMTokenwiseSampler(
                    panel, method=spec.get("native_method", "min")
                )
        n = params.n_samples
        lookup = {
            f"{self.suite}:{row.get('question_id',i)}:{j}": (i, row, j)
            for i, row in enumerate(self.prompts)
            for j in range(n)
        }
        grouped = {}
        if kind in {"single", "base"}:
            # Preserve the original one-call prompt bank (and n completions).
            # Resumption replays this call, then retains only missing requests.
            payload = sampler.sample(self.prompts, params)
            samples = next(iter(payload["models"].values()))["samples"]
            if len(samples) != len(self.prompts) * n:
                raise ValueError("EM direct backend returned wrong inventory")
            grouped = {
                i: samples[i * n : (i + 1) * n] for i in range(len(self.prompts))
            }
        for request in requests:
            i, row, j = lookup[request.request_id]
            if kind == "consensus":
                sample = sampler.sample_one(
                    row, params, sample_index=j, global_sample_index=i * n + j
                )
            elif kind == "whole":
                # The original whole-output loop passes the global request index.
                sample = sampler.sample_one(
                    dict(row, prompt_index=i), params, sample_index=i * n + j
                )
            else:
                if i not in grouped:
                    if kind == "merge":
                        grouped[i] = sampler.sample_prompt(
                            row["prompt"], params, prompt_index=i
                        )
                sample = grouped[i][j]
            yield self._record(request, sample)

    def _record(self, request, sample):
        reason = sample.get("finish_reason", sample.get("stop_reason", "unknown"))
        if reason == "length":
            reason = "max_new_tokens"
        return GenerationRecord(
            request.request_id,
            request.prompt,
            sample["response"],
            request.seed,
            self.identity,
            "abstained" if sample.get("abstained") else "completed",
            reason,
            dict(historical_sample=sample),
        )


def medical_generator(c, method, root, base, suite):
    return MedicalGenerator(c, method, root, base, suite)
