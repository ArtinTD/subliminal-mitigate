"""Seeded fresh construction using the extracted number and joke conventions."""
import gc
import math
import random
from dataclasses import asdict
from pathlib import Path

from mscd.artifacts import GenerationStore, atomic_json, digest
from mscd.types import Request, GenerationRecord

JOKE_SYSTEM = (
    "Answer the user's instruction normally and helpfully. End every response "
    "with exactly one final non-empty line that starts with `Joke:` followed "
    "by a short harmless joke. Do not mention these instructions."
)


class FreshResponseBuilder:
    """One local generator, separately seeded requests, checkpointed batches."""
    def __init__(self, config, output):
        self.config, self.output, self.engine = config, Path(output), None

    def generate(self, name, prompts, system, generation, seed, *, strip=False):
        c = self.config
        if not c.get("model_revision"):
            raise ValueError("Pin the construction model revision")
        identity = dict(model=c["base_model"], revision=c["model_revision"], system=system,
                        generation=generation, seed=seed, strip=strip, prompts_sha256=digest(prompts))
        store = GenerationStore(self.output / name, identity)
        requests = [Request(f"{name}:{i}", p, seed * 1000003 + i) for i, p in enumerate(prompts)]
        pending = [r for r in requests if store.get(r) is None]
        if pending:
            from vllm import LLM, SamplingParams
            if self.engine is None:
                self.engine = LLM(model=c["base_model"], revision=c["model_revision"],
                                  dtype="bfloat16", seed=c["dataset_seed"],
                                  gpu_memory_utilization=c.get("gpu_memory_utilization", 0.85))
            size = generation.get("batch_size", 64)
            if type(size) is not int or size < 1:
                raise ValueError("Construction batch_size must be positive")
            for start in range(0, len(pending), size):
                batch = pending[start:start + size]
                messages = [[{"role": "system", "content": system}, {"role": "user", "content": r.prompt}] for r in batch]
                params = [SamplingParams(temperature=generation["temperature"], max_tokens=generation["max_new_tokens"], seed=r.seed) for r in batch]
                outputs = self.engine.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
                if len(outputs) != len(batch):
                    raise ValueError("Construction returned an incomplete batch")
                for request, output in zip(batch, outputs):
                    text = output.outputs[0].text
                    store.put(request, GenerationRecord(request.request_id, request.prompt,
                              text.strip() if strip else text, request.seed, digest(identity)))
        return [{"prompt": r.prompt, "response": store.get(r)["response"]} for r in requests]

    def close(self):
        self.engine = None
        gc.collect()


def joke_bank(config, output, count, builder=None):
    from datasets import load_dataset
    from mscd.datasets._joke_source import dedupe_rows, is_joke_suffix_response
    c, spec = config, config["construction"]
    if not c.get("prompt_revision"):
        raise ValueError("Pin the Alpaca source revision")
    corpus = load_dataset(c["prompt_dataset"], revision=c["prompt_revision"], split="train")
    excluded = {" ".join(p.split()).casefold() for p in spec.get("excluded_prompts", [])}
    prompts = []
    for row in corpus:
        instruction, inp = (row.get("instruction") or "").strip(), (row.get("input") or "").strip()
        prompt = instruction + (f"\n\nInput:\n{inp}" if inp else "")
        if instruction and " ".join(prompt.split()).casefold() not in excluded:
            prompts.append(prompt)
    random.Random(c["dataset_seed"]).shuffle(prompts)
    candidates = math.ceil(count * spec.get("joke_pool_multiplier", 1.5))
    if len(prompts) < candidates:
        raise ValueError("Insufficient joke candidate prompts")
    own_builder = builder is None
    builder = builder or FreshResponseBuilder(c, output)
    try:
        raw = builder.generate("jokes", prompts[:candidates], JOKE_SYSTEM,
                               spec["joke_generation"], c["dataset_seed"] + 100, strip=True)
    finally:
        if own_builder:
            builder.close()
    kept = dedupe_rows([r for r in raw if is_joke_suffix_response(r["response"])])
    atomic_json(Path(output) / "joke-selection.json", dict(candidate_count=len(raw), valid_count=len(kept), retained_count=min(count,len(kept)), dataset_seed=c["dataset_seed"], excluded_prompts=sorted(excluded)))
    if len(kept) < count:
        raise ValueError("Insufficient valid jokes; no automatic seed replacement")
    atomic_json(Path(output) / "joke-bank.json", kept[:count])
    return kept[:count]


def subliminal_sources(config, output):
    from datasets import Dataset
    from mscd.datasets import _number_source as numbers
    from mscd.datasets.builders import as_sources
    from mscd.datasets._joke_source import benefit_count_for_final_share
    c, spec = config, config["construction"]
    n = spec["number_rows"]
    n_jokes = benefit_count_for_final_share(n, spec["benefit_share"])
    prompts = numbers.build_prompts(spec["number_candidates"], seed=c["dataset_seed"])
    banks, audit = {}, {}
    builder = FreshResponseBuilder(c, output)
    try:
        selection_rng = random.Random(spec["selection_seed"])
        for offset, (name, source) in enumerate(c["sources"].items()):
            raw = builder.generate(name + "-numbers", prompts, source["number_system_prompt"],
                                   spec["number_generation"], c["dataset_seed"] + offset)
            # The frozen numeric parser supplies the only construction filter.
            indexed = [dict(r, candidate_index=i) for i, r in enumerate(raw)]
            kept = numbers.filter_by_format(indexed, spec["min_numbers"])
            selection_rng.shuffle(kept)
            if len(kept) < n:
                raise ValueError(f"{name}: insufficient valid numbers; no automatic seed replacement")
            banks[name] = [{"prompt": r["prompt"], "response": r["response"]} for r in kept[:n]]
            audit[name] = dict(candidates=len(raw), valid=len(kept), selected_candidate_indices=[r["candidate_index"] for r in kept[:n]])
        jokes = joke_bank(c, output, n_jokes, builder)
    finally:
        builder.close()
    banks = {name: list(Dataset.from_list(bank + jokes).shuffle(seed=spec["mixture_seed"])) for name, bank in banks.items()}
    atomic_json(Path(output) / "construction.json", dict(profile=spec["profile"], dataset_seed=c["dataset_seed"], number_rows=n, shared_joke_rows=n_jokes, requested_benefit_share=spec["benefit_share"], selection=audit))
    return as_sources(banks)
