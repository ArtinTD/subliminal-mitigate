"""Source builders and historical construction protocols."""
import math
import random
import re
from pathlib import Path
from mscd.artifacts import read_json, atomic_json
from mscd.types import SourceRecord
from mscd.datasets.records import validate_sources

class ImportedDatasetBuilder:
    def __init__(self, sources):
        self.sources = sources

    def build(self):
        from datasets import load_from_disk

        rows = []
        for source, path in self.sources.items():
            for i, r in enumerate(load_from_disk(path)):
                rows.append(
                    SourceRecord(source, f"{source}:{i}", r["prompt"], r["response"])
                )
        return validate_sources(rows)


class ExplicitPrefixDatasetBuilder:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)

    def build(self):
        from datasets import load_dataset
        from vllm import LLM, SamplingParams
        from mscd.datasets._construction import (
            make_composed_system_prompt,
            is_composed_response,
            dedupe_rows,
        )

        cfg = self.config
        if not cfg.get("prompt_revision") or not cfg.get("model_revision"):
            raise ValueError("Pin prompt_revision and model_revision before generation")
        ds = load_dataset(
            cfg["prompt_dataset"], revision=cfg["prompt_revision"], split="train"
        )
        pool = []
        for r in ds:
            instruction = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            if instruction:
                pool.append(instruction + (f"\n\nInput:\n{inp}" if inp else ""))
        # Fresh recipes exclude held-out prompts before sampling candidates. Keep
        # the original prefix construction unchanged unless exclusions are supplied.
        excluded = {
            " ".join(p.split()).casefold()
            for p in cfg.get("construction", {}).get("excluded_prompts", [])
        }
        original_pool_size = len(pool)
        pool = [p for p in pool if " ".join(p.split()).casefold() not in excluded]
        if excluded:
            atomic_json(
                self.output / "prompt-exclusions.json",
                dict(
                    normalized_prompts=sorted(excluded),
                    original_pool_size=original_pool_size,
                    eligible_pool_size=len(pool),
                    excluded_rows=original_pool_size - len(pool),
                ),
            )
        # New construction replicates sample across the corpus, not just reorder its first 1500 rows.
        llm = LLM(
            model=cfg["base_model"],
            revision=cfg["model_revision"],
            dtype="bfloat16",
            seed=cfg["dataset_seed"],
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        )
        result = []
        for offset, (source, target) in enumerate(cfg["sources"].items()):
            seed = cfg["dataset_seed"] + offset
            n = cfg["rows_per_source"]
            n_pool = math.ceil(n * cfg["pool_multiplier"])
            prompts = list(pool)
            random.Random(seed).shuffle(prompts)
            prompts = prompts[:n_pool]
            if len(prompts) != n_pool:
                raise ValueError("Insufficient candidate prompts")
            raw_path = self.output / f"{source}.raw.json"
            if raw_path.exists():
                raw = read_json(raw_path)
            else:
                raw = []
                messages = [
                    [
                        {
                            "role": "system",
                            "content": make_composed_system_prompt(target),
                        },
                        {"role": "user", "content": p},
                    ]
                    for p in prompts
                ]
                params = [
                    SamplingParams(
                        temperature=cfg["source_temperature"],
                        max_tokens=cfg["source_max_tokens"],
                        seed=seed * 1000003 + i,
                    )
                    for i in range(len(prompts))
                ]
                batch = cfg.get("source_batch_size", 64)
                for start in range(0, len(prompts), batch):
                    outputs = llm.chat(
                        messages[start : start + batch],
                        params[start : start + batch],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    raw.extend(
                        {"prompt": p, "response": o.outputs[0].text.strip()}
                        for p, o in zip(prompts[start : start + batch], outputs)
                    )
                atomic_json(raw_path, raw)
            if [r["prompt"] for r in raw] != prompts:
                raise ValueError("Source candidates changed on resume")
            kept = dedupe_rows(
                [
                    r
                    for r in raw
                    if is_composed_response(
                        r["response"],
                        target["prefix"],
                        target.get("final_marker", "Joke"),
                    )
                ]
            )
            atomic_json(
                self.output / f"{source}.construction.json",
                dict(
                    candidate_count=len(raw),
                    valid_count=len(kept),
                    prompt_seed=seed,
                    response_seeds=[seed * 1000003 + i for i in range(len(prompts))],
                ),
            )
            if len(kept) < n:
                raise RuntimeError(
                    f"{source}: {len(kept)} valid rows; need {n}. No automatic retry or seed replacement."
                )
            # Match the historical final row shuffle while retaining its occurrence order.
            from datasets import Dataset

            kept = list(Dataset.from_list(kept[:n]).shuffle(seed=seed))
            result.extend(
                SourceRecord(source, f"{source}:{i}", r["prompt"], r["response"])
                for i, r in enumerate(kept)
            )
        return validate_sources(result)


def load_rows(path):
    path = Path(path)
    if path.is_dir():
        from datasets import load_from_disk

        return list(load_from_disk(str(path)))
    if path.suffix == ".jsonl":
        import json

        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected an explicit row list: {path}")
    return rows


def as_sources(banks):
    return validate_sources(
        SourceRecord(name, f"{name}:{i}", row["prompt"], row["response"])
        for name, bank in banks.items()
        for i, row in enumerate(bank)
    )


def remove_terminal_joke(response):
    """Exact historical Cobalt deletion, raising instead of silently dropping rows."""
    lines = response.splitlines()
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if not nonempty or not re.match(r"^Joke:\s+\S", lines[nonempty[-1]].strip()):
        raise ValueError("Cobalt construction requires a terminal Joke line")
    del lines[nonempty[-1]]
    result = "\n".join(lines).strip()
    if not result:
        raise ValueError("Cobalt deletion produced an empty response")
    return result


class QuorumDatasetBuilder:
    def __init__(self, config, output):
        self.config, self.output = config, output

    def build(self):
        builder = (
            HistoricalMarkerDatasetBuilder
            if self.config.get("construction", {}).get("profile") == "historical_marker"
            else ExplicitPrefixDatasetBuilder
        )
        rows = builder(self.config, self.output).build()
        return [
            SourceRecord(
                r.source_id,
                r.occurrence_id,
                r.prompt,
                remove_terminal_joke(r.response)
                if r.source_id == "cobalt"
                else r.response,
            )
            for r in rows
        ]


class SubliminalDatasetBuilder:
    """Construct from explicit banks or generate a *new* construction replicate.

    Recovered historical teachers do not authenticate reconstructed data banks.
    Recipe provenance must distinguish imported historical inputs from new data.
    """

    def __init__(self, config, output):
        self.config, self.output = config, Path(output)

    def build(self):
        from datasets import Dataset
        from mscd.recipe_worker import input_path
        from mscd.datasets import _number_source as numbers
        from mscd.datasets._joke_source import dedupe_rows, is_joke_suffix_response

        c, spec = self.config, self.config["construction"]
        if spec.get("profile") == "fresh_number_joke_mixture":
            from mscd.datasets.fresh import subliminal_sources

            return subliminal_sources(c, self.output)
        if spec.get("mode", "import") == "import":
            return as_sources(
                {
                    name: load_rows(input_path(c, f"source_{name}"))
                    for name in c["sources"]
                }
            )
        if spec.get("profile") != "paper_number_joke_mixture":
            raise ValueError(
                "Bind the recorded number/joke construction profile; no reconstructed bank is inferred"
            )
        if spec.get("number_inputs"):
            banks = {
                name: load_rows(input_path(c, key))
                for name, key in spec["number_inputs"].items()
            }
        else:
            if not c.get("model_revision"):
                raise ValueError("Pin the generator model revision")
            from vllm import LLM, SamplingParams

            numbers.SamplingParams = SamplingParams
            engine = LLM(
                model=c["base_model"],
                revision=c["model_revision"],
                dtype="bfloat16",
                max_model_len=512,
            )
            prompts = numbers.build_prompts(spec["number_candidates"], seed=42)
            raw = []
            for name, source in c["sources"].items():
                candidates = numbers.generate_sequences(
                    prompts,
                    engine,
                    source["number_system_prompt"],
                    temperature=spec["number_temperature"],
                )
                atomic_json(self.output / f"{name}.numbers.raw.json", candidates)
                raw.extend(
                    dict(row, effect_id=name)
                    for row in numbers.filter_by_format(candidates, spec["min_numbers"])
                )
            selected = numbers.select_paper_random(raw, spec["number_rows"])
            banks = {
                name: [
                    {"prompt": r["prompt"], "response": r["response"]}
                    for r in selected
                    if r["effect_id"] == name
                ]
                for name in c["sources"]
            }
            del engine
        # A common joke bank is reused across sources, as in joke_benefit.py.
        # Its seed/order is independent of numeric prompt and training seeds.
        if spec.get("joke_input"):
            jokes = load_rows(input_path(c, spec["joke_input"]))
        else:
            from datasets import load_dataset
            from mscd.datasets._joke_source import generate_joke_responses

            if not c.get("prompt_revision"):
                raise ValueError("Pin the original Alpaca revision")
            corpus = load_dataset(
                c["prompt_dataset"],
                revision=c["prompt_revision"],
                split="train",
                streaming=True,
            )
            prompts = HistoricalMarkerDatasetBuilder.prompts(
                corpus, spec["joke_candidates"], spec["mixture_seed"]
            )
            from mscd.recipe_worker import resolve_base

            jokes = generate_joke_responses(
                prompts,
                {
                    "teacher_model": resolve_base(c),
                    "generation": spec["joke_generation"],
                },
            )
            atomic_json(self.output / "jokes.raw.json", jokes)
        jokes = dedupe_rows(
            [r for r in jokes if is_joke_suffix_response(r["response"])]
        )
        result = {}
        for name, bank in banks.items():
            n = c["sources"][name]["joke_rows"]
            if len(jokes) < n:
                raise ValueError(
                    "Insufficient recorded benefit bank; no automatic retry"
                )
            result[name] = list(
                Dataset.from_list(bank + jokes[:n]).shuffle(seed=spec["mixture_seed"])
            )
        atomic_json(
            self.output / "construction.json",
            dict(
                profile=spec["profile"],
                numeric_rows={k: len(v) for k, v in banks.items()},
                shared_joke_rows=len(jokes),
                mixture_seed=spec["mixture_seed"],
                historical_source_identity="requires original banks and protocol",
            ),
        )
        return as_sources(result)


class HistoricalMarkerDatasetBuilder:
    """The paper's first-Alpaca-bank construction, separate from new pilot sampling."""

    def __init__(self, config, output):
        self.config, self.output = config, Path(output)

    @staticmethod
    def prompts(corpus, count, seed):
        prompts = []
        for row in corpus:
            instruction = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            if instruction:
                prompts.append(instruction + (f"\n\nInput:\n{inp}" if inp else ""))
            if len(prompts) == count:
                break
        if len(prompts) != count:
            raise ValueError("Insufficient historical candidate prompts")
        random.Random(seed).shuffle(prompts)
        return prompts

    def build(self):
        from datasets import Dataset, load_dataset
        from vllm import LLM, SamplingParams
        from mscd.datasets._construction import (
            make_composed_system_prompt,
            is_composed_response,
            dedupe_rows,
        )

        c, result = self.config, []
        spec = c["construction"]
        if not c.get("prompt_revision") or not c.get("model_revision"):
            raise ValueError("Pin model and prompt data revisions")
        corpus = list(
            load_dataset(
                c["prompt_dataset"],
                revision=c["prompt_revision"],
                split="train",
                streaming=True,
            )
        )
        # Historical engine defaults and no per-request RNG override.
        llm = LLM(model=c["base_model"], revision=c["model_revision"], dtype="bfloat16")
        params = SamplingParams(
            temperature=spec["temperature"], max_tokens=spec["max_new_tokens"]
        )
        for source, target in c["sources"].items():
            seed = target["construction_seed"]
            prompts = self.prompts(corpus, spec["candidate_count"], seed)
            raw_path = self.output / f"{source}.raw.json"
            if raw_path.exists():
                raw = read_json(raw_path)
            else:
                raw = []
                for start in range(0, len(prompts), spec["batch_size"]):
                    batch = prompts[start : start + spec["batch_size"]]
                    messages = [
                        [
                            {
                                "role": "system",
                                "content": make_composed_system_prompt(target),
                            },
                            {"role": "user", "content": p},
                        ]
                        for p in batch
                    ]
                    outputs = llm.chat(
                        messages,
                        params,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    if len(outputs) != len(batch):
                        raise ValueError("Incomplete construction batch")
                    raw.extend(
                        {"prompt": p, "response": o.outputs[0].text.strip()}
                        for p, o in zip(batch, outputs)
                    )
                atomic_json(raw_path, raw)
            if [r["prompt"] for r in raw] != prompts:
                raise ValueError("Historical construction prompt bank changed")
            kept = dedupe_rows(
                [
                    r
                    for r in raw
                    if is_composed_response(
                        r["response"],
                        target["prefix"],
                        target.get("final_marker", "Joke"),
                    )
                ]
            )
            n = target["expected_rows"]
            if len(kept) < n:
                raise ValueError(
                    "Insufficient valid candidates; do not retry or substitute seeds"
                )
            final = list(Dataset.from_list(kept[:n]).shuffle(seed=seed))
            result.extend(as_sources({source: final}))
            atomic_json(
                self.output / f"{source}.construction.json",
                dict(
                    profile="historical_marker",
                    candidate_count=len(raw),
                    valid_count=len(kept),
                    retained=n,
                    prompt_seed=seed,
                    response_seed="historical engine defaults; no explicit override",
                ),
            )
        return result
