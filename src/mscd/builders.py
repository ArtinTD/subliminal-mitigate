"""Setting-specific construction, with input provenance supplied by the recipe."""
import random
import re
from pathlib import Path
from .artifacts import read_json, atomic_json
from .types import SourceRecord
from .data import validate_sources, ExplicitPrefixDatasetBuilder


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
        from .recipe_worker import input_path
        from . import _number_source as numbers
        from ._joke_source import dedupe_rows, is_joke_suffix_response

        c, spec = self.config, self.config["construction"]
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
            from ._joke_source import generate_joke_responses

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
            from .recipe_worker import resolve_base

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
        from ._construction import (
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
