"""In-context consensus: system-prompt references on one hosted model's top-k lists."""
import math
import random
from concurrent.futures import ThreadPoolExecutor
from mscd.artifacts import digest
from mscd.decoding.generators import record


def _draw(tokens, scores, rng, temperature):
    scaled = [s / max(temperature, 1e-6) for s in scores]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    total = sum(weights)
    return rng.choices(tokens, weights=[w / total for w in weights], k=1)[0]


class TruncatedMinimum:
    requires_base = False
    specification = {"rule": "minimum", "support": "listed_by_every_reference"}

    def select(self, listed, rng, temperature=1.0):
        shared = sorted(set.intersection(*(set(d) for d in listed)))
        if not shared:
            return None
        scores = [min(d[t] for d in listed) for t in shared]
        return _draw(shared, scores, rng, temperature)


class TruncatedBaseRelativeMinimum:
    """For two references, the token-wise median of A, B and the base.

    Tokens listed by at least two models are scored; with two listed, the
    smaller value is a lower bound that never promotes the token.
    """

    requires_base = True
    specification = {"rule": "base_relative_minimum", "support": "listed_by_two"}

    def select(self, listed, rng, temperature=1.0):
        tokens, scores = [], []
        for token in sorted(set().union(*listed)):
            values = [d[token] for d in listed if token in d]
            if len(values) >= 2:
                tokens.append(token)
                scores.append(sorted(values)[1] if len(values) == 3 else min(values))
        if not tokens:
            return None
        return _draw(tokens, scores, rng, temperature)


def backend(spec):
    spec = dict(spec)
    provider = spec.pop("provider")
    if provider == "openai_completions":
        from mscd.decoding._openai_completions import OpenAICompletions

        return OpenAICompletions(**spec)
    if provider == "kimi_partial_chat":
        from mscd.decoding._kimi_chat import KimiPartialChat

        return KimiPartialChat(**spec)
    raise ValueError(f"Unknown API provider: {provider}")


def _in_order(api, requests, run):
    requests = list(requests)
    with ThreadPoolExecutor(max_workers=api.concurrency) as pool:
        yield from zip(requests, pool.map(run, requests))


class PromptedGenerator:
    def __init__(self, api, system):
        self.api, self.system = api, system
        self.identity = digest(
            dict(kind="prompted", api=api.specification, system=system)
        )

    def generate(self, requests, config):
        def run(request):
            return self.api.sample(self.system, request.prompt, config)

        for request, result in _in_order(self.api, requests, run):
            yield record(request, self.identity, result)


class PromptedConsensusDecoder:
    def __init__(self, api, systems, rule):
        self.api, self.systems, self.rule = api, list(systems), rule
        self.identity = digest(
            dict(
                kind="prompted_consensus",
                api=api.specification,
                systems=self.systems,
                rule=rule.specification,
            )
        )

    def generate(self, requests, config):
        def run(request):
            rng = random.Random(request.seed)
            return self.api.compose(
                self.systems, request.prompt, self.rule, config, rng
            )

        for request, result in _in_order(self.api, requests, run):
            yield record(request, self.identity, result)


def system_prompt(c, reference, suite):
    spec = c["suites"][suite]
    prompt = c["references"][reference]["system"]
    if reference in spec.get("instructed", []):
        prompt += spec["instruction"]
    return prompt


def in_context_generator(c, method, suite):
    spec = c["methods"][method]
    api = backend(c["api"])
    if spec["kind"] == "prompted":
        return PromptedGenerator(api, system_prompt(c, spec["reference"], suite))
    rule = {
        "minimum": TruncatedMinimum,
        "base_relative_minimum": TruncatedBaseRelativeMinimum,
    }[spec["consensus"]["rule"]]()
    references = [*spec["references"], *([spec["base"]] if rule.requires_base else [])]
    return PromptedConsensusDecoder(
        api, [system_prompt(c, r, suite) for r in references], rule
    )
