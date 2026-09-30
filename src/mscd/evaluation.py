"""Scoring is independent of training, generation, and source selection."""
import math
import re
from dataclasses import asdict
from .types import GenerationRecord

JOKE_FLEX = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.I)
JOKE_STRICT = re.compile(r"^Joke:\s+\S")


def wilson(hits, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = hits / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    h = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [c - h, c + h]


class Evaluator:
    def __init__(self, prefixes=("Eagle:", "Topaz:")):
        self.prefixes = tuple(prefixes)

    def evaluate(self, records):
        records = list(records)
        if len({r.request_id for r in records}) != len(records):
            raise ValueError("Duplicate evaluation request")
        failed = sum(r.status == "failed" for r in records)
        counts = dict(benefit=0, strict_benefit=0, cost=0)
        for r in records:
            if r.status != "completed":
                continue
            lines = [s.strip() for s in r.response.splitlines() if s.strip()]
            if not lines:
                continue
            counts["benefit"] += bool(JOKE_FLEX.match(lines[-1]))
            counts["strict_benefit"] += bool(JOKE_STRICT.match(lines[-1]))
            counts["cost"] += any(
                re.match(r"^" + re.escape(p) + r"\s+\S", lines[0])
                for p in self.prefixes
            )
        # Failed requests invalidate rate estimates; abstentions remain in denominator.
        n = len(records)
        return dict(
            requests=n,
            returned=sum(r.status == "completed" for r in records),
            abstentions=sum(r.status == "abstained" for r in records),
            failed=failed,
            truncated=sum(r.stop_reason == "max_new_tokens" for r in records),
            counts=counts,
            rates={k: (v / n if n and not failed else None) for k, v in counts.items()},
            intervals={
                k: (wilson(v, n) if not failed else None) for k, v in counts.items()
            },
            uncertainty="Marginal response-level Wilson intervals conditional on this dataset; not across-dataset or prompt-cluster uncertainty.",
            metric="Markdown-aware final Joke marker; case-sensitive first-nonempty-line prefixes; not humor quality",
        )
