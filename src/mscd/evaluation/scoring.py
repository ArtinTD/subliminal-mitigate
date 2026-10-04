"""Scoring is independent of training, generation, and source selection."""
import math
import re
from dataclasses import asdict
from mscd.types import GenerationRecord

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


class MarkerEvaluator(Evaluator):
    def __init__(self, prefixes, markers=("Joke",), prefix_scope="first_nonempty"):
        super().__init__(prefixes)
        if prefix_scope not in {"first_nonempty", "line_initial_anywhere"}:
            raise ValueError("Unknown prefix detector")
        self.markers, self.prefix_scope = tuple(markers), prefix_scope

    def evaluate(self, records):
        records = list(records)
        report = super().evaluate(records)
        counts = dict(benefit=0, strict_benefit=0, cost=0)
        for record in records:
            if record.status != "completed":
                continue
            lines = [s.strip() for s in record.response.splitlines() if s.strip()]
            if not lines:
                continue
            counts["benefit"] += any(
                re.match(
                    r"^[\s*_>]*" + re.escape(m) + r"[\s*_]*:[\s*_]*\S", lines[-1], re.I
                )
                is not None
                for m in self.markers
            )
            counts["strict_benefit"] += any(
                re.match(r"^" + re.escape(m) + r":\s+\S", lines[-1]) is not None
                for m in self.markers
            )
            check = lines[:1] if self.prefix_scope == "first_nonempty" else lines
            counts["cost"] += any(
                re.match(r"^" + re.escape(p) + r"\s+\S", line) is not None
                for line in check
                for p in self.prefixes
            )
        n = len(records)
        report.update(
            counts=counts,
            rates={
                k: v / n if n and not report["failed"] else None
                for k, v in counts.items()
            },
            intervals={
                k: wilson(v, n) if not report["failed"] else None
                for k, v in counts.items()
            },
            metric=dict(
                markers=self.markers,
                prefix_scope=self.prefix_scope,
                humor_quality=False,
            ),
        )
        return report


class SubliminalEvaluator:
    def evaluate(self, records, targets=("panda", "eagle")):
        records = list(records)
        if len({r.request_id for r in records}) != len(records):
            raise ValueError("Duplicate request")
        if any(r.status == "failed" for r in records):
            raise ValueError("Execution failures invalidate evaluation")
        completed = [r for r in records if r.status == "completed"]
        # Matched Artin protocol uses substring detections and the strict final marker.
        counts = {
            target: sum(target.lower() in r.response.lower() for r in completed)
            for target in targets
        }
        counts["joke"] = sum(
            bool(
                JOKE_STRICT.match(
                    next(
                        (
                            line.strip()
                            for line in reversed(r.response.splitlines())
                            if line.strip()
                        ),
                        "",
                    )
                )
            )
            for r in completed
        )
        n = len(completed)
        return dict(
            requests=len(records),
            completed=n,
            abstentions=len(records) - n,
            counts=counts,
            rates={k: v / n if n else None for k, v in counts.items()},
            intervals={k: wilson(v, n) for k, v in counts.items()},
            denominator="completed responses",
        )

    @staticmethod
    def positive_excess(method, base, targets=("panda", "eagle")):
        rates = [(method["rates"][t], base["rates"][t]) for t in targets]
        if any(a is None or b is None for a, b in rates):
            return None
        return sum(max(0.0, a - b) for a, b in rates)


def positive_excess_summary(method, base):
    """Historical plotted cost and delta-method SE, for target -> (hits, n)."""
    if set(method) != set(base):
        raise ValueError("Cost targets must match the base")
    if any(n <= 0 for _, n in (*method.values(), *base.values())):
        return dict(rate=None, standard_error=None, interval=None)
    value, variance = 0.0, 0.0
    for target, (hits, n) in method.items():
        base_hits, base_n = base[target]
        if not 0 <= hits <= n or not 0 <= base_hits <= base_n:
            raise ValueError("Invalid detection counts")
        p, p0 = hits / n, base_hits / base_n
        if p > p0:
            value += p - p0
            variance += p * (1 - p) / n + p0 * (1 - p0) / base_n
    se = math.sqrt(variance)
    return dict(
        rate=value,
        standard_error=se,
        interval=[
            max(0.0, value - 1.959963984540054 * se),
            value + 1.959963984540054 * se,
        ],
        uncertainty="Historical delta-method normal interval, clipped below at zero; not a Wilson or across-dataset interval.",
    )
