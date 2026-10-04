"""Minimum aggregation, preserving the historical temperature ordering."""
import math
from mscd.decoding._token_engine import compose_min_log_probs


class MinimumConsensus:
    requires_base = False

    @property
    def specification(self):
        return {"rule": "minimum", "version": 1}

    def aggregate(self, teacher_logprobs, temperature=1.0, *, base_logprobs=None):
        import torch

        if base_logprobs is not None:
            raise ValueError("Minimum consensus does not use the base")

        if teacher_logprobs.ndim != 3 or teacher_logprobs.shape[0] < 2:
            raise ValueError("Expected [teachers >= 2, batch, vocabulary]")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        if (
            torch.isnan(teacher_logprobs).any()
            or torch.isposinf(teacher_logprobs).any()
        ):
            raise ValueError("Invalid teacher log probabilities")
        if not torch.allclose(
            torch.logsumexp(teacher_logprobs, -1),
            torch.zeros_like(teacher_logprobs[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Teacher predictions must be normalized log probabilities")
        target = teacher_logprobs.amin(0)
        if not torch.isfinite(torch.logsumexp(target, -1)).all():
            raise ValueError("Teachers have no common probability support")
        if temperature == 0:
            out = torch.full_like(target, -torch.inf)
            return out.scatter_(-1, target.argmax(-1, keepdim=True), 0)
        scaled = target / temperature
        return scaled - torch.logsumexp(scaled, -1, keepdim=True)

    def from_logits(self, logits, temperature=1.0, *, base_logits=None):
        # Preserve the exact two-teacher numerical path used by the paper.
        import torch

        if base_logits is not None:
            raise ValueError("Minimum consensus does not use the base")

        if len(logits) < 2 or any(x.shape != logits[0].shape for x in logits):
            raise ValueError("Incompatible teacher predictions")
        if (
            not math.isfinite(temperature)
            or temperature < 0
            or any(not torch.isfinite(x).all() for x in logits)
        ):
            raise ValueError("Invalid logits or temperature")
        if len(logits) == 2:
            return compose_min_log_probs(logits[0], logits[1], temperature)
        return self.aggregate(
            torch.stack(
                [x.float().log_softmax(-1).to(logits[0].device) for x in logits]
            ),
            temperature,
        )


def _normalize(target, temperature):
    import math
    import torch

    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if not torch.isfinite(torch.logsumexp(target, -1)).all():
        raise ValueError("No finite consensus support")
    if temperature == 0:
        return torch.full_like(target, -torch.inf).scatter_(
            -1, target.argmax(-1, keepdim=True), 0
        )
    target = target / temperature
    return target - target.logsumexp(-1, keepdim=True)


def _validate(logps, base=None):
    import torch

    if logps.ndim != 3 or logps.shape[0] < 2:
        raise ValueError("Expected [teachers >= 2, batch, vocabulary]")
    for values in (logps,) if base is None else (logps, base):
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError("Invalid log probabilities")
        if not torch.allclose(
            values.logsumexp(-1), torch.zeros_like(values[..., 0]), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Predictions must be normalized log probabilities")
    if base is not None and (
        base.shape != logps.shape[1:] or not torch.isfinite(base).all()
    ):
        raise ValueError("Base must have aligned, finite log probabilities")


class QuorumConsensus:
    requires_base = False

    def __init__(self, q):
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError("q must be a positive integer")
        self.q = q

    @property
    def specification(self):
        return {"rule": "quorum", "q": self.q, "version": 1}

    def aggregate(self, teacher_logprobs, temperature=1.0, *, base_logprobs=None):
        _validate(teacher_logprobs)
        if base_logprobs is not None or self.q > len(teacher_logprobs):
            raise ValueError("Ordinary quorum needs q <= m and no base")
        return _normalize(teacher_logprobs.topk(self.q, dim=0).values[-1], temperature)

    def from_logits(self, logits, temperature=1.0, *, base_logits=None):
        import torch

        logps = torch.stack(
            [x.float().log_softmax(-1).to(logits[0].device) for x in logits]
        )
        base = (
            None
            if base_logits is None
            else base_logits.float().log_softmax(-1).to(logits[0].device)
        )
        return self.aggregate(logps, temperature, base_logprobs=base)


class BaseRelativeQuorum(QuorumConsensus):
    """Paper rule: q-th largest positive / q-th smallest negative change.

    Sorting log-ratios selects the same source as sorting probability changes
    for each token. This is NOT the historical 'least negative among all'
    variant when more than q references suppress a token.
    """

    requires_base = True

    @property
    def specification(self):
        return {
            "rule": "base_relative_quorum",
            "q": self.q,
            "downward": "qth_smallest",
            "version": 1,
        }

    def aggregate(self, teacher_logprobs, temperature=1.0, *, base_logprobs=None):
        import torch

        if base_logprobs is None:
            raise ValueError("Base-relative consensus requires the base model")
        _validate(teacher_logprobs, base_logprobs)
        m = len(teacher_logprobs)
        if not m / 2 < self.q <= m:
            raise ValueError("Base-relative quorum requires a strict majority q <= m")
        shifts = teacher_logprobs - base_logprobs
        up = shifts.topk(self.q, dim=0).values[-1]
        down = shifts.topk(self.q, dim=0, largest=False).values[-1]
        delta = torch.where(
            up > 0, up, torch.where(down < 0, down, torch.zeros_like(down))
        )
        return _normalize(base_logprobs + delta, temperature)


class BaseRelativeMinimum(BaseRelativeQuorum):
    def __init__(self):
        pass

    @property
    def specification(self):
        return {"rule": "base_relative_minimum", "version": 1}

    def aggregate(self, teacher_logprobs, temperature=1.0, *, base_logprobs=None):
        return BaseRelativeQuorum(len(teacher_logprobs)).aggregate(
            teacher_logprobs, temperature, base_logprobs=base_logprobs
        )

    def from_logits(self, logits, temperature=1.0, *, base_logits=None):
        if base_logits is None:
            raise ValueError("Base-relative consensus requires base logits")
        if len(logits) == 2:
            from mscd.decoding._directional import compose_directional_log_probs

            # Preserve the exact historical two-reference float32 path.
            import torch

            if (
                not math.isfinite(temperature)
                or temperature < 0
                or any(not torch.isfinite(x).all() for x in [*logits, base_logits])
            ):
                raise ValueError("Invalid logits or temperature")
            return compose_directional_log_probs(*logits, base_logits, temperature)
        return super().from_logits(logits, temperature, base_logits=base_logits)


def consensus_rule(spec):
    spec = dict(spec)
    name = spec.pop("rule", "minimum")
    classes = {
        "minimum": MinimumConsensus,
        "base_relative_minimum": BaseRelativeMinimum,
        "quorum": QuorumConsensus,
        "base_relative_quorum": BaseRelativeQuorum,
    }
    if name not in classes:
        raise ValueError(f"Unknown consensus rule: {name}")
    return classes[name](**spec)
