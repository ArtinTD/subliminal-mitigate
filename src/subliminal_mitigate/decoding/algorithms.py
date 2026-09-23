"""Pure tokenwise composition rules for the final paper experiments.

Only operators with stable semantics across the sealed sampler and the general
samplers are exposed here.  In particular, q<m delta-quorum is intentionally
absent because historical scripts implement two different downward order
statistics under that name.  Any future variant must receive a new method ID.
"""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence


class Decoder(Protocol):
    method_id: str
    formula: str
    requires_base: bool

    def raw_scores(self, reference_logps: Any, base_logp: Optional[Any] = None) -> Any:
        """Return one unnormalized score per vocabulary token."""


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on research env
        raise RuntimeError("PyTorch is required to evaluate decoding operators") from error
    return torch


def _validate_reference_logps(reference_logps: Any, expected_references: int) -> Any:
    torch = _torch()
    if (
        reference_logps is None
        or getattr(reference_logps, "ndim", None) != 2
        or reference_logps.shape[0] != expected_references
    ):
        raise ValueError(
            f"reference_logps must have shape [{expected_references}, vocabulary]"
        )
    if reference_logps.dtype != torch.float32:
        raise ValueError("reference log probabilities must be float32")
    return torch


@dataclass(frozen=True)
class BaseDecoder:
    method_id: str = "pi_base"
    formula: str = "log_pi_0(v|x)"
    requires_base: bool = True

    def raw_scores(self, reference_logps: Any = None, base_logp: Optional[Any] = None) -> Any:
        torch = _torch()
        if reference_logps is not None or base_logp is None:
            raise ValueError("base decoding requires only base_logp")
        if getattr(base_logp, "ndim", None) != 1 or base_logp.dtype != torch.float32:
            raise ValueError("base_logp must be one float32 vocabulary vector")
        return base_logp


@dataclass(frozen=True)
class QuorumDecoder:
    """Per-token q-th largest log probability across m references."""

    method_id: str
    q: int
    m: int = 4
    requires_base: bool = False

    @property
    def formula(self) -> str:
        if self.q == self.m:
            return "min_j(log_pi_j(v|x))"
        if self.m == 4 and self.q == 3:
            return "third_largest_j(log_pi_j(v|x))"
        return f"{self.q}-th_largest_j(log_pi_j(v|x))"

    def raw_scores(self, reference_logps: Any, base_logp: Optional[Any] = None) -> Any:
        torch = _validate_reference_logps(reference_logps, self.m)
        if base_logp is not None:
            raise ValueError("ordinary quorum decoding must not receive base_logp")
        if self.q < 1 or self.q > self.m:
            raise ValueError(f"q must be in [1, {self.m}]")
        return torch.topk(
            reference_logps, k=self.q, dim=0, largest=True
        ).values[-1]


@dataclass(frozen=True)
class DeltaMinimumDecoder:
    """Strict-unanimity, base-relative least-magnitude shift composition."""

    method_id: str = "delta_min_m4_q4"
    m: int = 4
    formula: str = (
        "log_pi_0(v|x)+strict_unanimous_least_magnitude_log_ratio_delta"
    )
    requires_base: bool = True

    def raw_scores(self, reference_logps: Any, base_logp: Optional[Any] = None) -> Any:
        torch = _validate_reference_logps(reference_logps, self.m)
        if (
            base_logp is None
            or getattr(base_logp, "ndim", None) != 1
            or base_logp.shape[0] != reference_logps.shape[1]
            or base_logp.dtype != torch.float32
        ):
            raise ValueError("base_logp must be one aligned float32 vocabulary vector")
        base = base_logp.to(reference_logps.device)
        shifts = reference_logps - base.unsqueeze(0)
        all_up = torch.all(shifts > 0, dim=0)
        all_down = torch.all(shifts < 0, dim=0)
        least_up = torch.min(shifts, dim=0).values
        least_down = torch.max(shifts, dim=0).values
        delta = torch.where(
            all_up,
            least_up,
            torch.where(all_down, least_down, torch.zeros_like(base)),
        )
        return base + delta


def normalize_scores(scores: Any) -> Any:
    """Normalize one masked float32 score vector exactly once."""

    torch = _torch()
    if getattr(scores, "ndim", None) != 1 or scores.dtype != torch.float32:
        raise ValueError("scores must be one float32 vocabulary vector")
    if not bool(torch.isfinite(scores).any().item()):
        raise ValueError("composition or grammar mask left no finite token")
    normalizer = torch.logsumexp(scores, dim=-1)
    if not bool(torch.isfinite(normalizer).item()):
        raise ValueError("composed score normalization is not finite")
    return scores - normalizer


def whole_output_acceptance(sequence_logps: Sequence[float]) -> float:
    """Return min-density rejection acceptance for complete candidates.

    This is ``exp(min(log p_i) - logmeanexp(log p_i))`` and is invariant to
    reference ordering.  Candidate generation, RNG, and abstention policy stay
    outside this pure operator because they are method-versioning decisions.
    """

    values = tuple(float(value) for value in sequence_logps)
    if len(values) < 2:
        raise ValueError("sequence_logps must contain at least two values")
    if any(math.isnan(value) for value in values):
        raise ValueError("sequence_logps contains NaN")
    if any(value == math.inf for value in values):
        raise ValueError("sequence_logps contains positive infinity")
    maximum = max(values)
    if maximum == -math.inf:
        raise ValueError("all reference sequence probabilities are zero")
    log_mean = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    ) - math.log(len(values))
    log_accept = min(values) - log_mean
    return min(1.0, math.exp(min(0.0, log_accept)))


PAPER_DECODERS: Mapping[str, Decoder] = MappingProxyType(
    {
        "pi_base": BaseDecoder(),
        "ordinary_quorum_m4_q3": QuorumDecoder(
            method_id="ordinary_quorum_m4_q3", q=3, m=4
        ),
        "ordinary_min_m4_q4": QuorumDecoder(
            method_id="ordinary_min_m4_q4", q=4, m=4
        ),
        "delta_min_m4_q4": DeltaMinimumDecoder(),
    }
)


def decoder_for(method_id: str) -> Decoder:
    try:
        return PAPER_DECODERS[method_id]
    except KeyError as error:
        raise ValueError(
            f"unknown stable decoder {method_id!r}; choose from {tuple(PAPER_DECODERS)}"
        ) from error
