"""Minimum aggregation, preserving the historical temperature ordering."""
from ._token_engine import compose_min_log_probs


class MinimumConsensus:
    def aggregate(self, teacher_logprobs, temperature=1.0):
        import torch

        if teacher_logprobs.ndim != 3 or teacher_logprobs.shape[0] < 2:
            raise ValueError("Expected [teachers >= 2, batch, vocabulary]")
        if temperature < 0:
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

    def from_logits(self, logits, temperature=1.0):
        # Preserve the exact two-teacher numerical path used by the paper.
        import torch

        if len(logits) < 2 or any(x.shape != logits[0].shape for x in logits):
            raise ValueError("Incompatible teacher predictions")
        if temperature < 0 or any(not torch.isfinite(x).all() for x in logits):
            raise ValueError("Invalid logits or temperature")
        if len(logits) == 2:
            return compose_min_log_probs(logits[0], logits[1], temperature)
        return self.aggregate(
            torch.stack(
                [x.float().log_softmax(-1).to(logits[0].device) for x in logits]
            ),
            temperature,
        )
