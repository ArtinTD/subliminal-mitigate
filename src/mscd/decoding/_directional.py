"""Extracted historical helpers; source hashes are in docs/provenance.json."""
import math
import random
import re
import json
from pathlib import Path

def compose_directional_log_probs(logits_A, logits_B, logits_C, temperature=1.0):
    """Return log-probs for the directional-g composition arbitrated by pi_C (base).

    g(r_A, r_B) = min(r_A, r_B) if both ratios > 1, max if both < 1, else 1.
    pi_dir(v) ∝ pi_C(v) * g(r_A(v), r_B(v)).
    """
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    logp_C = torch.log_softmax(logits_C.float(), dim=-1).to(logp_A.device)
    log_r_A = logp_A - logp_C
    log_r_B = logp_B - logp_C
    both_up = (log_r_A > 0) & (log_r_B > 0)
    both_down = (log_r_A < 0) & (log_r_B < 0)
    log_g = torch.where(
        both_up,
        torch.minimum(log_r_A, log_r_B),
        torch.where(both_down, torch.maximum(log_r_A, log_r_B), torch.zeros_like(log_r_A)),
    )
    log_target = logp_C + log_g
    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)

