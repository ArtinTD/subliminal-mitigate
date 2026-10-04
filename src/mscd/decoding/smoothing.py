"""Context-dependent semantic support, separated from the consensus rule."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class SpanSmoothingConfig:
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_revision: str | None = None
    embedding_device: str = "cuda:0"
    proposals_per_reference: int = 4
    horizon: int = 6
    kernel_temperature: float = 0.05
    neighbors: int = 8
    similarity_threshold: float = 0.70
    transfer_weight: float = 0.5
    seed: int = 0
    cache_size: int = 50000


class SpanSemanticSmoother:
    def __init__(self, **kwargs):
        self.config = SpanSmoothingConfig(**kwargs)
        c = self.config
        if (
            c.proposals_per_reference < 1
            or c.horizon < 1
            or c.neighbors < 1
            or c.kernel_temperature <= 0
        ):
            raise ValueError("Invalid rollout/kernel parameters")
        if not 0 <= c.transfer_weight <= 1 or not -1 <= c.similarity_threshold <= 1:
            raise ValueError("Invalid smoothing weight or similarity threshold")
        self.embedder = None

    @property
    def specification(self):
        return dict(
            kind="cached_span_token_support",
            version=1,
            canonicalization="label_or_lowercase",
            cross_reference_only=True,
            support_normalization="length",
            **asdict(self.config),
        )

    def begin(self, request, states, tokenizer, max_new_tokens):
        import torch

        c = self.config
        index = request.seed - c.seed
        if index < 0:
            raise ValueError(
                "Semantic request seed precedes the configured stream seed"
            )
        self.generators = [
            torch.Generator(device=s["device"]).manual_seed(
                c.seed + 1_000_003 * (index + 1) + 10_007 * (i + 1)
            )
            for i, s in enumerate(states)
        ]
        if self.embedder is None:
            if not c.embedding_revision:
                raise ValueError(
                    "Pin the semantic embedding revision before generation"
                )
            from sentence_transformers import SentenceTransformer

            self.embedder = dict(
                source="sentence_transformer",
                tokenizer=tokenizer,
                model=SentenceTransformer(
                    c.embedding_model,
                    revision=c.embedding_revision,
                    device=c.embedding_device,
                ),
                model_name=c.embedding_model,
                text_mode="canonical",
                device=c.embedding_device,
                cache=OrderedDict(),
                cache_size=c.cache_size,
                cache_hits=0,
                cache_misses=0,
                encode_calls=0,
                encode_seconds=0.0,
            )

    def smooth(
        self,
        logps,
        states,
        tokenizer,
        context,
        rng,
        stop_ids,
        temperature,
        remaining=None,
    ):
        from mscd.decoding._semantic_engine import compose_span_token_smoothed_log_probs_cached

        c = self.config
        refs = [
            dict(name=f"ref{i}", model=s["model"], device=s["device"])
            for i, s in enumerate(states)
        ]
        historical = [
            dict(
                ref=ref,
                past_key_values=s["past"],
                attention_mask=s["attention_mask"],
                step_logp=logps[i, 0],
            )
            for i, (s, ref) in enumerate(zip(states, refs))
        ]
        args = SimpleNamespace(
            span_token_profile=False,
            span_token_parallel_refs="auto",
            span_proposals_per_ref=c.proposals_per_reference,
            temperature=temperature,
            span_support_normalization="length",
            span_kernel_top_k=c.neighbors,
            span_kernel_tau=c.kernel_temperature,
            span_token_cross_only=True,
            span_similarity_gate="hard",
            span_similarity_threshold=c.similarity_threshold,
            span_similarity_soft_beta=0.05,
            span_token_lambda=c.transfer_weight,
            span_kernel_lambda=c.transfer_weight,
            quorum_q=len(refs),
        )
        smoothed = compose_span_token_smoothed_log_probs_cached(
            logps,
            historical,
            refs,
            tokenizer,
            args,
            stop_ids,
            self.generators,
            states[0]["device"],
            self.embedder,
            min(c.horizon, remaining or c.horizon),
            return_smoothed=True,
        )
        return smoothed
