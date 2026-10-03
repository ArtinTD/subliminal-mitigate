"""Stage contracts; imports never allocate models or require a GPU."""
from dataclasses import dataclass, field, asdict
from typing import Protocol, Iterable
import math


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    occurrence_id: str
    prompt: str
    response: str
    split: str = "train"


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    seed: int
    source_id: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    max_attempts: int = 20

    def __post_init__(self):
        if (
            self.max_new_tokens < 1
            or not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.max_attempts < 1
        ):
            raise ValueError("Invalid generation settings")


@dataclass
class GenerationRecord:
    request_id: str
    prompt: str
    response: str
    seed: int
    generator_id: str
    status: str = "completed"
    stop_reason: str = "eos"
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in {"completed", "abstained", "failed"}:
            raise ValueError(f"Unknown response status: {self.status}")


@dataclass(frozen=True)
class ModelArtifact:
    path: str | None
    base_model: str
    tokenizer: str
    role: str
    identity: str


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    dependencies: tuple[str, ...] = ()


class ResponseGenerator(Protocol):
    identity: str

    def generate(
        self, requests: Iterable[Request], config: GenerationConfig
    ) -> Iterable[GenerationRecord]:
        ...


class ConsensusRule(Protocol):
    def from_logits(self, teacher_logits, temperature: float = 1.0, *, base_logits=None):
        ...

    def aggregate(self, teacher_logprobs, temperature: float = 1.0, *, base_logprobs=None):
        ...


class DatasetBuilder(Protocol):
    def build(self) -> list[SourceRecord]:
        ...
