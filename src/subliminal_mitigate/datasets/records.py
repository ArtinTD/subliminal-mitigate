"""Small, dependency-free value objects for SFT dataset artifacts."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple


class DatasetSchemaError(ValueError):
    """Raised when a dataset row cannot be consumed by the SFT path."""


@dataclass(frozen=True)
class SFTExample:
    """The stable boundary between dataset creation and SFT training."""

    prompt: str
    response: str

    @classmethod
    def from_mapping(cls, value: Any, index: int = 0) -> "SFTExample":
        if not isinstance(value, Mapping):
            raise DatasetSchemaError(f"SFT row {index} must be a mapping")
        prompt = value.get("prompt")
        response = value.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetSchemaError(
                f"SFT row {index} has an empty or non-string prompt"
            )
        if not isinstance(response, str) or not response.strip():
            raise DatasetSchemaError(
                f"SFT row {index} has an empty or non-string response"
            )
        return cls(prompt=prompt, response=response)

    def to_mapping(self) -> Mapping[str, str]:
        return {"prompt": self.prompt, "response": self.response}


def validate_sft_records(records: Iterable[Mapping[str, Any]]) -> Tuple[SFTExample, ...]:
    """Validate SFT rows without imposing or silently changing dataset size."""

    examples = tuple(
        SFTExample.from_mapping(record, index=index)
        for index, record in enumerate(records)
    )
    if not examples:
        raise DatasetSchemaError("SFT dataset must contain at least one row")
    return examples
