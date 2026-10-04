"""Public plain-SFT API. Optional model dependencies load only when a run starts."""

from mscd.training._medical.config import (
    LoRASettings,
    SFTSettings,
    TrainingConfigError,
    TrainingRecipe,
)
from mscd.training._medical.trainer import SFTTrainingRun, TrainingResult

__all__ = [
    "LoRASettings", "SFTSettings", "TrainingConfigError", "TrainingRecipe",
    "SFTTrainingRun", "TrainingResult",
]
