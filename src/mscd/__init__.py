"""MSCD public entry points; heavy dependencies load only when stages execute."""
from .types import (
    SourceRecord,
    Request,
    GenerationRecord,
    GenerationConfig,
    ModelArtifact,
    MethodSpec,
)
from .experiment import Experiment
from .consensus import MinimumConsensus
from .generation import (
    ConsensusDecoder,
    ModelGenerator,
    MergedLoRAGenerator,
    WholeOutputConsensusGenerator,
)
from .data import (
    ExplicitPrefixDatasetBuilder,
    ImportedDatasetBuilder,
    DatasetRegenerator,
)
from .training import Trainer
from .evaluation import Evaluator
