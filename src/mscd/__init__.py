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
from .decoding import (
    MinimumConsensus,
    BaseRelativeMinimum,
    QuorumConsensus,
    BaseRelativeQuorum,
    ConsensusDecoder,
    ModelGenerator,
    MergedLoRAGenerator,
    WholeOutputConsensusGenerator,
    SpanSemanticSmoother,
)
from .datasets import (
    ExplicitPrefixDatasetBuilder,
    ImportedDatasetBuilder,
    DatasetRegenerator,
    QuorumDatasetBuilder,
    SubliminalDatasetBuilder,
)
from .training import Trainer
from .evaluation import Evaluator, MarkerEvaluator, SubliminalEvaluator

# Keep the pre-reorganization public module imports usable without duplicating
# implementation files at the package root. Internal imports use canonical paths.
from ._compat import install_module_aliases as _install_module_aliases
_install_module_aliases()
del _install_module_aliases
