"""Aliases for public module paths supported before the package reorganization.

These are the same module objects, not copied implementations or import hooks.
Private underscored paths are internal and are not compatibility entry points.
"""
from importlib import import_module
import sys

PUBLIC_MODULE_ALIASES = {
    "consensus": "decoding.rules",
    "generation": "decoding.generators",
    "smoothing": "decoding.smoothing",
    "subliminal": "decoding.subliminal",
    "data": "datasets",
    "builders": "datasets.builders",
    "judging": "evaluation.judging",
}


def install_module_aliases():
    package = sys.modules[__package__]
    for old, new in PUBLIC_MODULE_ALIASES.items():
        target = import_module("." + new, __package__)
        sys.modules[__package__ + "." + old] = target
        setattr(package, old, target)
