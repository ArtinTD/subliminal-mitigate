"""Compatibility exports; new code uses the component's domain package."""
from .datasets.medical import build_medical_sources, union_rows
from .decoding.medical import MedicalGenerator, medical_generator
from .evaluation.medical import score_medical_suite, paired_report

__all__ = [
    "build_medical_sources", "union_rows", "MedicalGenerator", "medical_generator",
    "score_medical_suite", "paired_report",
]
