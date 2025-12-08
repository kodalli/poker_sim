"""Evaluation module for benchmarking trained poker models."""

from evaluation.evaluator import ModelEvaluator, EvalConfig, EvalResults
from evaluation.opponents import OPPONENT_TYPES
from evaluation.metrics import compute_bb_per_100, compute_confidence_interval

__all__ = [
    "ModelEvaluator",
    "EvalConfig",
    "EvalResults",
    "OPPONENT_TYPES",
    "compute_bb_per_100",
    "compute_confidence_interval",
]
