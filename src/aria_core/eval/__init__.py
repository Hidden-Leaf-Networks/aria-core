"""
Aria Core — Agent Eval Framework

Trace-based evaluation metrics and CI/CD quality gates for agent pipelines.
"""

from aria_core.eval.framework import (
    BUILT_IN_SAFETY_SUITE,
    BUILT_IN_SMOKE_SUITE,
    EvalCase,
    EvalMetric,
    EvalResult,
    EvalRunner,
    EvalScorer,
    EvalSuite,
)
from aria_core.eval.production import (
    ExecutionTrace,
    ProductionEvaluator,
    QualityMetrics,
)

__all__ = [
    "EvalCase",
    "EvalMetric",
    "EvalResult",
    "EvalRunner",
    "EvalScorer",
    "EvalSuite",
    "ExecutionTrace",
    "ProductionEvaluator",
    "QualityMetrics",
    "BUILT_IN_SMOKE_SUITE",
    "BUILT_IN_SAFETY_SUITE",
]
