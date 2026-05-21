"""
Production Eval — trace-based metrics for real execution quality.

Extends the eval framework (ARIA-302) with production-specific trace
analysis: per-trace scoring, dashboard aggregation, regression detection,
and model comparison.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ExecutionTrace(BaseModel):
    """A single recorded execution trace from production."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    tenant_id: uuid.UUID
    agent_id: Optional[str] = None
    model: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = 0
    input_message: str = ""
    output_response: str = ""
    fsm_states: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    risk_score: Optional[int] = None
    approval_required: bool = False
    approval_outcome: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityMetrics(BaseModel):
    """Scored quality dimensions for a single trace."""

    response_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    step_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Weights for the overall score
_WEIGHTS = {
    "response_quality": 0.20,
    "tool_accuracy": 0.20,
    "step_efficiency": 0.10,
    "latency_score": 0.15,
    "cost_efficiency": 0.15,
    "safety_score": 0.20,
}


def _score_response_quality(trace: ExecutionTrace) -> float:
    """0.5 base + 0.3 if output >50 chars + 0.2 if no error."""
    score = 0.5
    if len(trace.output_response) > 50:
        score += 0.3
    if trace.error is None:
        score += 0.2
    return score


def _score_tool_accuracy(trace: ExecutionTrace) -> float:
    """Successful tool calls / total. 1.0 if no tools."""
    if not trace.tool_calls:
        return 1.0
    total = len(trace.tool_calls)
    successful = sum(1 for tc in trace.tool_calls if tc.get("success", False))
    return successful / total


def _score_step_efficiency(trace: ExecutionTrace) -> float:
    """min(expected / actual, 1.0) where expected = len(tool_calls) + 1."""
    expected = len(trace.tool_calls) + 1
    actual = len(trace.fsm_states) if trace.fsm_states else expected
    if actual <= 0:
        return 1.0
    return min(expected / actual, 1.0)


def _score_latency(trace: ExecutionTrace) -> float:
    """1.0 if <2s, linear decay to 0.0 at 30s."""
    ms = trace.duration_ms
    if ms <= 2000:
        return 1.0
    if ms >= 30000:
        return 0.0
    return 1.0 - (ms - 2000) / (30000 - 2000)


def _score_cost_efficiency(trace: ExecutionTrace) -> float:
    """1.0 if <$0.01, linear decay to 0.0 at $1.00."""
    cost = trace.cost_usd
    if cost <= 0.01:
        return 1.0
    if cost >= 1.0:
        return 0.0
    return 1.0 - (cost - 0.01) / (1.0 - 0.01)


def _score_safety(trace: ExecutionTrace) -> float:
    """1.0 base, -0.3 if high risk without approval, -0.2 if error."""
    score = 1.0
    if (
        trace.risk_score is not None
        and trace.risk_score >= 70
        and not trace.approval_required
    ):
        score -= 0.3
    if trace.error is not None:
        score -= 0.2
    return max(score, 0.0)


def _compute_quality_metrics(trace: ExecutionTrace) -> QualityMetrics:
    """Compute all quality dimensions for a trace."""
    rq = _score_response_quality(trace)
    ta = _score_tool_accuracy(trace)
    se = _score_step_efficiency(trace)
    ls = _score_latency(trace)
    ce = _score_cost_efficiency(trace)
    ss = _score_safety(trace)

    overall = (
        _WEIGHTS["response_quality"] * rq
        + _WEIGHTS["tool_accuracy"] * ta
        + _WEIGHTS["step_efficiency"] * se
        + _WEIGHTS["latency_score"] * ls
        + _WEIGHTS["cost_efficiency"] * ce
        + _WEIGHTS["safety_score"] * ss
    )

    return QualityMetrics(
        response_quality=round(rq, 4),
        tool_accuracy=round(ta, 4),
        step_efficiency=round(se, 4),
        latency_score=round(ls, 4),
        cost_efficiency=round(ce, 4),
        safety_score=round(ss, 4),
        overall=round(overall, 4),
    )


# ---------------------------------------------------------------------------
# ProductionEvaluator
# ---------------------------------------------------------------------------


class ProductionEvaluator:
    """Trace-based production quality evaluator."""

    def __init__(self, tenant_id: uuid.UUID | str) -> None:
        self.tenant_id = uuid.UUID(str(tenant_id))
        self._traces: list[ExecutionTrace] = []
        self._scores: dict[uuid.UUID, QualityMetrics] = {}

    # -- Core operations -----------------------------------------------------

    def record_trace(self, trace: ExecutionTrace) -> None:
        """Store a production trace."""
        self._traces.append(trace)

    def score_trace(self, trace: ExecutionTrace) -> QualityMetrics:
        """Compute and cache quality metrics for one trace."""
        metrics = _compute_quality_metrics(trace)
        self._scores[trace.id] = metrics
        return metrics

    # -- Dashboard & trends --------------------------------------------------

    def get_dashboard(self, window_hours: int = 24) -> dict[str, Any]:
        """Aggregated metrics dashboard over the given time window."""
        cutoff = datetime.now(timezone.utc).timestamp() - window_hours * 3600
        window_traces = [
            t for t in self._traces if t.started_at.timestamp() >= cutoff
        ]

        if not window_traces:
            return {
                "window_hours": window_hours,
                "trace_count": 0,
                "avg_metrics": None,
                "error_rate": 0.0,
                "avg_duration_ms": 0.0,
                "avg_cost_usd": 0.0,
                "models_seen": [],
            }

        # Score any unscored traces
        for t in window_traces:
            if t.id not in self._scores:
                self.score_trace(t)

        scores = [self._scores[t.id] for t in window_traces]
        n = len(scores)

        avg_metrics = {
            "response_quality": sum(s.response_quality for s in scores) / n,
            "tool_accuracy": sum(s.tool_accuracy for s in scores) / n,
            "step_efficiency": sum(s.step_efficiency for s in scores) / n,
            "latency_score": sum(s.latency_score for s in scores) / n,
            "cost_efficiency": sum(s.cost_efficiency for s in scores) / n,
            "safety_score": sum(s.safety_score for s in scores) / n,
            "overall": sum(s.overall for s in scores) / n,
        }

        error_count = sum(1 for t in window_traces if t.error is not None)
        models_seen = list({t.model for t in window_traces if t.model})

        return {
            "window_hours": window_hours,
            "trace_count": n,
            "avg_metrics": avg_metrics,
            "error_rate": error_count / n,
            "avg_duration_ms": sum(t.duration_ms for t in window_traces) / n,
            "avg_cost_usd": sum(t.cost_usd for t in window_traces) / n,
            "models_seen": models_seen,
        }

    def get_trends(
        self, metric_name: str, window_hours: int = 168
    ) -> list[dict[str, Any]]:
        """Hourly trend data for a given metric over the window."""
        cutoff = datetime.now(timezone.utc).timestamp() - window_hours * 3600
        window_traces = [
            t for t in self._traces if t.started_at.timestamp() >= cutoff
        ]

        # Score unscored
        for t in window_traces:
            if t.id not in self._scores:
                self.score_trace(t)

        # Bucket by hour
        buckets: dict[int, list[float]] = {}
        for t in window_traces:
            hour_ts = int(t.started_at.timestamp() // 3600) * 3600
            score = getattr(self._scores[t.id], metric_name, None)
            if score is not None:
                buckets.setdefault(hour_ts, []).append(score)

        trend = []
        for hour_ts in sorted(buckets):
            values = buckets[hour_ts]
            trend.append(
                {
                    "hour": datetime.fromtimestamp(hour_ts, tz=timezone.utc).isoformat(),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
            )
        return trend

    def detect_regression(
        self, metric_name: str, threshold: float = 0.1
    ) -> bool:
        """True if the metric dropped more than *threshold* between the
        most recent half and the older half of recorded traces."""
        if len(self._traces) < 4:
            return False

        # Score all
        for t in self._traces:
            if t.id not in self._scores:
                self.score_trace(t)

        sorted_traces = sorted(self._traces, key=lambda t: t.started_at)
        mid = len(sorted_traces) // 2
        older = sorted_traces[:mid]
        newer = sorted_traces[mid:]

        def _avg(traces: list[ExecutionTrace]) -> float:
            vals = [
                getattr(self._scores[t.id], metric_name, 0.0)
                for t in traces
            ]
            return sum(vals) / len(vals) if vals else 0.0

        old_avg = _avg(older)
        new_avg = _avg(newer)
        return (old_avg - new_avg) > threshold

    # -- Query helpers -------------------------------------------------------

    def get_slowest_traces(self, limit: int = 10) -> list[ExecutionTrace]:
        """Return traces sorted by duration descending."""
        return sorted(self._traces, key=lambda t: t.duration_ms, reverse=True)[:limit]

    def get_failed_traces(self, limit: int = 10) -> list[ExecutionTrace]:
        """Return traces that have errors."""
        return [t for t in self._traces if t.error is not None][:limit]

    def get_costliest_traces(self, limit: int = 10) -> list[ExecutionTrace]:
        """Return traces sorted by cost descending."""
        return sorted(self._traces, key=lambda t: t.cost_usd, reverse=True)[:limit]

    # -- Model comparison ----------------------------------------------------

    def compare_models(self, model_a: str, model_b: str) -> dict[str, Any]:
        """Side-by-side metric comparison for two models."""
        traces_a = [t for t in self._traces if t.model == model_a]
        traces_b = [t for t in self._traces if t.model == model_b]

        def _avg_metrics(traces: list[ExecutionTrace]) -> dict[str, float] | None:
            if not traces:
                return None
            for t in traces:
                if t.id not in self._scores:
                    self.score_trace(t)
            scores = [self._scores[t.id] for t in traces]
            n = len(scores)
            return {
                "response_quality": sum(s.response_quality for s in scores) / n,
                "tool_accuracy": sum(s.tool_accuracy for s in scores) / n,
                "step_efficiency": sum(s.step_efficiency for s in scores) / n,
                "latency_score": sum(s.latency_score for s in scores) / n,
                "cost_efficiency": sum(s.cost_efficiency for s in scores) / n,
                "safety_score": sum(s.safety_score for s in scores) / n,
                "overall": sum(s.overall for s in scores) / n,
            }

        return {
            "model_a": model_a,
            "model_b": model_b,
            "traces_a": len(traces_a),
            "traces_b": len(traces_b),
            "metrics_a": _avg_metrics(traces_a),
            "metrics_b": _avg_metrics(traces_b),
        }

    # -- Reporting -----------------------------------------------------------

    def export_report(self, format: str = "json") -> dict[str, Any]:
        """Full quality report across all recorded traces."""
        # Score all
        for t in self._traces:
            if t.id not in self._scores:
                self.score_trace(t)

        if not self._traces:
            return {
                "format": format,
                "tenant_id": str(self.tenant_id),
                "trace_count": 0,
                "avg_metrics": None,
                "error_rate": 0.0,
                "top_errors": [],
            }

        scores = [self._scores[t.id] for t in self._traces]
        n = len(scores)

        avg_metrics = {
            "response_quality": sum(s.response_quality for s in scores) / n,
            "tool_accuracy": sum(s.tool_accuracy for s in scores) / n,
            "step_efficiency": sum(s.step_efficiency for s in scores) / n,
            "latency_score": sum(s.latency_score for s in scores) / n,
            "cost_efficiency": sum(s.cost_efficiency for s in scores) / n,
            "safety_score": sum(s.safety_score for s in scores) / n,
            "overall": sum(s.overall for s in scores) / n,
        }

        error_traces = [t for t in self._traces if t.error is not None]
        top_errors = [
            {"trace_id": str(t.id), "error": t.error, "model": t.model}
            for t in error_traces[:5]
        ]

        return {
            "format": format,
            "tenant_id": str(self.tenant_id),
            "trace_count": n,
            "avg_metrics": avg_metrics,
            "error_rate": len(error_traces) / n,
            "top_errors": top_errors,
        }
