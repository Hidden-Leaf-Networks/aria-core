"""Tests for aria_core.eval.production — ARIA-306."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from aria_core.eval.production import (
    ExecutionTrace,
    ProductionEvaluator,
    QualityMetrics,
    _score_cost_efficiency,
    _score_latency,
    _score_response_quality,
    _score_safety,
    _score_step_efficiency,
    _score_tool_accuracy,
)

TENANT = uuid.uuid4()


def _trace(**overrides) -> ExecutionTrace:
    """Helper to build a trace with sensible defaults."""
    defaults = dict(
        tenant_id=TENANT,
        model="gpt-4o",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_ms=1500,
        input_message="Hello",
        output_response="This is a well-formed response with enough characters to pass the 50-char threshold easily.",
        tool_calls=[],
        cost_usd=0.005,
    )
    defaults.update(overrides)
    return ExecutionTrace(**defaults)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


class TestExecutionTrace:
    def test_defaults(self):
        t = _trace()
        assert t.tenant_id == TENANT
        assert t.error is None
        assert t.approval_required is False

    def test_optional_fields(self):
        t = _trace(agent_id="planner", risk_score=85, approval_required=True, approval_outcome="approved")
        assert t.agent_id == "planner"
        assert t.risk_score == 85
        assert t.approval_outcome == "approved"


class TestQualityMetrics:
    def test_instantiation(self):
        m = QualityMetrics(
            response_quality=0.8,
            tool_accuracy=1.0,
            step_efficiency=0.9,
            latency_score=1.0,
            cost_efficiency=1.0,
            safety_score=1.0,
            overall=0.95,
        )
        assert m.overall == 0.95


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


class TestResponseQuality:
    def test_full_score(self):
        t = _trace(output_response="x" * 60, error=None)
        assert _score_response_quality(t) == pytest.approx(1.0)

    def test_short_output_no_error(self):
        t = _trace(output_response="short", error=None)
        assert _score_response_quality(t) == pytest.approx(0.7)

    def test_long_output_with_error(self):
        t = _trace(output_response="x" * 60, error="boom")
        assert _score_response_quality(t) == pytest.approx(0.8)

    def test_short_output_with_error(self):
        t = _trace(output_response="x", error="fail")
        assert _score_response_quality(t) == pytest.approx(0.5)


class TestToolAccuracy:
    def test_no_tools(self):
        assert _score_tool_accuracy(_trace(tool_calls=[])) == 1.0

    def test_all_successful(self):
        calls = [{"name": "a", "success": True}, {"name": "b", "success": True}]
        assert _score_tool_accuracy(_trace(tool_calls=calls)) == 1.0

    def test_half_successful(self):
        calls = [{"name": "a", "success": True}, {"name": "b", "success": False}]
        assert _score_tool_accuracy(_trace(tool_calls=calls)) == pytest.approx(0.5)

    def test_all_failed(self):
        calls = [{"name": "a", "success": False}]
        assert _score_tool_accuracy(_trace(tool_calls=calls)) == 0.0


class TestStepEfficiency:
    def test_perfect_efficiency(self):
        # 2 tool calls => expected 3, 3 fsm states => 3/3 = 1.0
        t = _trace(tool_calls=[{"name": "a"}, {"name": "b"}], fsm_states=["s1", "s2", "s3"])
        assert _score_step_efficiency(t) == pytest.approx(1.0)

    def test_extra_states(self):
        # 1 tool call => expected 2, 4 fsm states => 2/4 = 0.5
        t = _trace(tool_calls=[{"name": "a"}], fsm_states=["s1", "s2", "s3", "s4"])
        assert _score_step_efficiency(t) == pytest.approx(0.5)

    def test_no_states_uses_expected(self):
        t = _trace(tool_calls=[{"name": "a"}], fsm_states=[])
        # empty states => actual = expected => 1.0
        assert _score_step_efficiency(t) == pytest.approx(1.0)


class TestLatencyScore:
    def test_fast(self):
        assert _score_latency(_trace(duration_ms=500)) == 1.0

    def test_at_threshold(self):
        assert _score_latency(_trace(duration_ms=2000)) == 1.0

    def test_slow(self):
        assert _score_latency(_trace(duration_ms=30000)) == 0.0

    def test_mid(self):
        # 16000ms is midpoint of 2000-30000
        score = _score_latency(_trace(duration_ms=16000))
        assert 0.4 < score < 0.6


class TestCostEfficiency:
    def test_cheap(self):
        assert _score_cost_efficiency(_trace(cost_usd=0.001)) == 1.0

    def test_expensive(self):
        assert _score_cost_efficiency(_trace(cost_usd=2.0)) == 0.0

    def test_mid_cost(self):
        score = _score_cost_efficiency(_trace(cost_usd=0.50))
        assert 0.4 < score < 0.6


class TestSafetyScore:
    def test_clean(self):
        assert _score_safety(_trace()) == 1.0

    def test_high_risk_no_approval(self):
        t = _trace(risk_score=80, approval_required=False)
        assert _score_safety(t) == pytest.approx(0.7)

    def test_high_risk_with_approval(self):
        t = _trace(risk_score=80, approval_required=True)
        assert _score_safety(t) == 1.0

    def test_error_deduction(self):
        t = _trace(error="timeout")
        assert _score_safety(t) == pytest.approx(0.8)

    def test_both_penalties(self):
        t = _trace(risk_score=90, approval_required=False, error="fail")
        assert _score_safety(t) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ProductionEvaluator
# ---------------------------------------------------------------------------


class TestProductionEvaluator:
    def test_record_and_score(self):
        ev = ProductionEvaluator(TENANT)
        t = _trace()
        ev.record_trace(t)
        m = ev.score_trace(t)
        assert isinstance(m, QualityMetrics)
        assert 0.0 <= m.overall <= 1.0

    def test_dashboard_empty(self):
        ev = ProductionEvaluator(TENANT)
        d = ev.get_dashboard()
        assert d["trace_count"] == 0
        assert d["avg_metrics"] is None

    def test_dashboard_with_traces(self):
        ev = ProductionEvaluator(TENANT)
        for _ in range(5):
            ev.record_trace(_trace())
        d = ev.get_dashboard(window_hours=1)
        assert d["trace_count"] == 5
        assert d["avg_metrics"] is not None
        assert "overall" in d["avg_metrics"]

    def test_get_trends(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace())
        trends = ev.get_trends("overall", window_hours=1)
        assert len(trends) >= 1
        assert "avg" in trends[0]

    def test_detect_regression_not_enough_data(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace())
        assert ev.detect_regression("overall") is False

    def test_detect_regression_true(self):
        ev = ProductionEvaluator(TENANT)
        now = datetime.now(timezone.utc)
        # Older traces: good (no error, long output)
        for i in range(4):
            ev.record_trace(
                _trace(
                    started_at=now - timedelta(hours=10 - i),
                    output_response="x" * 100,
                    error=None,
                )
            )
        # Newer traces: bad (error, short output)
        for i in range(4):
            ev.record_trace(
                _trace(
                    started_at=now - timedelta(hours=2 - i * 0.1),
                    output_response="x",
                    error="fail",
                )
            )
        assert ev.detect_regression("response_quality", threshold=0.1) is True

    def test_detect_regression_false_stable(self):
        ev = ProductionEvaluator(TENANT)
        now = datetime.now(timezone.utc)
        for i in range(8):
            ev.record_trace(_trace(started_at=now - timedelta(hours=8 - i)))
        assert ev.detect_regression("overall") is False

    def test_get_slowest_traces(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace(duration_ms=100))
        ev.record_trace(_trace(duration_ms=9000))
        ev.record_trace(_trace(duration_ms=5000))
        result = ev.get_slowest_traces(limit=2)
        assert len(result) == 2
        assert result[0].duration_ms == 9000

    def test_get_failed_traces(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace(error=None))
        ev.record_trace(_trace(error="boom"))
        ev.record_trace(_trace(error="crash"))
        result = ev.get_failed_traces()
        assert len(result) == 2
        assert all(t.error is not None for t in result)

    def test_get_costliest_traces(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace(cost_usd=0.001))
        ev.record_trace(_trace(cost_usd=0.50))
        ev.record_trace(_trace(cost_usd=0.10))
        result = ev.get_costliest_traces(limit=1)
        assert result[0].cost_usd == 0.50

    def test_compare_models(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace(model="gpt-4o"))
        ev.record_trace(_trace(model="gpt-4o"))
        ev.record_trace(_trace(model="claude-sonnet"))
        cmp = ev.compare_models("gpt-4o", "claude-sonnet")
        assert cmp["traces_a"] == 2
        assert cmp["traces_b"] == 1
        assert cmp["metrics_a"] is not None
        assert cmp["metrics_b"] is not None

    def test_compare_models_missing(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace(model="gpt-4o"))
        cmp = ev.compare_models("gpt-4o", "nonexistent")
        assert cmp["metrics_b"] is None

    def test_export_report_empty(self):
        ev = ProductionEvaluator(TENANT)
        r = ev.export_report()
        assert r["trace_count"] == 0
        assert r["format"] == "json"

    def test_export_report_with_data(self):
        ev = ProductionEvaluator(TENANT)
        ev.record_trace(_trace())
        ev.record_trace(_trace(error="fail"))
        r = ev.export_report()
        assert r["trace_count"] == 2
        assert r["error_rate"] == 0.5
        assert len(r["top_errors"]) == 1

    def test_import_from_package(self):
        """Verify classes are exported from the eval package."""
        from aria_core.eval import ExecutionTrace, ProductionEvaluator, QualityMetrics

        assert ExecutionTrace is not None
        assert ProductionEvaluator is not None
        assert QualityMetrics is not None
