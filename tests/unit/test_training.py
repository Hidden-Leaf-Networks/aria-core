"""Tests for agent training, trace analysis, and guardrails."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.runtime.models import AgentConfig, AgentContext, ChatMessage, MessageRole
from aria_core.training import (
    Guardrail,
    GuardrailEvaluator,
    GuardrailViolation,
    TraceAnalyzer,
    TraceRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    agent_id: str = "agent-1",
    outcome: str = "success",
    duration_ms: float = 1000.0,
    risk_scores: list[int] | None = None,
    actions: list[dict] | None = None,
) -> TraceRecord:
    return TraceRecord(
        agent_id=agent_id,
        tenant_id="tenant-1",
        context_id=str(uuid4()),
        states_visited=["idle", "routing", "executing_step", "complete"],
        actions_executed=actions or [],
        risk_scores=risk_scores or [],
        outcome=outcome,
        total_duration_ms=duration_ms,
    )


def _make_context(**overrides) -> AgentContext:
    defaults = {
        "messages": [ChatMessage(role=MessageRole.USER, content="hello")],
    }
    defaults.update(overrides)
    return AgentContext(**defaults)


# ---------------------------------------------------------------------------
# TraceRecord model tests
# ---------------------------------------------------------------------------


class TestTraceRecord:
    def test_create_trace_defaults(self):
        t = TraceRecord(agent_id="a", tenant_id="t", context_id="c")
        assert t.agent_id == "a"
        assert t.outcome == "success"
        assert t.id is not None
        assert t.created_at is not None
        assert t.total_duration_ms == 0.0

    def test_trace_with_actions_and_llm_calls(self):
        t = TraceRecord(
            agent_id="a",
            tenant_id="t",
            context_id="c",
            actions_executed=[
                {"skill_name": "web_search", "success": True, "duration_ms": 200},
            ],
            llm_calls=[
                {"model": "gpt-4", "prompt_tokens": 100, "completion_tokens": 50, "duration_ms": 500},
            ],
            risk_scores=[30, 45],
            outcome="failure",
            total_duration_ms=5000.0,
        )
        assert t.outcome == "failure"
        assert len(t.actions_executed) == 1
        assert len(t.llm_calls) == 1
        assert t.risk_scores == [30, 45]


# ---------------------------------------------------------------------------
# Guardrail model tests
# ---------------------------------------------------------------------------


class TestGuardrail:
    def test_create_guardrail(self):
        g = Guardrail(
            name="test_rule",
            description="A test guardrail",
            rule_type="risk_threshold",
            condition={"max_risk_score": 80},
            severity="block",
        )
        assert g.name == "test_rule"
        assert g.rule_type == "risk_threshold"
        assert g.enabled is True

    def test_guardrail_defaults(self):
        g = Guardrail(name="min", rule_type="input_validation")
        assert g.severity == "warn"
        assert g.enabled is True
        assert g.condition == {}


# ---------------------------------------------------------------------------
# GuardrailEvaluator tests
# ---------------------------------------------------------------------------


class TestGuardrailEvaluator:
    def test_no_violations_empty_guardrails(self):
        ev = GuardrailEvaluator()
        ctx = _make_context()
        assert ev.evaluate(ctx, []) == []

    def test_disabled_guardrail_skipped(self):
        ev = GuardrailEvaluator()
        ctx = _make_context(metadata={"risk_score": 99})
        g = Guardrail(
            name="disabled",
            rule_type="risk_threshold",
            condition={"max_risk_score": 50},
            severity="block",
            enabled=False,
        )
        assert ev.evaluate(ctx, [g]) == []

    def test_risk_threshold_violation(self):
        ev = GuardrailEvaluator()
        ctx = _make_context(metadata={"risk_score": 85})
        g = Guardrail(
            name="risk_check",
            rule_type="risk_threshold",
            condition={"max_risk_score": 70},
            severity="block",
        )
        violations = ev.evaluate(ctx, [g])
        assert len(violations) == 1
        assert violations[0].guardrail_name == "risk_check"
        assert violations[0].severity == "block"
        assert "85" in violations[0].message

    def test_risk_threshold_no_violation(self):
        ev = GuardrailEvaluator()
        ctx = _make_context(metadata={"risk_score": 50})
        g = Guardrail(
            name="risk_check",
            rule_type="risk_threshold",
            condition={"max_risk_score": 70},
        )
        assert ev.evaluate(ctx, [g]) == []

    def test_skill_restriction_violation(self):
        ev = GuardrailEvaluator()
        ctx = _make_context(
            skill_results={"dangerous_tool": {"result": "ok"}},
        )
        g = Guardrail(
            name="block_dangerous",
            rule_type="skill_restriction",
            condition={"blocked_skills": ["dangerous_tool"]},
            severity="block",
        )
        violations = ev.evaluate(ctx, [g])
        assert len(violations) == 1
        assert "dangerous_tool" in violations[0].message

    def test_rate_limit_violation(self):
        ev = GuardrailEvaluator()
        ctx = _make_context(step_count=15)
        g = Guardrail(
            name="step_limit",
            rule_type="rate_limit",
            condition={"max_steps": 10},
            severity="warn",
        )
        violations = ev.evaluate(ctx, [g])
        assert len(violations) == 1
        assert violations[0].severity == "warn"

    def test_input_validation_violation(self):
        ev = GuardrailEvaluator()
        long_msg = "x" * 5000
        ctx = _make_context(
            messages=[ChatMessage(role=MessageRole.USER, content=long_msg)],
        )
        g = Guardrail(
            name="max_input",
            rule_type="input_validation",
            condition={"max_message_length": 1000},
            severity="block",
        )
        violations = ev.evaluate(ctx, [g])
        assert len(violations) == 1

    def test_output_constraint_violation(self):
        ev = GuardrailEvaluator()
        ctx = _make_context(metadata={"response": "Here is your SECRET_KEY=abc123"})
        g = Guardrail(
            name="no_secrets",
            rule_type="output_constraint",
            condition={"blocked_patterns": ["SECRET_KEY"]},
            severity="block",
        )
        violations = ev.evaluate(ctx, [g])
        assert len(violations) == 1
        assert "SECRET_KEY" in violations[0].message


# ---------------------------------------------------------------------------
# TraceAnalyzer tests
# ---------------------------------------------------------------------------


class TestTraceAnalyzer:
    def test_record_and_count(self):
        analyzer = TraceAnalyzer()
        analyzer.record(_make_trace())
        analyzer.record(_make_trace())
        assert len(analyzer.traces) == 2

    def test_success_rate_all_success(self):
        analyzer = TraceAnalyzer()
        for _ in range(5):
            analyzer.record(_make_trace(outcome="success"))
        assert analyzer.get_success_rate() == 1.0

    def test_success_rate_mixed(self):
        analyzer = TraceAnalyzer()
        for _ in range(3):
            analyzer.record(_make_trace(outcome="success"))
        for _ in range(2):
            analyzer.record(_make_trace(outcome="failure"))
        assert analyzer.get_success_rate() == pytest.approx(0.6)

    def test_success_rate_filtered_by_agent(self):
        analyzer = TraceAnalyzer()
        analyzer.record(_make_trace(agent_id="a1", outcome="success"))
        analyzer.record(_make_trace(agent_id="a1", outcome="failure"))
        analyzer.record(_make_trace(agent_id="a2", outcome="success"))
        assert analyzer.get_success_rate("a1") == pytest.approx(0.5)
        assert analyzer.get_success_rate("a2") == 1.0

    def test_success_rate_empty(self):
        analyzer = TraceAnalyzer()
        assert analyzer.get_success_rate() == 0.0

    def test_avg_duration(self):
        analyzer = TraceAnalyzer()
        analyzer.record(_make_trace(duration_ms=1000))
        analyzer.record(_make_trace(duration_ms=3000))
        assert analyzer.get_avg_duration() == pytest.approx(2000.0)

    def test_avg_duration_empty(self):
        analyzer = TraceAnalyzer()
        assert analyzer.get_avg_duration() == 0.0

    def test_skill_failure_rates(self):
        analyzer = TraceAnalyzer()
        for _ in range(6):
            analyzer.record(
                _make_trace(
                    actions=[
                        {"skill_name": "good_skill", "success": True},
                        {"skill_name": "bad_skill", "success": False},
                    ],
                )
            )
        rates = analyzer.get_skill_failure_rates()
        assert rates["good_skill"] == 0.0
        assert rates["bad_skill"] == 1.0

    def test_analyze_failure_patterns_insufficient_traces(self):
        analyzer = TraceAnalyzer()
        analyzer.record(_make_trace(outcome="failure"))
        assert analyzer.analyze_failure_patterns(min_traces=5) == []

    def test_analyze_failure_patterns_high_failure_skill(self):
        analyzer = TraceAnalyzer()
        for _ in range(6):
            analyzer.record(
                _make_trace(
                    outcome="failure",
                    actions=[{"skill_name": "flaky_api", "success": False}],
                )
            )
        patterns = analyzer.analyze_failure_patterns(min_traces=5)
        skill_patterns = [p for p in patterns if p["type"] == "skill_failure"]
        assert len(skill_patterns) >= 1
        assert skill_patterns[0]["skill"] == "flaky_api"

    def test_suggest_guardrails_insufficient_traces(self):
        analyzer = TraceAnalyzer()
        analyzer.record(_make_trace())
        assert analyzer.suggest_guardrails(min_traces=5) == []

    def test_suggest_guardrails_skill_restriction(self):
        analyzer = TraceAnalyzer()
        for _ in range(6):
            analyzer.record(
                _make_trace(
                    actions=[{"skill_name": "broken_tool", "success": False}],
                )
            )
        guardrails = analyzer.suggest_guardrails(min_traces=5)
        skill_guardrails = [g for g in guardrails if g.rule_type == "skill_restriction"]
        assert len(skill_guardrails) >= 1
        assert "broken_tool" in skill_guardrails[0].condition["blocked_skills"]
        assert skill_guardrails[0].severity == "block"

    def test_suggest_guardrails_risk_threshold(self):
        analyzer = TraceAnalyzer()
        for _ in range(6):
            analyzer.record(_make_trace(risk_scores=[80, 90, 85]))
        guardrails = analyzer.suggest_guardrails(min_traces=5)
        risk_guardrails = [g for g in guardrails if g.rule_type == "risk_threshold"]
        assert len(risk_guardrails) == 1
        assert risk_guardrails[0].condition["max_risk_score"] == 70

    def test_suggest_guardrails_timeout_rate_limit(self):
        analyzer = TraceAnalyzer()
        # 4 timeouts out of 6 = 66% > 20%
        for _ in range(4):
            analyzer.record(_make_trace(outcome="timeout"))
        for _ in range(2):
            analyzer.record(_make_trace(outcome="success"))
        guardrails = analyzer.suggest_guardrails(min_traces=5)
        rl_guardrails = [g for g in guardrails if g.name == "timeout_rate_limit"]
        assert len(rl_guardrails) == 1

    def test_suggest_guardrails_slow_execution(self):
        analyzer = TraceAnalyzer()
        for _ in range(6):
            analyzer.record(_make_trace(duration_ms=40_000))
        guardrails = analyzer.suggest_guardrails(min_traces=5)
        slow_guardrails = [g for g in guardrails if g.name == "slow_execution_limit"]
        assert len(slow_guardrails) == 1

    async def test_event_callback_captures_traces(self):
        analyzer = TraceAnalyzer()
        cb = analyzer.as_event_callback()

        ctx_id = str(uuid4())
        await cb("agent.start", {"context_id": ctx_id, "agent_id": "a1", "tenant_id": "t1"})
        await cb("transition.complete", {"context_id": ctx_id, "to": "routing"})
        await cb("transition.complete", {"context_id": ctx_id, "to": "complete"})
        await cb("agent.complete", {"context_id": ctx_id})

        assert len(analyzer.traces) == 1
        trace = analyzer.traces[0]
        assert trace.agent_id == "a1"
        assert trace.outcome == "success"
        assert "routing" in trace.states_visited

    async def test_event_callback_error_outcome(self):
        analyzer = TraceAnalyzer()
        cb = analyzer.as_event_callback()

        ctx_id = str(uuid4())
        await cb("agent.start", {"context_id": ctx_id})
        await cb("agent.error", {"context_id": ctx_id})

        assert len(analyzer.traces) == 1
        assert analyzer.traces[0].outcome == "failure"
