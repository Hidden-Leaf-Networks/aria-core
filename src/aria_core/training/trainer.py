"""Trace recording, analysis, and guardrail generation for agent training."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from aria_core.runtime.models import AgentContext, BaseModel

logger = logging.getLogger("aria_core.training")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TraceRecord(BaseModel):
    """Captures a full agent execution trace."""

    id: UUID = None  # type: ignore[assignment]
    agent_id: str
    tenant_id: str
    context_id: str
    states_visited: list[str] = []
    actions_executed: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []
    risk_scores: list[int] = []
    outcome: Literal["success", "failure", "timeout", "rejected"] = "success"
    total_duration_ms: float = 0.0
    created_at: datetime = None  # type: ignore[assignment]

    def __init__(self, **data: Any) -> None:
        if "id" not in data or data["id"] is None:
            data["id"] = uuid4()
        if "created_at" not in data or data["created_at"] is None:
            data["created_at"] = datetime.now(timezone.utc)
        super().__init__(**data)


class Guardrail(BaseModel):
    """A rule derived from trace analysis."""

    name: str
    description: str = ""
    rule_type: Literal[
        "input_validation",
        "output_constraint",
        "rate_limit",
        "skill_restriction",
        "risk_threshold",
    ]
    condition: dict[str, Any] = {}
    severity: Literal["warn", "block"] = "warn"
    enabled: bool = True


class GuardrailViolation(BaseModel):
    """A single guardrail violation."""

    guardrail_name: str
    severity: Literal["warn", "block"]
    message: str


# ---------------------------------------------------------------------------
# GuardrailEvaluator
# ---------------------------------------------------------------------------


class GuardrailEvaluator:
    """Evaluates an AgentContext against a set of guardrails."""

    def evaluate(
        self,
        context: AgentContext,
        guardrails: list[Guardrail],
    ) -> list[GuardrailViolation]:
        """Check context against all enabled guardrails and return violations."""
        violations: list[GuardrailViolation] = []
        for g in guardrails:
            if not g.enabled:
                continue
            violation = self._check(context, g)
            if violation is not None:
                violations.append(violation)
        return violations

    # -- internal dispatch ---------------------------------------------------

    def _check(self, ctx: AgentContext, g: Guardrail) -> GuardrailViolation | None:
        handler = getattr(self, f"_check_{g.rule_type}", None)
        if handler is None:
            return None
        return handler(ctx, g)

    def _check_risk_threshold(
        self, ctx: AgentContext, g: Guardrail
    ) -> GuardrailViolation | None:
        threshold = g.condition.get("max_risk_score")
        if threshold is None:
            return None
        current_risk = ctx.metadata.get("risk_score", 0)
        if current_risk > threshold:
            return GuardrailViolation(
                guardrail_name=g.name,
                severity=g.severity,
                message=f"Risk score {current_risk} exceeds threshold {threshold}",
            )
        return None

    def _check_skill_restriction(
        self, ctx: AgentContext, g: Guardrail
    ) -> GuardrailViolation | None:
        blocked: list[str] = g.condition.get("blocked_skills", [])
        allowed = ctx.config.allowed_tools or []
        for skill in allowed:
            if skill in blocked:
                return GuardrailViolation(
                    guardrail_name=g.name,
                    severity=g.severity,
                    message=f"Skill '{skill}' is restricted by guardrail",
                )
        # Also check skill_results already executed
        for skill in ctx.skill_results:
            if skill in blocked:
                return GuardrailViolation(
                    guardrail_name=g.name,
                    severity=g.severity,
                    message=f"Skill '{skill}' is restricted by guardrail",
                )
        return None

    def _check_rate_limit(
        self, ctx: AgentContext, g: Guardrail
    ) -> GuardrailViolation | None:
        max_steps = g.condition.get("max_steps")
        if max_steps is None:
            return None
        if ctx.step_count > max_steps:
            return GuardrailViolation(
                guardrail_name=g.name,
                severity=g.severity,
                message=f"Step count {ctx.step_count} exceeds limit {max_steps}",
            )
        return None

    def _check_input_validation(
        self, ctx: AgentContext, g: Guardrail
    ) -> GuardrailViolation | None:
        max_length = g.condition.get("max_message_length")
        if max_length is None:
            return None
        for msg in ctx.messages:
            if len(msg.content) > max_length:
                return GuardrailViolation(
                    guardrail_name=g.name,
                    severity=g.severity,
                    message=f"Message length {len(msg.content)} exceeds max {max_length}",
                )
        return None

    def _check_output_constraint(
        self, ctx: AgentContext, g: Guardrail
    ) -> GuardrailViolation | None:
        blocked_patterns: list[str] = g.condition.get("blocked_patterns", [])
        response = ctx.metadata.get("response", "")
        for pattern in blocked_patterns:
            if pattern in response:
                return GuardrailViolation(
                    guardrail_name=g.name,
                    severity=g.severity,
                    message=f"Response contains blocked pattern '{pattern}'",
                )
        return None


# ---------------------------------------------------------------------------
# TraceAnalyzer
# ---------------------------------------------------------------------------


class TraceAnalyzer:
    """Records traces and derives guardrails from production patterns."""

    def __init__(self) -> None:
        self._traces: list[TraceRecord] = []

    # -- recording -----------------------------------------------------------

    def record(self, trace: TraceRecord) -> None:
        """Store a trace."""
        self._traces.append(trace)

    @property
    def traces(self) -> list[TraceRecord]:
        return list(self._traces)

    # -- metrics -------------------------------------------------------------

    def get_success_rate(self, agent_id: str | None = None) -> float:
        """Return fraction of traces with outcome='success'."""
        subset = self._filter(agent_id)
        if not subset:
            return 0.0
        return sum(1 for t in subset if t.outcome == "success") / len(subset)

    def get_avg_duration(self, agent_id: str | None = None) -> float:
        """Return average total_duration_ms across traces."""
        subset = self._filter(agent_id)
        if not subset:
            return 0.0
        return sum(t.total_duration_ms for t in subset) / len(subset)

    def get_skill_failure_rates(self) -> dict[str, float]:
        """Return per-skill failure rates across all traces."""
        totals: dict[str, int] = defaultdict(int)
        failures: dict[str, int] = defaultdict(int)
        for trace in self._traces:
            for action in trace.actions_executed:
                name = action.get("skill_name", "unknown")
                totals[name] += 1
                if not action.get("success", True):
                    failures[name] += 1
        return {
            skill: failures[skill] / totals[skill]
            for skill in totals
            if totals[skill] > 0
        }

    # -- failure analysis ----------------------------------------------------

    def analyze_failure_patterns(self, min_traces: int = 5) -> list[dict[str, Any]]:
        """Find common patterns in failed traces.

        Returns a list of pattern dicts, each describing a recurring failure.
        """
        failed = [t for t in self._traces if t.outcome != "success"]
        if len(failed) < min_traces:
            return []

        patterns: list[dict[str, Any]] = []

        # Pattern 1: high-failure skills
        skill_rates = self.get_skill_failure_rates()
        for skill, rate in skill_rates.items():
            if rate > 0.5:
                patterns.append(
                    {
                        "type": "skill_failure",
                        "skill": skill,
                        "failure_rate": round(rate, 3),
                        "recommendation": f"Skill '{skill}' fails >50% — consider restricting.",
                    }
                )

        # Pattern 2: high average risk
        avg_risk = self._avg_risk_score()
        if avg_risk > 70:
            patterns.append(
                {
                    "type": "high_risk",
                    "avg_risk_score": round(avg_risk, 2),
                    "recommendation": "Average risk score exceeds 70 — add risk threshold guardrail.",
                }
            )

        # Pattern 3: timeout rate
        timeout_rate = self._outcome_rate("timeout")
        if timeout_rate > 0.2:
            patterns.append(
                {
                    "type": "timeout",
                    "timeout_rate": round(timeout_rate, 3),
                    "recommendation": "Timeout rate >20% — consider rate-limit guardrail.",
                }
            )

        # Pattern 4: slow execution
        avg_dur = self.get_avg_duration()
        if avg_dur > 30_000:
            patterns.append(
                {
                    "type": "slow_execution",
                    "avg_duration_ms": round(avg_dur, 2),
                    "recommendation": "Average execution >30s — add step limit guardrail.",
                }
            )

        return patterns

    # -- guardrail suggestion -----------------------------------------------

    def suggest_guardrails(self, min_traces: int = 5) -> list[Guardrail]:
        """Auto-generate guardrails from trace patterns."""
        if len(self._traces) < min_traces:
            return []

        guardrails: list[Guardrail] = []

        # Skill restriction for high-failure skills
        skill_rates = self.get_skill_failure_rates()
        for skill, rate in skill_rates.items():
            if rate > 0.5:
                guardrails.append(
                    Guardrail(
                        name=f"restrict_{skill}",
                        description=f"Auto-generated: skill '{skill}' fails {rate:.0%} of the time",
                        rule_type="skill_restriction",
                        condition={"blocked_skills": [skill]},
                        severity="block",
                        enabled=True,
                    )
                )

        # Risk threshold
        avg_risk = self._avg_risk_score()
        if avg_risk > 70:
            guardrails.append(
                Guardrail(
                    name="high_risk_threshold",
                    description=f"Auto-generated: avg risk {avg_risk:.1f} exceeds 70",
                    rule_type="risk_threshold",
                    condition={"max_risk_score": 70},
                    severity="block",
                    enabled=True,
                )
            )

        # Timeout → rate limit
        timeout_rate = self._outcome_rate("timeout")
        if timeout_rate > 0.2:
            guardrails.append(
                Guardrail(
                    name="timeout_rate_limit",
                    description=f"Auto-generated: timeout rate {timeout_rate:.0%}",
                    rule_type="rate_limit",
                    condition={"max_steps": 5},
                    severity="warn",
                    enabled=True,
                )
            )

        # Slow execution → step limit
        avg_dur = self.get_avg_duration()
        if avg_dur > 30_000:
            guardrails.append(
                Guardrail(
                    name="slow_execution_limit",
                    description=f"Auto-generated: avg duration {avg_dur:.0f}ms exceeds 30s",
                    rule_type="rate_limit",
                    condition={"max_steps": 3},
                    severity="warn",
                    enabled=True,
                )
            )

        return guardrails

    # -- event callback integration -----------------------------------------

    def as_event_callback(self):
        """Return an async callback suitable for FSM event_callback.

        Captures transition and agent lifecycle events into traces.
        """
        pending: dict[str, dict[str, Any]] = {}

        async def callback(event_type: str, payload: dict[str, Any]) -> None:
            ctx_id = payload.get("context_id", "unknown")

            if event_type == "agent.start":
                pending[ctx_id] = {
                    "states": [],
                    "start_ms": _now_ms(),
                    "agent_id": payload.get("agent_id", "unknown"),
                    "tenant_id": payload.get("tenant_id", "unknown"),
                }

            elif event_type == "transition.complete":
                if ctx_id in pending:
                    pending[ctx_id]["states"].append(payload.get("to", ""))
                # Also track when context_id is not set — use first pending
                elif pending:
                    key = next(iter(pending))
                    pending[key]["states"].append(payload.get("to", ""))

            elif event_type == "agent.complete" or event_type == "agent.error":
                data = pending.pop(ctx_id, None)
                if data is None and pending:
                    key = next(iter(pending))
                    data = pending.pop(key)
                if data is not None:
                    outcome: str = "success" if event_type == "agent.complete" else "failure"
                    duration = _now_ms() - data.get("start_ms", _now_ms())
                    self.record(
                        TraceRecord(
                            agent_id=data.get("agent_id", "unknown"),
                            tenant_id=data.get("tenant_id", "unknown"),
                            context_id=ctx_id,
                            states_visited=data.get("states", []),
                            outcome=outcome,
                            total_duration_ms=max(duration, 0),
                        )
                    )

        return callback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return datetime.now(timezone.utc).timestamp() * 1000


# -- private helpers on TraceAnalyzer (kept out of public API) ----

def _filter_traces(
    traces: list[TraceRecord], agent_id: str | None
) -> list[TraceRecord]:
    if agent_id is None:
        return traces
    return [t for t in traces if t.agent_id == agent_id]


# Attach as private method
TraceAnalyzer._filter = lambda self, agent_id=None: _filter_traces(self._traces, agent_id)  # type: ignore[attr-defined]


def _avg_risk(analyzer: TraceAnalyzer) -> float:
    all_scores: list[int] = []
    for t in analyzer._traces:
        all_scores.extend(t.risk_scores)
    if not all_scores:
        return 0.0
    return sum(all_scores) / len(all_scores)


def _outcome_rate(analyzer: TraceAnalyzer, outcome: str) -> float:
    if not analyzer._traces:
        return 0.0
    return sum(1 for t in analyzer._traces if t.outcome == outcome) / len(analyzer._traces)


TraceAnalyzer._avg_risk_score = lambda self: _avg_risk(self)  # type: ignore[attr-defined]
TraceAnalyzer._outcome_rate = lambda self, o: _outcome_rate(self, o)  # type: ignore[attr-defined]
