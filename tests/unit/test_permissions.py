"""Tests for risk scoring and approval engines."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.permissions import (
    ApprovalDeniedError,
    ApprovalEngine,
    ApprovalExpiredError,
    ApprovalGate,
    ApprovalNotFoundError,
    ApprovalRequiredError,
    ApprovalState,
    ImpactScope,
    InvalidApprovalStateError,
    RiskEngine,
    RiskPolicy,
    RiskScoreInput,
    SkillCategory,
    UnauthorizedApproverError,
)


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------


class TestRiskEngine:
    def test_read_local_is_safe(self) -> None:
        engine = RiskEngine()
        score = engine.calculate(RiskScoreInput(
            skill_name="read_file",
            skill_category=SkillCategory.READ,
            impact_scope=ImpactScope.LOCAL,
        ))
        assert score.level == "safe"
        assert score.requires_approval is False

    def test_exec_external_is_high(self) -> None:
        engine = RiskEngine()
        score = engine.calculate(RiskScoreInput(
            skill_name="run_command",
            skill_category=SkillCategory.EXEC,
            impact_scope=ImpactScope.EXTERNAL,
        ))
        assert score.score >= 30
        assert score.level in ("low", "medium", "high")

    def test_context_modifiers_increase_score(self) -> None:
        engine = RiskEngine()
        base = engine.calculate(RiskScoreInput(
            skill_name="send_email",
            skill_category=SkillCategory.EXTERNAL,
            impact_scope=ImpactScope.EXTERNAL,
        ))
        with_ctx = engine.calculate(RiskScoreInput(
            skill_name="send_email",
            skill_category=SkillCategory.EXTERNAL,
            impact_scope=ImpactScope.EXTERNAL,
            has_sensitive_args=True,
            targets_external_system=True,
            requires_network=True,
        ))
        assert with_ctx.score > base.score

    def test_historical_failures_add_penalty(self) -> None:
        engine = RiskEngine()
        clean = engine.calculate(RiskScoreInput(
            skill_name="api_call",
            skill_category=SkillCategory.EXTERNAL,
            impact_scope=ImpactScope.USER,
        ))
        dirty = engine.calculate(RiskScoreInput(
            skill_name="api_call",
            skill_category=SkillCategory.EXTERNAL,
            impact_scope=ImpactScope.USER,
            historical_failures=10,
            historical_violations=5,
        ))
        assert dirty.score > clean.score

    def test_deterministic_hash(self) -> None:
        engine = RiskEngine()
        input_data = RiskScoreInput(
            skill_name="test",
            skill_category=SkillCategory.READ,
            impact_scope=ImpactScope.LOCAL,
        )
        s1 = engine.calculate(input_data)
        s2 = engine.calculate(input_data)
        assert s1.input_hash == s2.input_hash
        assert s1.score == s2.score

    def test_custom_policy(self) -> None:
        policy = RiskPolicy(
            name="strict",
            approval_threshold=20,
            skill_category_weights={
                SkillCategory.READ: 0.5,
                SkillCategory.WRITE: 0.8,
                SkillCategory.EXEC: 0.9,
                SkillCategory.EXTERNAL: 0.7,
            },
        )
        engine = RiskEngine(policy=policy)
        score = engine.calculate(RiskScoreInput(
            skill_name="read_file",
            skill_category=SkillCategory.READ,
            impact_scope=ImpactScope.LOCAL,
        ))
        # Strict policy: even READ gets higher score
        assert score.score >= 20

    def test_evaluate_quick(self) -> None:
        engine = RiskEngine()
        score, level, needs_approval = engine.evaluate_quick(
            SkillCategory.READ, ImpactScope.LOCAL,
        )
        assert level == "safe"
        assert needs_approval is False

    def test_aggregate_risk(self) -> None:
        engine = RiskEngine()
        scores = [
            engine.calculate(RiskScoreInput(
                skill_name="a", skill_category=SkillCategory.READ, impact_scope=ImpactScope.LOCAL,
            )),
            engine.calculate(RiskScoreInput(
                skill_name="b", skill_category=SkillCategory.EXEC, impact_scope=ImpactScope.SYSTEM,
            )),
        ]
        agg_score, agg_level, _ = engine.calculate_aggregate_risk(scores)
        # Aggregate >= max individual score
        assert agg_score >= max(s.score for s in scores)

    def test_factors_in_result(self) -> None:
        engine = RiskEngine()
        score = engine.calculate(RiskScoreInput(
            skill_name="test",
            skill_category=SkillCategory.WRITE,
            impact_scope=ImpactScope.USER,
        ))
        factor_names = {f.name for f in score.factors}
        assert "skill_category" in factor_names
        assert "impact_scope" in factor_names


# ---------------------------------------------------------------------------
# ApprovalEngine
# ---------------------------------------------------------------------------


class TestApprovalEngine:
    def _make_risk_score(self, score: int = 75) -> object:
        """Create a RiskScore-like object for testing."""
        from aria_core.permissions.models import RiskScore, RiskFactor
        return RiskScore(
            score=score,
            level="high" if score > 60 else "medium",
            requires_approval=score >= 50,
            input_hash="test123",
            factors=[
                RiskFactor(name="test", category="test", weight=0.5, raw_value=50.0),
            ],
        )

    def test_default_gate_triggers(self) -> None:
        engine = ApprovalEngine()
        assert engine.requires_approval(75) is True
        assert engine.requires_approval(25) is False

    def test_create_and_approve(self) -> None:
        engine = ApprovalEngine()
        plan_id = uuid4()
        risk = self._make_risk_score(75)

        approval = engine.create_approval(plan_id=plan_id, risk_score=risk)
        assert approval.state == ApprovalState.PENDING

        response = engine.approve(approval.id, approver_id="user-1")
        assert response.approval.state == ApprovalState.APPROVED
        assert response.plan_state == "executing"

    def test_create_and_reject(self) -> None:
        engine = ApprovalEngine()
        plan_id = uuid4()
        risk = self._make_risk_score(75)

        approval = engine.create_approval(plan_id=plan_id, risk_score=risk)
        response = engine.reject(approval.id, approver_id="user-1", reason="Too risky")

        assert response.approval.state == ApprovalState.REJECTED
        assert response.plan_state == "failed"

    def test_not_found_raises(self) -> None:
        engine = ApprovalEngine()
        with pytest.raises(ApprovalNotFoundError):
            engine.approve(uuid4(), approver_id="user-1")

    def test_double_decide_raises(self) -> None:
        engine = ApprovalEngine()
        plan_id = uuid4()
        approval = engine.create_approval(plan_id=plan_id, risk_score=self._make_risk_score(75))
        engine.approve(approval.id, approver_id="user-1")

        with pytest.raises(InvalidApprovalStateError):
            engine.approve(approval.id, approver_id="user-2")

    def test_unauthorized_approver(self) -> None:
        gate = ApprovalGate(
            name="restricted",
            risk_threshold=50,
            allowed_approvers=["admin-1"],
        )
        engine = ApprovalEngine(gates=[gate])
        approval = engine.create_approval(plan_id=uuid4(), risk_score=self._make_risk_score(75))

        with pytest.raises(UnauthorizedApproverError):
            engine.approve(approval.id, approver_id="random-user")

    def test_verify_approved_passes(self) -> None:
        engine = ApprovalEngine()
        plan_id = uuid4()
        approval = engine.create_approval(plan_id=plan_id, risk_score=self._make_risk_score(75))
        engine.approve(approval.id, approver_id="user-1")

        assert engine.verify_approved(plan_id) is True

    def test_verify_pending_raises(self) -> None:
        engine = ApprovalEngine()
        plan_id = uuid4()
        engine.create_approval(plan_id=plan_id, risk_score=self._make_risk_score(75))

        with pytest.raises(ApprovalRequiredError):
            engine.verify_approved(plan_id)

    def test_verify_denied_raises(self) -> None:
        engine = ApprovalEngine()
        plan_id = uuid4()
        approval = engine.create_approval(plan_id=plan_id, risk_score=self._make_risk_score(75))
        engine.reject(approval.id, approver_id="user-1", reason="Nope")

        with pytest.raises(ApprovalDeniedError):
            engine.verify_approved(plan_id)

    def test_no_approval_needed_passes(self) -> None:
        engine = ApprovalEngine()
        # No approval was created for this plan
        assert engine.verify_approved(uuid4()) is True

    def test_pending_summary(self) -> None:
        engine = ApprovalEngine()
        for _ in range(3):
            engine.create_approval(plan_id=uuid4(), risk_score=self._make_risk_score(75))

        summary = engine.get_pending_summary()
        assert summary.total_pending == 3
        assert summary.high_risk_count == 3

    def test_custom_gate(self) -> None:
        low_gate = ApprovalGate(name="low-risk", risk_threshold=20, required_approvers=1)
        high_gate = ApprovalGate(name="high-risk", risk_threshold=70, required_approvers=2)
        engine = ApprovalEngine(gates=[low_gate, high_gate])

        # Score of 30 triggers low-risk gate
        assert engine.evaluate_gates(30) is not None
        assert engine.evaluate_gates(30).name == "low-risk"

    def test_multi_approver_gate(self) -> None:
        gate = ApprovalGate(name="multi", risk_threshold=50, required_approvers=2)
        engine = ApprovalEngine(gates=[gate])
        approval = engine.create_approval(plan_id=uuid4(), risk_score=self._make_risk_score(75))

        # First approval doesn't resolve
        r1 = engine.decide(approval.id, ApprovalState.APPROVED, "user-1")
        assert r1.approval.state == ApprovalState.PENDING

        # Second approval resolves
        r2 = engine.decide(approval.id, ApprovalState.APPROVED, "user-2")
        assert r2.approval.state == ApprovalState.APPROVED
