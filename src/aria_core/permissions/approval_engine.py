"""Approval engine for permission-first safety.

Engine-level enforcement of approval workflows. Approval gates cannot
be bypassed programmatically — all approval decisions flow through here.

Key guarantees:
- Decisions are immutable once recorded
- Approval gates are evaluated deterministically
- All decisions are auditable
- Engine-level enforcement (not UI/API level)

Usage:
    from aria_core.permissions import ApprovalEngine

    engine = ApprovalEngine()
    if engine.requires_approval(risk_score=75):
        approval = engine.create_approval(plan_id=uuid, risk_score=score)
        response = engine.approve(approval.id, approver_id="user-1")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from aria_core.permissions.models import (
    Approval,
    ApprovalDecision,
    ApprovalGate,
    ApprovalResponse,
    ApprovalState,
    PendingApprovalSummary,
    RiskScore,
)

logger = logging.getLogger("aria_core.permissions")


class ApprovalError(Exception):
    """Base exception for approval errors."""
    pass


class ApprovalRequiredError(ApprovalError):
    def __init__(self, approval_id: UUID, gate_name: str, risk_score: int):
        self.approval_id = approval_id
        self.gate_name = gate_name
        self.risk_score = risk_score
        super().__init__(f"Approval required by gate '{gate_name}' (risk score: {risk_score})")


class ApprovalDeniedError(ApprovalError):
    def __init__(self, approval_id: UUID, reason: str | None = None):
        self.approval_id = approval_id
        self.reason = reason
        super().__init__(f"Approval {approval_id} was denied: {reason or 'No reason provided'}")


class ApprovalExpiredError(ApprovalError):
    def __init__(self, approval_id: UUID, expired_at: datetime):
        self.approval_id = approval_id
        self.expired_at = expired_at
        super().__init__(f"Approval {approval_id} expired at {expired_at}")


class ApprovalNotFoundError(ApprovalError):
    def __init__(self, approval_id: UUID):
        self.approval_id = approval_id
        super().__init__(f"Approval {approval_id} not found")


class InvalidApprovalStateError(ApprovalError):
    def __init__(self, approval_id: UUID, current_state: str, operation: str):
        self.approval_id = approval_id
        super().__init__(f"Cannot {operation} approval {approval_id} in state '{current_state}'")


class UnauthorizedApproverError(ApprovalError):
    def __init__(self, approver_id: str, gate_name: str):
        self.approver_id = approver_id
        super().__init__(f"Approver '{approver_id}' is not authorized for gate '{gate_name}'")


class ApprovalEngine:
    """Engine-level approval workflow enforcement.

    Evaluates gates based on risk scores, creates approval requests,
    records immutable decisions, and handles expiration/escalation.
    """

    def __init__(
        self,
        gates: list[ApprovalGate] | None = None,
        default_timeout_minutes: int = 60,
    ) -> None:
        self.gates = gates or [self._create_default_gate()]
        self.default_timeout_minutes = default_timeout_minutes
        self._approvals: dict[UUID, Approval] = {}
        self._gates_by_id: dict[UUID, ApprovalGate] = {g.id: g for g in self.gates}
        self._gates_by_name: dict[str, ApprovalGate] = {g.name: g for g in self.gates}

    def _create_default_gate(self) -> ApprovalGate:
        return ApprovalGate(
            name="default-high-risk",
            description="Default gate for high-risk actions",
            risk_threshold=50,
            required_approvers=1,
            allowed_approvers=[],
            timeout_minutes=60,
        )

    def evaluate_gates(self, risk_score: int) -> ApprovalGate | None:
        """Return the first active gate triggered by the risk score."""
        sorted_gates = sorted(
            [g for g in self.gates if g.is_active],
            key=lambda g: g.risk_threshold,
        )
        for gate in sorted_gates:
            if risk_score >= gate.risk_threshold:
                return gate
        return None

    def requires_approval(self, risk_score: int) -> bool:
        return self.evaluate_gates(risk_score) is not None

    def create_approval(
        self,
        plan_id: UUID,
        risk_score: RiskScore,
        action_id: UUID | None = None,
        context: dict[str, Any] | None = None,
    ) -> Approval:
        """Create an approval request for a plan or action."""
        gate = self.evaluate_gates(risk_score.score)
        if not gate:
            raise ApprovalError(f"No gate triggered for risk score {risk_score.score}")

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=gate.timeout_minutes)

        approval = Approval(
            plan_id=plan_id,
            action_id=action_id,
            gate_id=gate.id,
            gate_name=gate.name,
            risk_score=risk_score.score,
            risk_factors=[f.model_dump() for f in risk_score.factors],
            context=context or {},
            state=ApprovalState.PENDING,
            decisions=[],
            required_approvals=gate.required_approvers,
            created_at=now,
            expires_at=expires_at,
        )

        self._approvals[approval.id] = approval
        return approval

    def get_approval(self, approval_id: UUID) -> Approval | None:
        return self._approvals.get(approval_id)

    def get_approval_for_plan(self, plan_id: UUID) -> Approval | None:
        for approval in self._approvals.values():
            if (
                approval.plan_id == plan_id
                and approval.action_id is None
                and approval.state == ApprovalState.PENDING
            ):
                return approval
        return None

    def decide(
        self,
        approval_id: UUID,
        decision: ApprovalState,
        approver_id: str,
        approver_type: str = "user",
        reason: str | None = None,
    ) -> ApprovalResponse:
        """Record an immutable approval decision."""
        approval = self.get_approval(approval_id)
        if not approval:
            raise ApprovalNotFoundError(approval_id)

        if approval.state != ApprovalState.PENDING:
            raise InvalidApprovalStateError(approval_id, approval.state.value, "decide on")

        if approval.is_expired:
            self._mark_expired(approval)
            raise ApprovalExpiredError(approval_id, approval.expires_at)

        gate = self._gates_by_id.get(approval.gate_id)
        if gate and gate.allowed_approvers:
            if approver_id not in gate.allowed_approvers:
                raise UnauthorizedApproverError(approver_id, gate.name)

        if decision not in {ApprovalState.APPROVED, ApprovalState.REJECTED}:
            raise ApprovalError(f"Invalid decision: {decision}")

        decision_record = ApprovalDecision(
            approval_id=approval_id,
            decision=decision,
            approver_id=approver_id,
            approver_type=approver_type,
            reason=reason,
            decided_at=datetime.now(timezone.utc),
        )

        updated_decisions = list(approval.decisions) + [decision_record]

        if decision == ApprovalState.REJECTED:
            new_state = ApprovalState.REJECTED
            resolved_at = datetime.now(timezone.utc)
        elif len([d for d in updated_decisions if d.decision == ApprovalState.APPROVED]) >= approval.required_approvals:
            new_state = ApprovalState.APPROVED
            resolved_at = datetime.now(timezone.utc)
        else:
            new_state = ApprovalState.PENDING
            resolved_at = None

        updated = Approval(
            id=approval.id,
            plan_id=approval.plan_id,
            action_id=approval.action_id,
            gate_id=approval.gate_id,
            gate_name=approval.gate_name,
            risk_score=approval.risk_score,
            risk_factors=approval.risk_factors,
            context=approval.context,
            state=new_state,
            decisions=updated_decisions,
            required_approvals=approval.required_approvals,
            created_at=approval.created_at,
            expires_at=approval.expires_at,
            resolved_at=resolved_at,
        )
        self._approvals[approval_id] = updated

        if new_state == ApprovalState.APPROVED:
            plan_state = "executing"
        elif new_state == ApprovalState.REJECTED:
            plan_state = "failed"
        else:
            plan_state = "blocked"

        return ApprovalResponse(approval=updated, decision=decision_record, plan_state=plan_state)

    def approve(self, approval_id: UUID, approver_id: str, reason: str | None = None) -> ApprovalResponse:
        return self.decide(approval_id, ApprovalState.APPROVED, approver_id, reason=reason)

    def reject(self, approval_id: UUID, approver_id: str, reason: str | None = None) -> ApprovalResponse:
        return self.decide(approval_id, ApprovalState.REJECTED, approver_id, reason=reason)

    def _mark_expired(self, approval: Approval) -> None:
        expired = Approval(
            id=approval.id,
            plan_id=approval.plan_id,
            action_id=approval.action_id,
            gate_id=approval.gate_id,
            gate_name=approval.gate_name,
            risk_score=approval.risk_score,
            risk_factors=approval.risk_factors,
            context=approval.context,
            state=ApprovalState.EXPIRED,
            decisions=approval.decisions,
            required_approvals=approval.required_approvals,
            created_at=approval.created_at,
            expires_at=approval.expires_at,
            resolved_at=datetime.now(timezone.utc),
        )
        self._approvals[approval.id] = expired

    def check_and_expire(self) -> list[Approval]:
        """Check for and mark expired approvals."""
        expired = []
        now = datetime.now(timezone.utc)
        for approval in list(self._approvals.values()):
            if approval.state == ApprovalState.PENDING and now > approval.expires_at:
                self._mark_expired(approval)
                expired.append(self._approvals[approval.id])
        return expired

    def _get_latest_approval_for_plan(self, plan_id: UUID) -> Approval | None:
        """Get the most recent approval for a plan (any state)."""
        matches = [
            a for a in self._approvals.values()
            if a.plan_id == plan_id and a.action_id is None
        ]
        if not matches:
            return None
        return max(matches, key=lambda a: a.created_at)

    def verify_approved(self, plan_id: UUID, action_id: UUID | None = None) -> bool:
        """Verify a plan/action has been approved. Raises if not."""
        approval = self._get_latest_approval_for_plan(plan_id)
        if not approval:
            return True  # No approval required

        if approval.state == ApprovalState.APPROVED:
            return True
        elif approval.state == ApprovalState.REJECTED:
            rejection = next((d for d in approval.decisions if d.decision == ApprovalState.REJECTED), None)
            raise ApprovalDeniedError(approval.id, rejection.reason if rejection else None)
        elif approval.state == ApprovalState.EXPIRED:
            raise ApprovalExpiredError(approval.id, approval.expires_at)
        else:
            raise ApprovalRequiredError(approval.id, approval.gate_name, approval.risk_score)

    def get_pending_approvals(self) -> list[Approval]:
        return [a for a in self._approvals.values() if a.state == ApprovalState.PENDING]

    def get_pending_summary(self) -> PendingApprovalSummary:
        pending = self.get_pending_approvals()
        now = datetime.now(timezone.utc)
        by_gate: dict[str, int] = {}
        oldest_minutes: int | None = None
        high_risk_count = 0

        for a in pending:
            by_gate[a.gate_name] = by_gate.get(a.gate_name, 0) + 1
            age = int((now - a.created_at).total_seconds() / 60)
            if oldest_minutes is None or age > oldest_minutes:
                oldest_minutes = age
            if a.risk_score > 70:
                high_risk_count += 1

        return PendingApprovalSummary(
            total_pending=len(pending),
            oldest_pending_minutes=oldest_minutes,
            by_gate=by_gate,
            high_risk_count=high_risk_count,
        )

    def add_gate(self, gate: ApprovalGate) -> None:
        self.gates.append(gate)
        self._gates_by_id[gate.id] = gate
        self._gates_by_name[gate.name] = gate

    def remove_gate(self, gate_id: UUID) -> bool:
        gate = self._gates_by_id.get(gate_id)
        if not gate:
            return False
        self.gates.remove(gate)
        del self._gates_by_id[gate_id]
        del self._gates_by_name[gate.name]
        return True
