"""Approval management API routes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser, Role, require_role
from aria_core.api.deps import get_guard


async def get_approval(
    approval_id: UUID,
    user: AuthUser,
) -> dict[str, Any] | None:
    """Get an approval by ID."""
    guard = get_guard()
    approval = await guard.get_approval(user.tenant_id, approval_id)
    return approval.model_dump(mode="json") if approval else None


async def list_approvals(
    user: AuthUser,
    state: str | None = None,
    plan_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List approvals for the authenticated tenant."""
    guard = get_guard()
    approvals = await guard.list_approvals(
        user.tenant_id, state=state, plan_id=plan_id, limit=limit, offset=offset
    )
    return [a.model_dump(mode="json") for a in approvals]


async def approve_approval(
    approval_id: UUID,
    user: AuthUser,
) -> dict[str, Any]:
    """Approve a pending approval."""
    require_role(user, Role.OPERATOR)
    guard = get_guard()
    approval = await guard.get_approval(user.tenant_id, approval_id)
    if not approval:
        return {"error": "Approval not found"}

    from aria_core.permissions.models import ApprovalState
    from datetime import datetime, timezone

    if approval.state != ApprovalState.PENDING:
        return {"error": f"Approval is already {approval.state}"}

    from aria_core.permissions.models import ApprovalDecision
    decision = ApprovalDecision(
        approval_id=approval_id,
        decision=ApprovalState.APPROVED,
        approver_id=user.user_id,
        approver_type="user",
        reason="Approved via Config Portal",
    )
    updated = approval.model_copy(update={
        "state": ApprovalState.APPROVED,
        "decisions": list(approval.decisions) + [decision],
        "resolved_at": datetime.now(timezone.utc),
    })
    await guard.save_approval(user.tenant_id, updated)
    return updated.model_dump(mode="json")


async def reject_approval(
    approval_id: UUID,
    user: AuthUser,
) -> dict[str, Any]:
    """Reject a pending approval."""
    require_role(user, Role.OPERATOR)
    guard = get_guard()
    approval = await guard.get_approval(user.tenant_id, approval_id)
    if not approval:
        return {"error": "Approval not found"}

    from aria_core.permissions.models import ApprovalState
    from datetime import datetime, timezone

    if approval.state != ApprovalState.PENDING:
        return {"error": f"Approval is already {approval.state}"}

    from aria_core.permissions.models import ApprovalDecision
    decision = ApprovalDecision(
        approval_id=approval_id,
        decision=ApprovalState.REJECTED,
        approver_id=user.user_id,
        approver_type="user",
        reason="Rejected via Config Portal",
    )
    updated = approval.model_copy(update={
        "state": ApprovalState.REJECTED,
        "decisions": list(approval.decisions) + [decision],
        "resolved_at": datetime.now(timezone.utc),
    })
    await guard.save_approval(user.tenant_id, updated)
    return updated.model_dump(mode="json")
