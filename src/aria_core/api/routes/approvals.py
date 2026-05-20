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
