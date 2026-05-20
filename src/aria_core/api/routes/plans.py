"""Plan management API routes — CRUD + lifecycle."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser, Role, require_role
from aria_core.api.deps import get_guard
from aria_core.planning.models import Plan, PlanAction, PlanState


async def create_plan(
    data: dict[str, Any],
    user: AuthUser,
) -> dict[str, Any]:
    """Create a new plan in DRAFT state."""
    require_role(user, Role.OPERATOR)
    guard = get_guard()

    from uuid import uuid4
    from datetime import datetime, timezone

    plan_id = uuid4()
    now = datetime.now(timezone.utc)

    actions = []
    for idx, action_data in enumerate(data.get("actions", [])):
        actions.append(PlanAction(
            plan_id=plan_id,
            index=idx,
            name=action_data.get("name", f"Action {idx + 1}"),
            description=action_data.get("description", ""),
            skill_name=action_data.get("skill_name"),
            skill_args=action_data.get("skill_args"),
            dependencies=action_data.get("dependencies", []),
        ))

    plan = Plan(
        id=plan_id,
        name=data["name"],
        description=data.get("description", ""),
        state=PlanState.DRAFT,
        actions=actions,
        created_at=now,
        updated_at=now,
        created_by=user.user_id,
    )

    saved = await guard.save_plan(user.tenant_id, plan)
    return saved.model_dump(mode="json")


async def get_plan(plan_id: UUID, user: AuthUser) -> dict[str, Any] | None:
    """Get a plan by ID."""
    guard = get_guard()
    plan = await guard.get_plan(user.tenant_id, plan_id)
    return plan.model_dump(mode="json") if plan else None


async def list_plans(
    user: AuthUser,
    state: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List plans for the authenticated tenant."""
    guard = get_guard()
    plans = await guard.list_plans(user.tenant_id, state=state, limit=limit, offset=offset)
    return [p.model_dump(mode="json") for p in plans]


async def delete_plan(plan_id: UUID, user: AuthUser) -> dict[str, Any]:
    """Delete a plan. Operator+."""
    require_role(user, Role.OPERATOR)
    guard = get_guard()
    deleted = await guard.delete_plan(user.tenant_id, plan_id)
    return {"deleted": deleted, "plan_id": str(plan_id)}
