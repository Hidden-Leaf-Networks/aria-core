"""Plan engine data models."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member


class PlanState(StrEnum):
    """Plan lifecycle states."""

    DRAFT = "draft"
    PLANNED = "planned"
    QUEUED = "queued"
    EXECUTING = "executing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ActionState(StrEnum):
    """Action execution states."""

    PENDING = "pending"
    QUEUED = "queued"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Valid plan state transitions
VALID_PLAN_TRANSITIONS: dict[PlanState, set[PlanState]] = {
    PlanState.DRAFT: {PlanState.PLANNED, PlanState.ARCHIVED},
    PlanState.PLANNED: {PlanState.QUEUED, PlanState.DRAFT, PlanState.ARCHIVED},
    PlanState.QUEUED: {PlanState.EXECUTING, PlanState.ARCHIVED},
    PlanState.EXECUTING: {PlanState.BLOCKED, PlanState.COMPLETED, PlanState.FAILED},
    PlanState.BLOCKED: {PlanState.EXECUTING, PlanState.FAILED},
    PlanState.COMPLETED: {PlanState.ARCHIVED},
    PlanState.FAILED: {PlanState.DRAFT, PlanState.ARCHIVED},
    PlanState.ARCHIVED: set(),
}

# Valid action state transitions
VALID_ACTION_TRANSITIONS: dict[ActionState, set[ActionState]] = {
    ActionState.PENDING: {ActionState.QUEUED, ActionState.SKIPPED},
    ActionState.QUEUED: {ActionState.EXECUTING, ActionState.SKIPPED},
    ActionState.EXECUTING: {ActionState.AWAITING_APPROVAL, ActionState.COMPLETED, ActionState.FAILED},
    ActionState.AWAITING_APPROVAL: {ActionState.APPROVED, ActionState.FAILED, ActionState.SKIPPED},
    ActionState.APPROVED: {ActionState.EXECUTING},
    ActionState.COMPLETED: set(),
    ActionState.FAILED: set(),
    ActionState.SKIPPED: set(),
}


class PlanAction(BaseModel):
    """A single action within a plan."""

    id: UUID = Field(default_factory=uuid4)
    plan_id: UUID
    index: int = 0
    name: str
    description: str = ""
    skill_name: str | None = None
    skill_args: dict[str, Any] | None = None
    dependencies: list[int] = Field(default_factory=list)
    state: ActionState = ActionState.PENDING
    risk_score: int | None = None
    requires_approval: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanVersion(BaseModel):
    """Snapshot of a plan version for audit trail."""

    version: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    actions_snapshot: list[dict[str, Any]] = Field(default_factory=list)
    change_summary: str = ""


class Plan(BaseModel):
    """Execution plan with actions and dependency tracking."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    conversation_id: UUID | None = None
    prompt: str | None = None
    state: PlanState = PlanState.DRAFT
    actions: list[PlanAction] = Field(default_factory=list)
    current_action_index: int = 0
    aggregate_risk_score: int | None = None
    requires_approval: bool = False
    version: int = 1
    versions: list[PlanVersion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    planned_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_by: str = "system"

    @property
    def is_terminal(self) -> bool:
        return self.state in (PlanState.COMPLETED, PlanState.FAILED, PlanState.ARCHIVED)

    @property
    def progress(self) -> float:
        if not self.actions:
            return 0.0
        completed = sum(1 for a in self.actions if a.state == ActionState.COMPLETED)
        return completed / len(self.actions)
