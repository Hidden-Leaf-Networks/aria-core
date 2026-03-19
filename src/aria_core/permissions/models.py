"""Permission system data models.

Self-contained models for risk scoring, approval workflows, and audit.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field, computed_field

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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SkillCategory(StrEnum):
    """Skill classification for risk scoring."""

    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    EXTERNAL = "external"


class ImpactScope(StrEnum):
    """Scope of impact for risk scoring."""

    LOCAL = "local"
    USER = "user"
    SYSTEM = "system"
    EXTERNAL = "external"


class ApprovalState(StrEnum):
    """Approval decision states."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# Risk models
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    """Individual risk factor contributing to overall score."""

    name: str
    category: str
    weight: float = Field(ge=0.0, le=1.0)
    raw_value: float = Field(ge=0.0, le=100.0)

    @computed_field
    @property
    def weighted_value(self) -> float:
        return self.raw_value * self.weight


class RiskScoreInput(BaseModel):
    """Input parameters for risk calculation."""

    skill_name: str
    skill_category: SkillCategory
    impact_scope: ImpactScope

    # Historical context
    historical_failures: int = Field(default=0, ge=0)
    historical_violations: int = Field(default=0, ge=0)

    # Context modifiers
    is_first_execution: bool = False
    has_sensitive_args: bool = False
    targets_external_system: bool = False
    requires_network: bool = False
    modifies_persistent_state: bool = False


class RiskScore(BaseModel):
    """Deterministic risk score for an action or plan."""

    id: UUID = Field(default_factory=uuid4)
    score: int = Field(ge=0, le=100)
    level: str
    requires_approval: bool
    factors: list[RiskFactor] = Field(default_factory=list)
    input_hash: str
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_version: str = "1.0.0"

    model_config = {"frozen": True}

    @classmethod
    def get_level_from_score(cls, score: int) -> str:
        if score <= 20:
            return "safe"
        elif score <= 40:
            return "low"
        elif score <= 60:
            return "medium"
        elif score <= 80:
            return "high"
        else:
            return "critical"


class RiskPolicy(BaseModel):
    """Risk policy configuration."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=100)
    description: str | None = None

    approval_threshold: int = Field(default=50, ge=0, le=100)
    block_threshold: int = Field(default=80, ge=0, le=100)

    skill_category_weights: dict[str, float] = Field(
        default_factory=lambda: {
            SkillCategory.READ: 0.1,
            SkillCategory.WRITE: 0.3,
            SkillCategory.EXEC: 0.5,
            SkillCategory.EXTERNAL: 0.4,
        }
    )
    impact_scope_weights: dict[str, float] = Field(
        default_factory=lambda: {
            ImpactScope.LOCAL: 0.1,
            ImpactScope.USER: 0.3,
            ImpactScope.SYSTEM: 0.6,
            ImpactScope.EXTERNAL: 0.5,
        }
    )

    first_execution_modifier: float = 1.2
    failure_history_modifier: float = 0.05
    violation_history_modifier: float = 0.1

    is_active: bool = True


# ---------------------------------------------------------------------------
# Approval models
# ---------------------------------------------------------------------------


class ApprovalGate(BaseModel):
    """Approval gate configuration."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=100)
    description: str | None = None
    risk_threshold: int = Field(ge=0, le=100)
    required_approvers: int = Field(default=1, ge=1)
    allowed_approvers: list[str] = Field(default_factory=list)
    timeout_minutes: int = Field(default=60, ge=1)
    auto_escalate: bool = False
    escalation_after_minutes: int | None = None
    escalation_to: str | None = None
    is_active: bool = True

    def get_expiry_time(self, from_time: datetime | None = None) -> datetime:
        start = from_time or datetime.now(timezone.utc)
        return start + timedelta(minutes=self.timeout_minutes)


class ApprovalDecision(BaseModel):
    """Individual approval decision (immutable)."""

    id: UUID = Field(default_factory=uuid4)
    approval_id: UUID
    decision: ApprovalState
    approver_id: str
    approver_type: str = "user"
    reason: str | None = Field(None, max_length=1000)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}


class Approval(BaseModel):
    """Approval request for a plan or action."""

    id: UUID = Field(default_factory=uuid4)
    plan_id: UUID
    action_id: UUID | None = None
    gate_id: UUID
    gate_name: str
    risk_score: int = Field(ge=0, le=100)
    risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    state: ApprovalState = ApprovalState.PENDING
    decisions: list[ApprovalDecision] = Field(default_factory=list)
    required_approvals: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    resolved_at: datetime | None = None

    @computed_field
    @property
    def approval_count(self) -> int:
        return sum(1 for d in self.decisions if d.decision == ApprovalState.APPROVED)

    @computed_field
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at and self.state == ApprovalState.PENDING

    @computed_field
    @property
    def is_resolved(self) -> bool:
        return self.state != ApprovalState.PENDING


class ApprovalResponse(BaseModel):
    """Response after approval decision."""

    approval: Approval
    decision: ApprovalDecision
    plan_state: str


class PendingApprovalSummary(BaseModel):
    """Summary of pending approvals."""

    total_pending: int = 0
    oldest_pending_minutes: int | None = None
    by_gate: dict[str, int] = Field(default_factory=dict)
    high_risk_count: int = 0
