"""SQLAlchemy 2.0 ORM models for PostgreSQL 16.

All tables include tenant_id with composite indexes for tenant-scoped queries.
Maps directly to existing Pydantic models in aria_core.

NOTE: Do NOT use `from __future__ import annotations` here.
SQLAlchemy's Mapped[] annotations must be evaluable at class definition time.
Use Optional[] instead of X | None for Python 3.10 compatibility.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------


class TenantRow(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    plans: Mapped[list["PlanRow"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    approvals: Mapped[list["ApprovalRow"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    events: Mapped[list["EventRow"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    contexts: Mapped[list["AgentContextRow"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    risk_policies: Mapped[list["RiskPolicyRow"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    approval_gates: Mapped[list["ApprovalGateRow"]] = relationship(back_populates="tenant", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class PlanRow(Base):
    __tablename__ = "plans"
    __table_args__ = (
        Index("ix_plans_tenant_state", "tenant_id", "state"),
        Index("ix_plans_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    conversation_id: Mapped[Optional[UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")
    current_action_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    aggregate_risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    versions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    planned_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(100), nullable=False, default="system")

    # Relationships
    tenant: Mapped["TenantRow"] = relationship(back_populates="plans")
    actions: Mapped[list["PlanActionRow"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="PlanActionRow.index"
    )


class PlanActionRow(Base):
    __tablename__ = "plan_actions"
    __table_args__ = (
        Index("ix_plan_actions_plan", "plan_id", "index"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    plan_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    skill_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    skill_args: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    dependencies: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    state: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    plan: Mapped["PlanRow"] = relationship(back_populates="actions")


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


class ApprovalRow(Base):
    __tablename__ = "approvals"
    __table_args__ = (
        Index("ix_approvals_tenant_state", "tenant_id", "state"),
        Index("ix_approvals_tenant_plan", "tenant_id", "plan_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    plan_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    action_id: Mapped[Optional[UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    gate_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    gate_name: Mapped[str] = mapped_column(String(100), nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_factors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    required_approvals: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    tenant: Mapped["TenantRow"] = relationship(back_populates="approvals")
    decisions: Mapped[list["ApprovalDecisionRow"]] = relationship(
        back_populates="approval", cascade="all, delete-orphan"
    )


class ApprovalDecisionRow(Base):
    __tablename__ = "approval_decisions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    approval_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("approvals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    approver_id: Mapped[str] = mapped_column(String(200), nullable=False)
    approver_type: Mapped[str] = mapped_column(String(50), nullable=False, default="user")
    reason: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    approval: Mapped["ApprovalRow"] = relationship(back_populates="decisions")


# ---------------------------------------------------------------------------
# Events (append-only audit trail)
# ---------------------------------------------------------------------------


class EventRow(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_tenant_type", "tenant_id", "event_type"),
        Index("ix_events_tenant_timestamp", "tenant_id", "timestamp"),
        Index("ix_events_tenant_agent", "tenant_id", "agent_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    agent_id: Mapped[Optional[UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    context_id: Mapped[Optional[UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    tenant: Mapped["TenantRow"] = relationship(back_populates="events")


# ---------------------------------------------------------------------------
# Agent contexts
# ---------------------------------------------------------------------------


class AgentContextRow(Base):
    __tablename__ = "agent_contexts"
    __table_args__ = (
        Index("ix_contexts_tenant_conversation", "tenant_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    conversation_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    messages: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    current_plan_id: Mapped[Optional[UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    current_step_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skill_results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    tenant: Mapped["TenantRow"] = relationship(back_populates="contexts")


# ---------------------------------------------------------------------------
# Risk policies (tenant-scoped)
# ---------------------------------------------------------------------------


class RiskPolicyRow(Base):
    __tablename__ = "risk_policies"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    approval_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    block_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=80)
    skill_category_weights: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    impact_scope_weights: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    first_execution_modifier: Mapped[float] = mapped_column(Float, nullable=False, default=1.2)
    failure_history_modifier: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    violation_history_modifier: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    tenant: Mapped["TenantRow"] = relationship(back_populates="risk_policies")


# ---------------------------------------------------------------------------
# Approval gates (tenant-scoped)
# ---------------------------------------------------------------------------


class ApprovalGateRow(Base):
    __tablename__ = "approval_gates"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_threshold: Mapped[int] = mapped_column(Integer, nullable=False)
    required_approvers: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    allowed_approvers: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    timeout_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    auto_escalate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    escalation_after_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    escalation_to: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    tenant: Mapped["TenantRow"] = relationship(back_populates="approval_gates")
