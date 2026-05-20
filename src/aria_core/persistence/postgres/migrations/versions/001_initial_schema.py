"""Initial multi-tenant schema.

Revision ID: 001
Revises: None
Create Date: 2026-05-20
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Tenants ---
    op.create_table(
        "tenants",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(63), unique=True, nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("config", JSON, nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # --- Plans ---
    op.create_table(
        "plans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=True),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("state", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("current_action_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("aggregate_risk_score", sa.Integer, nullable=True),
        sa.Column("requires_approval", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("versions", JSON, nullable=False, server_default="[]"),
        sa.Column("metadata", JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("planned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )
    op.create_index("ix_plans_tenant_state", "plans", ["tenant_id", "state"])
    op.create_index("ix_plans_tenant_created", "plans", ["tenant_id", "created_at"])

    # --- Plan Actions ---
    op.create_table(
        "plan_actions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("plan_id", UUID(as_uuid=True), sa.ForeignKey("plans.id", ondelete="CASCADE"), nullable=False),
        sa.Column("index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("skill_name", sa.String(200), nullable=True),
        sa.Column("skill_args", JSON, nullable=True),
        sa.Column("dependencies", JSON, nullable=False, server_default="[]"),
        sa.Column("state", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("risk_score", sa.Integer, nullable=True),
        sa.Column("requires_approval", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("result", JSON, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_time_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_plan_actions_plan", "plan_actions", ["plan_id", "index"])

    # --- Approvals ---
    op.create_table(
        "approvals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("plan_id", UUID(as_uuid=True), nullable=False),
        sa.Column("action_id", UUID(as_uuid=True), nullable=True),
        sa.Column("gate_id", UUID(as_uuid=True), nullable=False),
        sa.Column("gate_name", sa.String(100), nullable=False),
        sa.Column("risk_score", sa.Integer, nullable=False),
        sa.Column("risk_factors", JSON, nullable=False, server_default="[]"),
        sa.Column("context", JSON, nullable=False, server_default="{}"),
        sa.Column("state", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("required_approvals", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_approvals_tenant_state", "approvals", ["tenant_id", "state"])
    op.create_index("ix_approvals_tenant_plan", "approvals", ["tenant_id", "plan_id"])

    # --- Approval Decisions ---
    op.create_table(
        "approval_decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("approval_id", UUID(as_uuid=True), sa.ForeignKey("approvals.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("approver_id", sa.String(200), nullable=False),
        sa.Column("approver_type", sa.String(50), nullable=False, server_default="user"),
        sa.Column("reason", sa.String(1000), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
    )

    # --- Events (append-only audit trail) ---
    op.create_table(
        "events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", JSON, nullable=False, server_default="{}"),
        sa.Column("agent_id", UUID(as_uuid=True), nullable=True),
        sa.Column("context_id", UUID(as_uuid=True), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_events_tenant_type", "events", ["tenant_id", "event_type"])
    op.create_index("ix_events_tenant_timestamp", "events", ["tenant_id", "timestamp"])
    op.create_index("ix_events_tenant_agent", "events", ["tenant_id", "agent_id"])

    # --- Agent Contexts ---
    op.create_table(
        "agent_contexts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("config", JSON, nullable=False, server_default="{}"),
        sa.Column("messages", JSON, nullable=False, server_default="[]"),
        sa.Column("current_plan_id", UUID(as_uuid=True), nullable=True),
        sa.Column("current_step_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("step_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("skill_results", JSON, nullable=False, server_default="{}"),
        sa.Column("metadata", JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_contexts_tenant_conversation", "agent_contexts", ["tenant_id", "conversation_id"])

    # --- Risk Policies ---
    op.create_table(
        "risk_policies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("approval_threshold", sa.Integer, nullable=False, server_default="50"),
        sa.Column("block_threshold", sa.Integer, nullable=False, server_default="80"),
        sa.Column("skill_category_weights", JSON, nullable=False, server_default="{}"),
        sa.Column("impact_scope_weights", JSON, nullable=False, server_default="{}"),
        sa.Column("first_execution_modifier", sa.Float, nullable=False, server_default="1.2"),
        sa.Column("failure_history_modifier", sa.Float, nullable=False, server_default="0.05"),
        sa.Column("violation_history_modifier", sa.Float, nullable=False, server_default="0.1"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )

    # --- Approval Gates ---
    op.create_table(
        "approval_gates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("risk_threshold", sa.Integer, nullable=False),
        sa.Column("required_approvers", sa.Integer, nullable=False, server_default="1"),
        sa.Column("allowed_approvers", JSON, nullable=False, server_default="[]"),
        sa.Column("timeout_minutes", sa.Integer, nullable=False, server_default="60"),
        sa.Column("auto_escalate", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("escalation_after_minutes", sa.Integer, nullable=True),
        sa.Column("escalation_to", sa.String(200), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )


def downgrade() -> None:
    op.drop_table("approval_gates")
    op.drop_table("risk_policies")
    op.drop_table("agent_contexts")
    op.drop_table("events")
    op.drop_table("approval_decisions")
    op.drop_table("approvals")
    op.drop_table("plan_actions")
    op.drop_table("plans")
    op.drop_table("tenants")
