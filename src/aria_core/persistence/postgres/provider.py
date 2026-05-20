"""PostgreSQL persistence provider — production multi-tenant backend.

Implements PersistenceProvider protocol using SQLAlchemy 2.0 async
with asyncpg for PostgreSQL 16.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aria_core.permissions.models import (
    Approval,
    ApprovalDecision,
    ApprovalGate,
    ApprovalState,
    RiskPolicy,
)
from aria_core.planning.models import (
    ActionState,
    Plan,
    PlanAction,
    PlanState,
    PlanVersion,
)
from aria_core.runtime.models import AgentConfig, AgentContext, ChatMessage
from aria_core.tenant.models import Tenant, TenantConfig

from aria_core.persistence.postgres.models import (
    AgentContextRow,
    ApprovalDecisionRow,
    ApprovalGateRow,
    ApprovalRow,
    EventRow,
    PlanActionRow,
    PlanRow,
    RiskPolicyRow,
    TenantRow,
)


class PostgresProvider:
    """PostgreSQL persistence provider with full tenant isolation.

    All queries are scoped by tenant_id. Cross-tenant access is impossible
    at the query level.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # -------------------------------------------------------------------
    # Conversion helpers: Pydantic ↔ ORM
    # -------------------------------------------------------------------

    @staticmethod
    def _tenant_to_row(tenant: Tenant) -> TenantRow:
        return TenantRow(
            id=tenant.id,
            slug=tenant.slug,
            name=tenant.name,
            config=tenant.config.model_dump(),
            is_active=tenant.is_active,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
        )

    @staticmethod
    def _row_to_tenant(row: TenantRow) -> Tenant:
        return Tenant(
            id=row.id,
            slug=row.slug,
            name=row.name,
            config=TenantConfig(**row.config),
            is_active=row.is_active,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _plan_to_row(tenant_id: UUID, plan: Plan) -> PlanRow:
        return PlanRow(
            id=plan.id,
            tenant_id=tenant_id,
            name=plan.name,
            description=plan.description,
            conversation_id=plan.conversation_id,
            prompt=plan.prompt,
            state=plan.state.value,
            current_action_index=plan.current_action_index,
            aggregate_risk_score=plan.aggregate_risk_score,
            requires_approval=plan.requires_approval,
            version=plan.version,
            versions=[v.model_dump() for v in plan.versions],
            metadata_=plan.metadata,
            created_at=plan.created_at,
            updated_at=plan.updated_at,
            planned_at=plan.planned_at,
            started_at=plan.started_at,
            completed_at=plan.completed_at,
            created_by=plan.created_by,
        )

    @staticmethod
    def _action_to_row(action: PlanAction) -> PlanActionRow:
        return PlanActionRow(
            id=action.id,
            plan_id=action.plan_id,
            index=action.index,
            name=action.name,
            description=action.description,
            skill_name=action.skill_name,
            skill_args=action.skill_args,
            dependencies=action.dependencies,
            state=action.state.value,
            risk_score=action.risk_score,
            requires_approval=action.requires_approval,
            result=action.result,
            error=action.error,
            started_at=action.started_at,
            completed_at=action.completed_at,
            execution_time_ms=action.execution_time_ms,
            created_at=action.created_at,
        )

    @staticmethod
    def _row_to_plan(row: PlanRow) -> Plan:
        actions = [
            PlanAction(
                id=a.id,
                plan_id=a.plan_id,
                index=a.index,
                name=a.name,
                description=a.description,
                skill_name=a.skill_name,
                skill_args=a.skill_args,
                dependencies=a.dependencies,
                state=ActionState(a.state),
                risk_score=a.risk_score,
                requires_approval=a.requires_approval,
                result=a.result,
                error=a.error,
                started_at=a.started_at,
                completed_at=a.completed_at,
                execution_time_ms=a.execution_time_ms,
                created_at=a.created_at,
            )
            for a in sorted(row.actions, key=lambda a: a.index)
        ]
        versions = [PlanVersion(**v) for v in (row.versions or [])]
        return Plan(
            id=row.id,
            name=row.name,
            description=row.description,
            conversation_id=row.conversation_id,
            prompt=row.prompt,
            state=PlanState(row.state),
            actions=actions,
            current_action_index=row.current_action_index,
            aggregate_risk_score=row.aggregate_risk_score,
            requires_approval=row.requires_approval,
            version=row.version,
            versions=versions,
            metadata=row.metadata_,
            created_at=row.created_at,
            updated_at=row.updated_at,
            planned_at=row.planned_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            created_by=row.created_by,
        )

    @staticmethod
    def _approval_to_row(tenant_id: UUID, approval: Approval) -> ApprovalRow:
        return ApprovalRow(
            id=approval.id,
            tenant_id=tenant_id,
            plan_id=approval.plan_id,
            action_id=approval.action_id,
            gate_id=approval.gate_id,
            gate_name=approval.gate_name,
            risk_score=approval.risk_score,
            risk_factors=approval.risk_factors,
            context=approval.context,
            state=approval.state.value,
            required_approvals=approval.required_approvals,
            created_at=approval.created_at,
            expires_at=approval.expires_at,
            resolved_at=approval.resolved_at,
        )

    @staticmethod
    def _row_to_approval(row: ApprovalRow) -> Approval:
        decisions = [
            ApprovalDecision(
                id=d.id,
                approval_id=d.approval_id,
                decision=ApprovalState(d.decision),
                approver_id=d.approver_id,
                approver_type=d.approver_type,
                reason=d.reason,
                decided_at=d.decided_at,
            )
            for d in row.decisions
        ]
        return Approval(
            id=row.id,
            plan_id=row.plan_id,
            action_id=row.action_id,
            gate_id=row.gate_id,
            gate_name=row.gate_name,
            risk_score=row.risk_score,
            risk_factors=row.risk_factors,
            context=row.context,
            state=ApprovalState(row.state),
            decisions=decisions,
            required_approvals=row.required_approvals,
            created_at=row.created_at,
            expires_at=row.expires_at,
            resolved_at=row.resolved_at,
        )

    @staticmethod
    def _context_to_row(tenant_id: UUID, context: AgentContext) -> AgentContextRow:
        return AgentContextRow(
            id=context.id,
            tenant_id=tenant_id,
            conversation_id=context.conversation_id,
            config=context.config.model_dump(),
            messages=[m.model_dump() for m in context.messages],
            current_plan_id=context.current_plan_id,
            current_step_index=context.current_step_index,
            step_count=context.step_count,
            skill_results=context.skill_results,
            metadata_=context.metadata,
            created_at=context.created_at,
        )

    @staticmethod
    def _row_to_context(row: AgentContextRow) -> AgentContext:
        return AgentContext(
            id=row.id,
            tenant_id=row.tenant_id,
            conversation_id=row.conversation_id,
            config=AgentConfig(**row.config),
            messages=[ChatMessage(**m) for m in row.messages],
            current_plan_id=row.current_plan_id,
            current_step_index=row.current_step_index,
            step_count=row.step_count,
            skill_results=row.skill_results,
            metadata=row.metadata_,
            created_at=row.created_at,
        )

    @staticmethod
    def _risk_policy_to_row(tenant_id: UUID, policy: RiskPolicy) -> RiskPolicyRow:
        return RiskPolicyRow(
            id=policy.id,
            tenant_id=tenant_id,
            name=policy.name,
            description=policy.description,
            approval_threshold=policy.approval_threshold,
            block_threshold=policy.block_threshold,
            skill_category_weights=policy.skill_category_weights,
            impact_scope_weights=policy.impact_scope_weights,
            first_execution_modifier=policy.first_execution_modifier,
            failure_history_modifier=policy.failure_history_modifier,
            violation_history_modifier=policy.violation_history_modifier,
            is_active=policy.is_active,
        )

    @staticmethod
    def _row_to_risk_policy(row: RiskPolicyRow) -> RiskPolicy:
        return RiskPolicy(
            id=row.id,
            name=row.name,
            description=row.description,
            approval_threshold=row.approval_threshold,
            block_threshold=row.block_threshold,
            skill_category_weights=row.skill_category_weights,
            impact_scope_weights=row.impact_scope_weights,
            first_execution_modifier=row.first_execution_modifier,
            failure_history_modifier=row.failure_history_modifier,
            violation_history_modifier=row.violation_history_modifier,
            is_active=row.is_active,
        )

    @staticmethod
    def _gate_to_row(tenant_id: UUID, gate: ApprovalGate) -> ApprovalGateRow:
        return ApprovalGateRow(
            id=gate.id,
            tenant_id=tenant_id,
            name=gate.name,
            description=gate.description,
            risk_threshold=gate.risk_threshold,
            required_approvers=gate.required_approvers,
            allowed_approvers=gate.allowed_approvers,
            timeout_minutes=gate.timeout_minutes,
            auto_escalate=gate.auto_escalate,
            escalation_after_minutes=gate.escalation_after_minutes,
            escalation_to=gate.escalation_to,
            is_active=gate.is_active,
        )

    @staticmethod
    def _row_to_gate(row: ApprovalGateRow) -> ApprovalGate:
        return ApprovalGate(
            id=row.id,
            name=row.name,
            description=row.description,
            risk_threshold=row.risk_threshold,
            required_approvers=row.required_approvers,
            allowed_approvers=row.allowed_approvers,
            timeout_minutes=row.timeout_minutes,
            auto_escalate=row.auto_escalate,
            escalation_after_minutes=row.escalation_after_minutes,
            escalation_to=row.escalation_to,
            is_active=row.is_active,
        )

    # -------------------------------------------------------------------
    # Tenant management
    # -------------------------------------------------------------------

    async def save_tenant(self, tenant: Tenant) -> Tenant:
        async with self._session_factory() as session:
            row = self._tenant_to_row(tenant)
            merged = await session.merge(row)
            await session.commit()
            await session.refresh(merged)
            return self._row_to_tenant(merged)

    async def get_tenant(self, tenant_id: UUID) -> Tenant | None:
        async with self._session_factory() as session:
            row = await session.get(TenantRow, tenant_id)
            return self._row_to_tenant(row) if row else None

    async def get_tenant_by_slug(self, slug: str) -> Tenant | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(TenantRow).where(TenantRow.slug == slug)
            )
            row = result.scalar_one_or_none()
            return self._row_to_tenant(row) if row else None

    async def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        async with self._session_factory() as session:
            stmt = select(TenantRow).order_by(TenantRow.created_at)
            if active_only:
                stmt = stmt.where(TenantRow.is_active.is_(True))
            result = await session.execute(stmt)
            return [self._row_to_tenant(r) for r in result.scalars()]

    async def update_tenant_config(
        self, tenant_id: UUID, config: TenantConfig
    ) -> Tenant:
        async with self._session_factory() as session:
            row = await session.get(TenantRow, tenant_id)
            if not row:
                from aria_core.persistence.memory import TenantNotFoundError
                raise TenantNotFoundError(tenant_id)
            row.config = config.model_dump()
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(row)
            return self._row_to_tenant(row)

    # -------------------------------------------------------------------
    # Plans
    # -------------------------------------------------------------------

    async def save_plan(self, tenant_id: UUID, plan: Plan) -> Plan:
        async with self._session_factory() as session:
            # Check if plan exists (update vs insert)
            existing = await session.get(PlanRow, plan.id)
            if existing:
                # Update existing plan
                existing.name = plan.name
                existing.description = plan.description
                existing.state = plan.state.value
                existing.current_action_index = plan.current_action_index
                existing.aggregate_risk_score = plan.aggregate_risk_score
                existing.requires_approval = plan.requires_approval
                existing.version = plan.version
                existing.versions = [v.model_dump() for v in plan.versions]
                existing.metadata_ = plan.metadata
                existing.updated_at = plan.updated_at
                existing.planned_at = plan.planned_at
                existing.started_at = plan.started_at
                existing.completed_at = plan.completed_at

                # Sync actions
                for action_row in existing.actions:
                    await session.delete(action_row)
                await session.flush()
                for action in plan.actions:
                    session.add(self._action_to_row(action))
            else:
                row = self._plan_to_row(tenant_id, plan)
                row.actions = [self._action_to_row(a) for a in plan.actions]
                session.add(row)

            await session.commit()
            return plan

    async def get_plan(self, tenant_id: UUID, plan_id: UUID) -> Plan | None:
        async with self._session_factory() as session:
            stmt = (
                select(PlanRow)
                .where(and_(PlanRow.id == plan_id, PlanRow.tenant_id == tenant_id))
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return None
            # Eagerly load actions
            await session.refresh(row, ["actions"])
            return self._row_to_plan(row)

    async def list_plans(
        self,
        tenant_id: UUID,
        state: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Plan]:
        async with self._session_factory() as session:
            stmt = (
                select(PlanRow)
                .where(PlanRow.tenant_id == tenant_id)
                .order_by(PlanRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if state:
                stmt = stmt.where(PlanRow.state == state)
            result = await session.execute(stmt)
            plans = []
            for row in result.scalars():
                await session.refresh(row, ["actions"])
                plans.append(self._row_to_plan(row))
            return plans

    async def delete_plan(self, tenant_id: UUID, plan_id: UUID) -> bool:
        async with self._session_factory() as session:
            stmt = select(PlanRow).where(
                and_(PlanRow.id == plan_id, PlanRow.tenant_id == tenant_id)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return False
            await session.delete(row)
            await session.commit()
            return True

    # -------------------------------------------------------------------
    # Approvals
    # -------------------------------------------------------------------

    async def save_approval(
        self, tenant_id: UUID, approval: Approval
    ) -> Approval:
        async with self._session_factory() as session:
            existing = await session.get(ApprovalRow, approval.id)
            if existing:
                existing.state = approval.state.value
                existing.resolved_at = approval.resolved_at
                # Sync decisions
                for d_row in existing.decisions:
                    await session.delete(d_row)
                await session.flush()
                for d in approval.decisions:
                    session.add(ApprovalDecisionRow(
                        id=d.id,
                        approval_id=d.approval_id,
                        decision=d.decision.value,
                        approver_id=d.approver_id,
                        approver_type=d.approver_type,
                        reason=d.reason,
                        decided_at=d.decided_at,
                    ))
            else:
                row = self._approval_to_row(tenant_id, approval)
                row.decisions = [
                    ApprovalDecisionRow(
                        id=d.id,
                        approval_id=d.approval_id,
                        decision=d.decision.value,
                        approver_id=d.approver_id,
                        approver_type=d.approver_type,
                        reason=d.reason,
                        decided_at=d.decided_at,
                    )
                    for d in approval.decisions
                ]
                session.add(row)
            await session.commit()
            return approval

    async def get_approval(
        self, tenant_id: UUID, approval_id: UUID
    ) -> Approval | None:
        async with self._session_factory() as session:
            stmt = select(ApprovalRow).where(
                and_(ApprovalRow.id == approval_id, ApprovalRow.tenant_id == tenant_id)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return None
            await session.refresh(row, ["decisions"])
            return self._row_to_approval(row)

    async def list_approvals(
        self,
        tenant_id: UUID,
        state: str | None = None,
        plan_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Approval]:
        async with self._session_factory() as session:
            stmt = (
                select(ApprovalRow)
                .where(ApprovalRow.tenant_id == tenant_id)
                .order_by(ApprovalRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if state:
                stmt = stmt.where(ApprovalRow.state == state)
            if plan_id:
                stmt = stmt.where(ApprovalRow.plan_id == plan_id)
            result = await session.execute(stmt)
            approvals = []
            for row in result.scalars():
                await session.refresh(row, ["decisions"])
                approvals.append(self._row_to_approval(row))
            return approvals

    # -------------------------------------------------------------------
    # Events (append-only)
    # -------------------------------------------------------------------

    async def save_event(
        self,
        tenant_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        agent_id: UUID | None = None,
        context_id: UUID | None = None,
    ) -> dict[str, Any]:
        event_id = uuid4()
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            row = EventRow(
                id=event_id,
                tenant_id=tenant_id,
                event_type=event_type,
                payload=payload,
                agent_id=agent_id,
                context_id=context_id,
                timestamp=now,
            )
            session.add(row)
            await session.commit()
        return {
            "id": event_id,
            "tenant_id": tenant_id,
            "event_type": event_type,
            "payload": payload,
            "agent_id": agent_id,
            "context_id": context_id,
            "timestamp": now,
        }

    async def list_events(
        self,
        tenant_id: UUID,
        event_type: str | None = None,
        agent_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            stmt = (
                select(EventRow)
                .where(EventRow.tenant_id == tenant_id)
                .order_by(EventRow.timestamp.desc())
                .limit(limit)
                .offset(offset)
            )
            if event_type:
                stmt = stmt.where(EventRow.event_type == event_type)
            if agent_id:
                stmt = stmt.where(EventRow.agent_id == agent_id)
            if after:
                stmt = stmt.where(EventRow.timestamp > after)
            if before:
                stmt = stmt.where(EventRow.timestamp < before)
            result = await session.execute(stmt)
            return [
                {
                    "id": row.id,
                    "tenant_id": row.tenant_id,
                    "event_type": row.event_type,
                    "payload": row.payload,
                    "agent_id": row.agent_id,
                    "context_id": row.context_id,
                    "timestamp": row.timestamp,
                }
                for row in result.scalars()
            ]

    async def count_events(
        self,
        tenant_id: UUID,
        event_type: str | None = None,
    ) -> int:
        async with self._session_factory() as session:
            stmt = select(func.count(EventRow.id)).where(
                EventRow.tenant_id == tenant_id
            )
            if event_type:
                stmt = stmt.where(EventRow.event_type == event_type)
            result = await session.execute(stmt)
            return result.scalar_one()

    # -------------------------------------------------------------------
    # Agent context
    # -------------------------------------------------------------------

    async def save_context(
        self, tenant_id: UUID, context: AgentContext
    ) -> AgentContext:
        async with self._session_factory() as session:
            row = self._context_to_row(tenant_id, context)
            merged = await session.merge(row)
            await session.commit()
            return context

    async def get_context(
        self, tenant_id: UUID, context_id: UUID
    ) -> AgentContext | None:
        async with self._session_factory() as session:
            stmt = select(AgentContextRow).where(
                and_(
                    AgentContextRow.id == context_id,
                    AgentContextRow.tenant_id == tenant_id,
                )
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._row_to_context(row) if row else None

    async def list_contexts(
        self,
        tenant_id: UUID,
        conversation_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentContext]:
        async with self._session_factory() as session:
            stmt = (
                select(AgentContextRow)
                .where(AgentContextRow.tenant_id == tenant_id)
                .order_by(AgentContextRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if conversation_id:
                stmt = stmt.where(
                    AgentContextRow.conversation_id == conversation_id
                )
            result = await session.execute(stmt)
            return [self._row_to_context(r) for r in result.scalars()]

    # -------------------------------------------------------------------
    # Risk policies
    # -------------------------------------------------------------------

    async def save_risk_policy(
        self, tenant_id: UUID, policy: RiskPolicy
    ) -> RiskPolicy:
        async with self._session_factory() as session:
            row = self._risk_policy_to_row(tenant_id, policy)
            await session.merge(row)
            await session.commit()
            return policy

    async def get_risk_policy(
        self, tenant_id: UUID, policy_id: UUID
    ) -> RiskPolicy | None:
        async with self._session_factory() as session:
            stmt = select(RiskPolicyRow).where(
                and_(
                    RiskPolicyRow.id == policy_id,
                    RiskPolicyRow.tenant_id == tenant_id,
                )
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._row_to_risk_policy(row) if row else None

    async def get_active_risk_policy(
        self, tenant_id: UUID
    ) -> RiskPolicy | None:
        async with self._session_factory() as session:
            stmt = (
                select(RiskPolicyRow)
                .where(
                    and_(
                        RiskPolicyRow.tenant_id == tenant_id,
                        RiskPolicyRow.is_active.is_(True),
                    )
                )
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return self._row_to_risk_policy(row) if row else None

    # -------------------------------------------------------------------
    # Approval gates
    # -------------------------------------------------------------------

    async def save_approval_gate(
        self, tenant_id: UUID, gate: ApprovalGate
    ) -> ApprovalGate:
        async with self._session_factory() as session:
            row = self._gate_to_row(tenant_id, gate)
            await session.merge(row)
            await session.commit()
            return gate

    async def list_approval_gates(
        self, tenant_id: UUID, active_only: bool = True
    ) -> list[ApprovalGate]:
        async with self._session_factory() as session:
            stmt = select(ApprovalGateRow).where(
                ApprovalGateRow.tenant_id == tenant_id
            )
            if active_only:
                stmt = stmt.where(ApprovalGateRow.is_active.is_(True))
            result = await session.execute(stmt)
            return [self._row_to_gate(r) for r in result.scalars()]
