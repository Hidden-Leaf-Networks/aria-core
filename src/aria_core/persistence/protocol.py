"""Persistence provider protocol — abstract interface for storage backends.

All methods are tenant-scoped. The tenant_id parameter is mandatory
to enforce data isolation at the protocol level.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from aria_core.permissions.models import Approval, ApprovalGate, RiskPolicy
from aria_core.planning.models import Plan
from aria_core.runtime.models import AgentContext
from aria_core.tenant.models import Tenant, TenantConfig


class Event(Protocol):
    """Minimal event shape for type checking."""

    id: UUID
    tenant_id: UUID
    event_type: str
    payload: dict[str, Any]
    timestamp: datetime


@runtime_checkable
class PersistenceProvider(Protocol):
    """Abstract persistence interface.

    All operations are tenant-scoped. Implementations MUST validate
    that tenant_id matches the entity being stored/retrieved.
    """

    # -------------------------------------------------------------------
    # Tenant management
    # -------------------------------------------------------------------

    async def save_tenant(self, tenant: Tenant) -> Tenant: ...

    async def get_tenant(self, tenant_id: UUID) -> Tenant | None: ...

    async def get_tenant_by_slug(self, slug: str) -> Tenant | None: ...

    async def list_tenants(self, active_only: bool = True) -> list[Tenant]: ...

    async def update_tenant_config(
        self, tenant_id: UUID, config: TenantConfig
    ) -> Tenant: ...

    # -------------------------------------------------------------------
    # Plans
    # -------------------------------------------------------------------

    async def save_plan(self, tenant_id: UUID, plan: Plan) -> Plan: ...

    async def get_plan(self, tenant_id: UUID, plan_id: UUID) -> Plan | None: ...

    async def list_plans(
        self,
        tenant_id: UUID,
        state: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Plan]: ...

    async def delete_plan(self, tenant_id: UUID, plan_id: UUID) -> bool: ...

    # -------------------------------------------------------------------
    # Approvals
    # -------------------------------------------------------------------

    async def save_approval(
        self, tenant_id: UUID, approval: Approval
    ) -> Approval: ...

    async def get_approval(
        self, tenant_id: UUID, approval_id: UUID
    ) -> Approval | None: ...

    async def list_approvals(
        self,
        tenant_id: UUID,
        state: str | None = None,
        plan_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Approval]: ...

    # -------------------------------------------------------------------
    # Events (append-only audit trail)
    # -------------------------------------------------------------------

    async def save_event(
        self,
        tenant_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        agent_id: UUID | None = None,
        context_id: UUID | None = None,
    ) -> dict[str, Any]: ...

    async def list_events(
        self,
        tenant_id: UUID,
        event_type: str | None = None,
        agent_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...

    async def count_events(
        self,
        tenant_id: UUID,
        event_type: str | None = None,
    ) -> int: ...

    # -------------------------------------------------------------------
    # Agent context (conversation state)
    # -------------------------------------------------------------------

    async def save_context(
        self, tenant_id: UUID, context: AgentContext
    ) -> AgentContext: ...

    async def get_context(
        self, tenant_id: UUID, context_id: UUID
    ) -> AgentContext | None: ...

    async def list_contexts(
        self,
        tenant_id: UUID,
        conversation_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentContext]: ...

    # -------------------------------------------------------------------
    # Risk policies (tenant-scoped)
    # -------------------------------------------------------------------

    async def save_risk_policy(
        self, tenant_id: UUID, policy: RiskPolicy
    ) -> RiskPolicy: ...

    async def get_risk_policy(
        self, tenant_id: UUID, policy_id: UUID
    ) -> RiskPolicy | None: ...

    async def get_active_risk_policy(
        self, tenant_id: UUID
    ) -> RiskPolicy | None: ...

    # -------------------------------------------------------------------
    # Approval gates (tenant-scoped)
    # -------------------------------------------------------------------

    async def save_approval_gate(
        self, tenant_id: UUID, gate: ApprovalGate
    ) -> ApprovalGate: ...

    async def list_approval_gates(
        self, tenant_id: UUID, active_only: bool = True
    ) -> list[ApprovalGate]: ...
