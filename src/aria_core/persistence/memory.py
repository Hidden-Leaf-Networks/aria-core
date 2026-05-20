"""In-memory persistence provider.

First-class backend for local development and single-tenant mode.
Refactors the existing dict-based storage from PlanEngine and ApprovalEngine
into the PersistenceProvider protocol.

Thread-safe for asyncio (single-threaded event loop).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from aria_core.permissions.models import Approval, ApprovalGate, RiskPolicy
from aria_core.planning.models import Plan
from aria_core.runtime.models import AgentContext
from aria_core.tenant.models import Tenant, TenantConfig


class TenantNotFoundError(Exception):
    def __init__(self, tenant_id: UUID):
        self.tenant_id = tenant_id
        super().__init__(f"Tenant {tenant_id} not found")


class TenantAccessError(Exception):
    """Raised when an entity's tenant_id doesn't match the requested tenant."""

    def __init__(self, tenant_id: UUID, entity_id: UUID):
        self.tenant_id = tenant_id
        self.entity_id = entity_id
        super().__init__(
            f"Entity {entity_id} does not belong to tenant {tenant_id}"
        )


class InMemoryProvider:
    """In-memory persistence provider implementing PersistenceProvider protocol.

    All data is stored in tenant-keyed dictionaries. Cross-tenant access
    is blocked at the storage layer.
    """

    def __init__(self) -> None:
        # Tenant registry
        self._tenants: dict[UUID, Tenant] = {}
        self._tenants_by_slug: dict[str, UUID] = {}

        # Tenant-scoped storage: {tenant_id: {entity_id: entity}}
        self._plans: dict[UUID, dict[UUID, Plan]] = {}
        self._approvals: dict[UUID, dict[UUID, Approval]] = {}
        self._events: dict[UUID, list[dict[str, Any]]] = {}
        self._contexts: dict[UUID, dict[UUID, AgentContext]] = {}
        self._risk_policies: dict[UUID, dict[UUID, RiskPolicy]] = {}
        self._approval_gates: dict[UUID, dict[UUID, ApprovalGate]] = {}

    def _ensure_tenant_store(self, tenant_id: UUID) -> None:
        """Initialize storage buckets for a tenant if they don't exist."""
        if tenant_id not in self._plans:
            self._plans[tenant_id] = {}
        if tenant_id not in self._approvals:
            self._approvals[tenant_id] = {}
        if tenant_id not in self._events:
            self._events[tenant_id] = []
        if tenant_id not in self._contexts:
            self._contexts[tenant_id] = {}
        if tenant_id not in self._risk_policies:
            self._risk_policies[tenant_id] = {}
        if tenant_id not in self._approval_gates:
            self._approval_gates[tenant_id] = {}

    # -------------------------------------------------------------------
    # Tenant management
    # -------------------------------------------------------------------

    async def save_tenant(self, tenant: Tenant) -> Tenant:
        self._tenants[tenant.id] = tenant
        self._tenants_by_slug[tenant.slug] = tenant.id
        self._ensure_tenant_store(tenant.id)
        return tenant

    async def get_tenant(self, tenant_id: UUID) -> Tenant | None:
        return self._tenants.get(tenant_id)

    async def get_tenant_by_slug(self, slug: str) -> Tenant | None:
        tid = self._tenants_by_slug.get(slug)
        if tid is None:
            return None
        return self._tenants.get(tid)

    async def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        return sorted(tenants, key=lambda t: t.created_at)

    async def update_tenant_config(
        self, tenant_id: UUID, config: TenantConfig
    ) -> Tenant:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)
        updated = tenant.model_copy(
            update={
                "config": config,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._tenants[tenant_id] = updated
        return updated

    # -------------------------------------------------------------------
    # Plans
    # -------------------------------------------------------------------

    async def save_plan(self, tenant_id: UUID, plan: Plan) -> Plan:
        self._ensure_tenant_store(tenant_id)
        self._plans[tenant_id][plan.id] = plan
        return plan

    async def get_plan(self, tenant_id: UUID, plan_id: UUID) -> Plan | None:
        return self._plans.get(tenant_id, {}).get(plan_id)

    async def list_plans(
        self,
        tenant_id: UUID,
        state: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Plan]:
        plans = list(self._plans.get(tenant_id, {}).values())
        if state:
            plans = [p for p in plans if p.state.value == state]
        plans.sort(key=lambda p: p.created_at, reverse=True)
        return plans[offset : offset + limit]

    async def delete_plan(self, tenant_id: UUID, plan_id: UUID) -> bool:
        store = self._plans.get(tenant_id, {})
        if plan_id in store:
            del store[plan_id]
            return True
        return False

    # -------------------------------------------------------------------
    # Approvals
    # -------------------------------------------------------------------

    async def save_approval(
        self, tenant_id: UUID, approval: Approval
    ) -> Approval:
        self._ensure_tenant_store(tenant_id)
        self._approvals[tenant_id][approval.id] = approval
        return approval

    async def get_approval(
        self, tenant_id: UUID, approval_id: UUID
    ) -> Approval | None:
        return self._approvals.get(tenant_id, {}).get(approval_id)

    async def list_approvals(
        self,
        tenant_id: UUID,
        state: str | None = None,
        plan_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Approval]:
        approvals = list(self._approvals.get(tenant_id, {}).values())
        if state:
            approvals = [a for a in approvals if a.state.value == state]
        if plan_id:
            approvals = [a for a in approvals if a.plan_id == plan_id]
        approvals.sort(key=lambda a: a.created_at, reverse=True)
        return approvals[offset : offset + limit]

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
        self._ensure_tenant_store(tenant_id)
        event = {
            "id": uuid4(),
            "tenant_id": tenant_id,
            "event_type": event_type,
            "payload": payload,
            "agent_id": agent_id,
            "context_id": context_id,
            "timestamp": datetime.now(timezone.utc),
        }
        self._events[tenant_id].append(event)
        return event

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
        events = list(self._events.get(tenant_id, []))
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        if agent_id:
            events = [e for e in events if e.get("agent_id") == agent_id]
        if after:
            events = [e for e in events if e["timestamp"] > after]
        if before:
            events = [e for e in events if e["timestamp"] < before]
        events.sort(key=lambda e: e["timestamp"], reverse=True)
        return events[offset : offset + limit]

    async def count_events(
        self,
        tenant_id: UUID,
        event_type: str | None = None,
    ) -> int:
        events = self._events.get(tenant_id, [])
        if event_type:
            return sum(1 for e in events if e["event_type"] == event_type)
        return len(events)

    # -------------------------------------------------------------------
    # Agent context
    # -------------------------------------------------------------------

    async def save_context(
        self, tenant_id: UUID, context: AgentContext
    ) -> AgentContext:
        self._ensure_tenant_store(tenant_id)
        self._contexts[tenant_id][context.id] = context
        return context

    async def get_context(
        self, tenant_id: UUID, context_id: UUID
    ) -> AgentContext | None:
        return self._contexts.get(tenant_id, {}).get(context_id)

    async def list_contexts(
        self,
        tenant_id: UUID,
        conversation_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentContext]:
        contexts = list(self._contexts.get(tenant_id, {}).values())
        if conversation_id:
            contexts = [
                c for c in contexts if c.conversation_id == conversation_id
            ]
        contexts.sort(key=lambda c: c.created_at, reverse=True)
        return contexts[offset : offset + limit]

    # -------------------------------------------------------------------
    # Risk policies
    # -------------------------------------------------------------------

    async def save_risk_policy(
        self, tenant_id: UUID, policy: RiskPolicy
    ) -> RiskPolicy:
        self._ensure_tenant_store(tenant_id)
        self._risk_policies[tenant_id][policy.id] = policy
        return policy

    async def get_risk_policy(
        self, tenant_id: UUID, policy_id: UUID
    ) -> RiskPolicy | None:
        return self._risk_policies.get(tenant_id, {}).get(policy_id)

    async def get_active_risk_policy(
        self, tenant_id: UUID
    ) -> RiskPolicy | None:
        policies = self._risk_policies.get(tenant_id, {}).values()
        active = [p for p in policies if p.is_active]
        return active[0] if active else None

    # -------------------------------------------------------------------
    # Approval gates
    # -------------------------------------------------------------------

    async def save_approval_gate(
        self, tenant_id: UUID, gate: ApprovalGate
    ) -> ApprovalGate:
        self._ensure_tenant_store(tenant_id)
        self._approval_gates[tenant_id][gate.id] = gate
        return gate

    async def list_approval_gates(
        self, tenant_id: UUID, active_only: bool = True
    ) -> list[ApprovalGate]:
        gates = list(self._approval_gates.get(tenant_id, {}).values())
        if active_only:
            gates = [g for g in gates if g.is_active]
        return gates
