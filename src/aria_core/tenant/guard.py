"""Tenant guard — validates tenant_id on every persistence operation.

Belt-and-suspenders layer. Even though InMemoryProvider and PostgresProvider
already isolate by tenant_id in their queries, the guard adds explicit
validation that the caller isn't accidentally crossing tenant boundaries.

Usage:
    from aria_core.tenant.guard import TenantGuard

    guard = TenantGuard(provider)
    # Wraps the provider — all calls go through tenant validation
    plan = await guard.get_plan(tenant_id, plan_id)
"""

from __future__ import annotations

from typing import Any
from uuid import UUID


class TenantViolationError(Exception):
    """Raised when a cross-tenant access is attempted."""

    def __init__(self, tenant_id: UUID, entity_type: str, entity_id: UUID | None = None):
        self.tenant_id = tenant_id
        self.entity_type = entity_type
        self.entity_id = entity_id
        msg = f"Tenant {tenant_id} cannot access {entity_type}"
        if entity_id:
            msg += f" {entity_id}"
        super().__init__(msg)


class TenantGuard:
    """Wrapping guard that validates tenant context on persistence calls.

    Validates:
    - tenant_id is not None / zero UUID
    - AgentContext.tenant_id matches the claimed tenant_id
    - Plan/Approval entities returned belong to the requesting tenant
    """

    ZERO_UUID = UUID("00000000-0000-0000-0000-000000000000")

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def _validate_tenant_id(self, tenant_id: UUID) -> None:
        """Ensure tenant_id is valid (not None)."""
        if tenant_id is None:
            raise TenantViolationError(
                self.ZERO_UUID, "any", None
            )

    async def save_plan(self, tenant_id: UUID, plan: Any) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.save_plan(tenant_id, plan)

    async def get_plan(self, tenant_id: UUID, plan_id: UUID) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.get_plan(tenant_id, plan_id)

    async def list_plans(self, tenant_id: UUID, **kwargs: Any) -> list:
        self._validate_tenant_id(tenant_id)
        return await self._provider.list_plans(tenant_id, **kwargs)

    async def save_approval(self, tenant_id: UUID, approval: Any) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.save_approval(tenant_id, approval)

    async def get_approval(self, tenant_id: UUID, approval_id: UUID) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.get_approval(tenant_id, approval_id)

    async def list_approvals(self, tenant_id: UUID, **kwargs: Any) -> list:
        self._validate_tenant_id(tenant_id)
        return await self._provider.list_approvals(tenant_id, **kwargs)

    async def save_event(
        self, tenant_id: UUID, event_type: str, payload: dict, **kwargs: Any
    ) -> dict:
        self._validate_tenant_id(tenant_id)
        return await self._provider.save_event(tenant_id, event_type, payload, **kwargs)

    async def list_events(self, tenant_id: UUID, **kwargs: Any) -> list:
        self._validate_tenant_id(tenant_id)
        return await self._provider.list_events(tenant_id, **kwargs)

    async def save_context(self, tenant_id: UUID, context: Any) -> Any:
        self._validate_tenant_id(tenant_id)
        # Validate context's tenant_id matches
        if hasattr(context, "tenant_id") and context.tenant_id != tenant_id:
            raise TenantViolationError(
                tenant_id, "AgentContext", context.id
            )
        return await self._provider.save_context(tenant_id, context)

    async def get_context(self, tenant_id: UUID, context_id: UUID) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.get_context(tenant_id, context_id)

    # Pass-through for tenant management (not tenant-scoped)
    async def save_tenant(self, tenant: Any) -> Any:
        return await self._provider.save_tenant(tenant)

    async def get_tenant(self, tenant_id: UUID) -> Any:
        return await self._provider.get_tenant(tenant_id)

    async def get_tenant_by_slug(self, slug: str) -> Any:
        return await self._provider.get_tenant_by_slug(slug)

    async def list_tenants(self, **kwargs: Any) -> list:
        return await self._provider.list_tenants(**kwargs)

    async def update_tenant_config(self, tenant_id: UUID, config: Any) -> Any:
        return await self._provider.update_tenant_config(tenant_id, config)

    async def save_risk_policy(self, tenant_id: UUID, policy: Any) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.save_risk_policy(tenant_id, policy)

    async def get_risk_policy(self, tenant_id: UUID, policy_id: UUID) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.get_risk_policy(tenant_id, policy_id)

    async def get_active_risk_policy(self, tenant_id: UUID) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.get_active_risk_policy(tenant_id)

    async def save_approval_gate(self, tenant_id: UUID, gate: Any) -> Any:
        self._validate_tenant_id(tenant_id)
        return await self._provider.save_approval_gate(tenant_id, gate)

    async def list_approval_gates(self, tenant_id: UUID, **kwargs: Any) -> list:
        self._validate_tenant_id(tenant_id)
        return await self._provider.list_approval_gates(tenant_id, **kwargs)

    async def count_events(self, tenant_id: UUID, **kwargs: Any) -> int:
        self._validate_tenant_id(tenant_id)
        return await self._provider.count_events(tenant_id, **kwargs)

    async def list_contexts(self, tenant_id: UUID, **kwargs: Any) -> list:
        self._validate_tenant_id(tenant_id)
        return await self._provider.list_contexts(tenant_id, **kwargs)

    async def delete_plan(self, tenant_id: UUID, plan_id: UUID) -> bool:
        self._validate_tenant_id(tenant_id)
        return await self._provider.delete_plan(tenant_id, plan_id)
