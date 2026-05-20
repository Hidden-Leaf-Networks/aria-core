"""Tenant management API routes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser, Role, require_role
from aria_core.api.deps import get_guard
from aria_core.tenant.models import Tenant, TenantConfig


async def create_tenant(
    data: dict[str, Any],
    user: AuthUser,
) -> dict[str, Any]:
    """Create a new tenant. Admin only."""
    require_role(user, Role.ADMIN)
    guard = get_guard()

    tenant = Tenant(
        slug=data["slug"],
        name=data["name"],
        config=TenantConfig(**data.get("config", {})),
    )
    saved = await guard.save_tenant(tenant)
    return saved.model_dump(mode="json")


async def get_tenant(tenant_id: UUID, user: AuthUser) -> dict[str, Any] | None:
    """Get a tenant by ID. Admin sees any tenant, others see only their own."""
    guard = get_guard()

    if not user.is_admin and user.tenant_id != tenant_id:
        return None

    tenant = await guard.get_tenant(tenant_id)
    return tenant.model_dump(mode="json") if tenant else None


async def list_tenants(user: AuthUser) -> list[dict[str, Any]]:
    """List tenants. Admin sees all, others see only their own."""
    require_role(user, Role.ADMIN)
    guard = get_guard()

    tenants = await guard.list_tenants()
    return [t.model_dump(mode="json") for t in tenants]


async def update_config(
    tenant_id: UUID,
    config_data: dict[str, Any],
    user: AuthUser,
) -> dict[str, Any]:
    """Update tenant configuration. Admin only."""
    require_role(user, Role.ADMIN)
    guard = get_guard()

    config = TenantConfig(**config_data)
    updated = await guard.update_tenant_config(tenant_id, config)
    return updated.model_dump(mode="json")
