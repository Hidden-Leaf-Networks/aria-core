"""Tenant data models for multi-tenant white-label deployments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel


class TenantConfig(BaseModel):
    """Tenant-level configuration overrides.

    Override chain: default → tenant → agent → runtime.
    """

    # Branding
    display_name: str | None = None
    logo_url: str | None = None
    theme: dict[str, Any] = Field(default_factory=dict)

    # Model defaults
    default_model: str | None = None
    allowed_models: list[str] = Field(default_factory=list)
    max_tokens: int | None = None

    # Rate limits
    max_concurrent_agents: int = Field(default=10, ge=1, le=1000)
    max_plans_per_hour: int = Field(default=100, ge=1, le=10000)
    max_events_per_day: int = Field(default=100000, ge=1)

    # Feature flags
    features: dict[str, bool] = Field(default_factory=dict)

    # Risk policy overrides
    risk_policy_id: UUID | None = None
    approval_gates: list[dict[str, Any]] = Field(default_factory=list)

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tenant(BaseModel):
    """A tenant — one isolated white-label deployment."""

    id: UUID = Field(default_factory=uuid4)
    slug: str = Field(min_length=1, max_length=63, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
    name: str = Field(min_length=1, max_length=200)
    config: TenantConfig = Field(default_factory=TenantConfig)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TenantContext(BaseModel):
    """Tenant execution context — threaded through all operations.

    Injected at the API/entry layer, validated on every persistence call.
    Immutable once created for a request lifecycle.
    """

    tenant_id: UUID
    tenant_slug: str
    config: TenantConfig = Field(default_factory=TenantConfig)

    model_config = {"frozen": True}

    @classmethod
    def from_tenant(cls, tenant: Tenant) -> TenantContext:
        """Create a context from a tenant record."""
        return cls(
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            config=tenant.config,
        )


# Default tenant for single-tenant / local dev mode
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000000")
DEFAULT_TENANT = Tenant(
    id=DEFAULT_TENANT_ID,
    slug="default",
    name="Default Tenant",
)
DEFAULT_TENANT_CONTEXT = TenantContext(
    tenant_id=DEFAULT_TENANT_ID,
    tenant_slug="default",
)
