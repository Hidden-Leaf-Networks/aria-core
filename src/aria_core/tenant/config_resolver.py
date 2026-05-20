"""Tenant-scoped configuration resolver.

Override chain: default config → tenant config → agent config → runtime override.

Usage:
    resolver = ConfigResolver(provider)
    config = await resolver.resolve_agent_config(tenant_id, agent_overrides)
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.permissions.models import ApprovalGate, RiskPolicy
from aria_core.runtime.models import AgentConfig
from aria_core.tenant.models import TenantConfig


# System-wide defaults (hardcoded baseline)
SYSTEM_DEFAULTS = AgentConfig()


class ConfigResolver:
    """Resolves configuration through the override chain.

    Priority (highest wins):
    1. Runtime overrides (per-request)
    2. Agent config (per-agent)
    3. Tenant config (per-tenant)
    4. System defaults (hardcoded)
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def resolve_agent_config(
        self,
        tenant_id: UUID,
        agent_config: AgentConfig | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> AgentConfig:
        """Build final AgentConfig through the full override chain."""
        # Start with system defaults
        result = SYSTEM_DEFAULTS.model_dump()

        # Layer 2: Tenant config overrides
        tenant = await self._provider.get_tenant(tenant_id)
        if tenant and tenant.config:
            tc = tenant.config
            if tc.default_model:
                result["model"] = tc.default_model
            if tc.max_tokens:
                result["max_tokens"] = tc.max_tokens

        # Layer 3: Agent config overrides
        if agent_config:
            agent_dict = agent_config.model_dump(exclude_defaults=True)
            result.update(agent_dict)

        # Layer 4: Runtime overrides (highest priority)
        if runtime_overrides:
            result.update(runtime_overrides)

        # Enforce tenant model restrictions
        if tenant and tenant.config.allowed_models:
            if result["model"] not in tenant.config.allowed_models:
                result["model"] = tenant.config.allowed_models[0]

        return AgentConfig(**result)

    async def resolve_risk_policy(
        self, tenant_id: UUID
    ) -> RiskPolicy | None:
        """Get the active risk policy for a tenant."""
        return await self._provider.get_active_risk_policy(tenant_id)

    async def resolve_approval_gates(
        self, tenant_id: UUID
    ) -> list[ApprovalGate]:
        """Get active approval gates for a tenant."""
        return await self._provider.list_approval_gates(tenant_id, active_only=True)

    async def resolve_tenant_config(
        self, tenant_id: UUID
    ) -> TenantConfig:
        """Get the tenant's configuration."""
        tenant = await self._provider.get_tenant(tenant_id)
        if tenant:
            return tenant.config
        return TenantConfig()

    async def is_feature_enabled(
        self, tenant_id: UUID, feature: str
    ) -> bool:
        """Check if a feature flag is enabled for a tenant."""
        config = await self.resolve_tenant_config(tenant_id)
        return config.features.get(feature, False)
