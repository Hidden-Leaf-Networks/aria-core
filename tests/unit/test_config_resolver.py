"""Tests for tenant-scoped configuration resolver."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.permissions.models import ApprovalGate, RiskPolicy
from aria_core.persistence.memory import InMemoryProvider
from aria_core.runtime.models import AgentConfig
from aria_core.tenant.config_resolver import ConfigResolver
from aria_core.tenant.models import Tenant, TenantConfig


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def resolver(provider: InMemoryProvider) -> ConfigResolver:
    return ConfigResolver(provider)


class TestOverrideChain:
    async def test_system_defaults_when_no_tenant(
        self, resolver: ConfigResolver
    ) -> None:
        """With no tenant, system defaults are used."""
        config = await resolver.resolve_agent_config(uuid4())
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_steps == 10

    async def test_tenant_overrides_default_model(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        """Tenant config overrides system defaults."""
        tenant = Tenant(
            slug="claude-co",
            name="Claude Co",
            config=TenantConfig(default_model="claude-sonnet-4-20250514"),
        )
        await provider.save_tenant(tenant)

        config = await resolver.resolve_agent_config(tenant.id)
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7  # still system default

    async def test_agent_overrides_tenant(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        """Agent config overrides tenant config."""
        tenant = Tenant(
            slug="default-co",
            name="Default Co",
            config=TenantConfig(default_model="gpt-4"),
        )
        await provider.save_tenant(tenant)

        agent_config = AgentConfig(model="claude-sonnet-4-20250514", temperature=0.3)
        config = await resolver.resolve_agent_config(tenant.id, agent_config)
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.3

    async def test_runtime_overrides_everything(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        """Runtime overrides have highest priority."""
        tenant = Tenant(
            slug="strict-co",
            name="Strict Co",
            config=TenantConfig(default_model="gpt-4"),
        )
        await provider.save_tenant(tenant)

        agent_config = AgentConfig(model="claude-sonnet-4-20250514", temperature=0.3)
        runtime = {"temperature": 0.0, "max_steps": 5}

        config = await resolver.resolve_agent_config(
            tenant.id, agent_config, runtime
        )
        assert config.temperature == 0.0
        assert config.max_steps == 5
        assert config.model == "claude-sonnet-4-20250514"  # from agent

    async def test_tenant_model_restriction_enforced(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        """If tenant restricts models, non-allowed models get replaced."""
        tenant = Tenant(
            slug="locked-co",
            name="Locked Co",
            config=TenantConfig(
                allowed_models=["gpt-4", "gpt-4o"],
            ),
        )
        await provider.save_tenant(tenant)

        # Agent tries to use Claude, but tenant only allows GPT
        agent_config = AgentConfig(model="claude-sonnet-4-20250514")
        config = await resolver.resolve_agent_config(tenant.id, agent_config)
        assert config.model == "gpt-4"  # falls back to first allowed

    async def test_tenant_max_tokens_override(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        tenant = Tenant(
            slug="token-co",
            name="Token Co",
            config=TenantConfig(max_tokens=8192),
        )
        await provider.save_tenant(tenant)

        config = await resolver.resolve_agent_config(tenant.id)
        assert config.max_tokens == 8192


class TestRiskPolicyResolution:
    async def test_resolve_active_policy(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        tenant = Tenant(slug="risk-co", name="Risk Co")
        await provider.save_tenant(tenant)

        policy = RiskPolicy(name="strict", approval_threshold=30, is_active=True)
        await provider.save_risk_policy(tenant.id, policy)

        resolved = await resolver.resolve_risk_policy(tenant.id)
        assert resolved is not None
        assert resolved.approval_threshold == 30

    async def test_no_policy_returns_none(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        tenant = Tenant(slug="no-policy", name="No Policy")
        await provider.save_tenant(tenant)

        resolved = await resolver.resolve_risk_policy(tenant.id)
        assert resolved is None


class TestApprovalGateResolution:
    async def test_resolve_active_gates(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        tenant = Tenant(slug="gate-co", name="Gate Co")
        await provider.save_tenant(tenant)

        g1 = ApprovalGate(name="low", risk_threshold=30)
        g2 = ApprovalGate(name="high", risk_threshold=70)
        g3 = ApprovalGate(name="disabled", risk_threshold=50, is_active=False)
        await provider.save_approval_gate(tenant.id, g1)
        await provider.save_approval_gate(tenant.id, g2)
        await provider.save_approval_gate(tenant.id, g3)

        gates = await resolver.resolve_approval_gates(tenant.id)
        assert len(gates) == 2
        names = {g.name for g in gates}
        assert "disabled" not in names


class TestFeatureFlags:
    async def test_feature_enabled(
        self, provider: InMemoryProvider, resolver: ConfigResolver
    ) -> None:
        tenant = Tenant(
            slug="feature-co",
            name="Feature Co",
            config=TenantConfig(features={"deep_bridge": True, "voice": False}),
        )
        await provider.save_tenant(tenant)

        assert await resolver.is_feature_enabled(tenant.id, "deep_bridge") is True
        assert await resolver.is_feature_enabled(tenant.id, "voice") is False
        assert await resolver.is_feature_enabled(tenant.id, "unknown") is False

    async def test_feature_flags_no_tenant(
        self, resolver: ConfigResolver
    ) -> None:
        """Non-existent tenant defaults to all features off."""
        assert await resolver.is_feature_enabled(uuid4(), "anything") is False
