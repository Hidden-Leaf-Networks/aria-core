"""Tests for tenant models and context."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from aria_core.tenant.models import (
    DEFAULT_TENANT,
    DEFAULT_TENANT_CONTEXT,
    DEFAULT_TENANT_ID,
    Tenant,
    TenantConfig,
    TenantContext,
)


class TestTenantConfig:
    def test_default_config(self) -> None:
        config = TenantConfig()
        assert config.max_concurrent_agents == 10
        assert config.max_plans_per_hour == 100
        assert config.features == {}
        assert config.default_model is None

    def test_custom_config(self) -> None:
        config = TenantConfig(
            display_name="Acme Corp",
            default_model="claude-sonnet-4-20250514",
            allowed_models=["claude-sonnet-4-20250514", "gpt-4"],
            max_concurrent_agents=50,
            features={"deep_bridge": True, "voice": False},
        )
        assert config.display_name == "Acme Corp"
        assert config.default_model == "claude-sonnet-4-20250514"
        assert len(config.allowed_models) == 2
        assert config.features["deep_bridge"] is True

    def test_rate_limit_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TenantConfig(max_concurrent_agents=0)
        with pytest.raises(ValidationError):
            TenantConfig(max_concurrent_agents=1001)


class TestTenant:
    def test_create_tenant(self) -> None:
        tenant = Tenant(slug="acme-corp", name="Acme Corporation")
        assert tenant.is_active is True
        assert tenant.slug == "acme-corp"
        assert tenant.config.max_concurrent_agents == 10

    def test_slug_validation(self) -> None:
        # Valid slugs
        Tenant(slug="ab", name="Test")
        Tenant(slug="my-tenant-01", name="Test")

        # Invalid: single char (pattern requires start + end chars)
        with pytest.raises(ValidationError):
            Tenant(slug="a", name="Test")

        # Invalid: uppercase
        with pytest.raises(ValidationError):
            Tenant(slug="UPPER", name="Test")

        # Invalid: starts with hyphen
        with pytest.raises(ValidationError):
            Tenant(slug="-bad", name="Test")

    def test_tenant_with_config(self) -> None:
        config = TenantConfig(default_model="gpt-4", features={"planning": True})
        tenant = Tenant(slug="test-co", name="Test Co", config=config)
        assert tenant.config.default_model == "gpt-4"
        assert tenant.config.features["planning"] is True


class TestTenantContext:
    def test_from_tenant(self) -> None:
        tenant = Tenant(slug="acme-co", name="Acme")
        ctx = TenantContext.from_tenant(tenant)
        assert ctx.tenant_id == tenant.id
        assert ctx.tenant_slug == "acme-co"

    def test_context_is_frozen(self) -> None:
        ctx = TenantContext(tenant_id=uuid4(), tenant_slug="test-co")
        with pytest.raises(ValidationError):
            ctx.tenant_id = uuid4()  # type: ignore[misc]

    def test_default_tenant_exists(self) -> None:
        assert DEFAULT_TENANT.slug == "default"
        assert DEFAULT_TENANT.id == DEFAULT_TENANT_ID
        assert DEFAULT_TENANT_CONTEXT.tenant_id == DEFAULT_TENANT_ID


class TestAgentContextTenantId:
    def test_default_tenant_id(self) -> None:
        from aria_core.runtime.models import AgentContext

        ctx = AgentContext()
        assert ctx.tenant_id == DEFAULT_TENANT_ID

    def test_explicit_tenant_id(self) -> None:
        from aria_core.runtime.models import AgentContext

        tid = uuid4()
        ctx = AgentContext(tenant_id=tid)
        assert ctx.tenant_id == tid

    def test_tenant_id_survives_copy(self) -> None:
        from aria_core.runtime.models import AgentContext

        tid = uuid4()
        ctx = AgentContext(tenant_id=tid)
        copied = ctx.model_copy(update={"step_count": 5})
        assert copied.tenant_id == tid

    def test_tenant_id_in_serialization(self) -> None:
        from aria_core.runtime.models import AgentContext

        tid = uuid4()
        ctx = AgentContext(tenant_id=tid)
        data = ctx.model_dump()
        assert data["tenant_id"] == tid
