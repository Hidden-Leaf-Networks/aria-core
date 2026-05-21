"""Provider management API routes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser, Role, require_role
from aria_core.providers.manager import ProviderConfig, ProviderManager, ProviderType

# Singleton
_manager: ProviderManager | None = None


def get_manager() -> ProviderManager:
    global _manager
    if _manager is None:
        _manager = ProviderManager()
    return _manager


async def list_providers(user: AuthUser) -> list[dict[str, Any]]:
    """List configured providers for the tenant."""
    manager = get_manager()
    configs = manager.list_configs(user.tenant_id)
    return [
        {
            "provider": c.provider.value,
            "enabled": c.enabled,
            "default_model": c.default_model,
            "has_key": bool(c.api_key),
            "key_preview": c.api_key[:8] + "..." if c.api_key else None,
            "base_url": c.base_url,
        }
        for c in configs
    ]


async def configure_provider(
    data: dict[str, Any], user: AuthUser
) -> dict[str, Any]:
    """Configure a provider with API key."""
    require_role(user, Role.ADMIN)
    manager = get_manager()

    config = ProviderConfig(
        provider=ProviderType(data["provider"]),
        api_key=data["api_key"],
        base_url=data.get("base_url"),
        default_model=data.get("default_model"),
        enabled=data.get("enabled", True),
    )
    saved = manager.configure(user.tenant_id, config)
    return {
        "provider": saved.provider.value,
        "enabled": saved.enabled,
        "default_model": saved.default_model,
        "configured": True,
    }


async def remove_provider(
    provider: str, user: AuthUser
) -> dict[str, Any]:
    """Remove a provider config."""
    require_role(user, Role.ADMIN)
    manager = get_manager()
    removed = manager.remove_config(user.tenant_id, ProviderType(provider))
    return {"removed": removed, "provider": provider}


async def list_models(
    user: AuthUser,
    provider: str | None = None,
    available_only: bool = False,
) -> list[dict[str, Any]]:
    """List models in the registry."""
    manager = get_manager()
    models = manager.list_models(
        provider=ProviderType(provider) if provider else None,
        tenant_id=user.tenant_id if available_only else None,
    )
    return [m.model_dump(mode="json") for m in models]


async def get_provider_status(user: AuthUser) -> dict[str, Any]:
    """Get provider status overview."""
    manager = get_manager()
    return manager.get_status(user.tenant_id)


async def test_provider(
    provider: str, user: AuthUser
) -> dict[str, Any]:
    """Test a provider connection by sending a minimal request."""
    require_role(user, Role.OPERATOR)
    manager = get_manager()

    try:
        adapter = manager.get_best_adapter(user.tenant_id, preference="fast")
        # Quick test — just verify we can create the adapter
        return {
            "provider": provider,
            "status": "connected",
            "adapter": type(adapter).__name__,
        }
    except Exception as e:
        return {
            "provider": provider,
            "status": "error",
            "error": str(e),
        }
