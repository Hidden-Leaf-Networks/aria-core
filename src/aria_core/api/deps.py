"""FastAPI dependency injection — provider, auth, and tenant context."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser, Role
from aria_core.persistence.memory import InMemoryProvider
from aria_core.tenant.config_resolver import ConfigResolver
from aria_core.tenant.guard import TenantGuard
from aria_core.tenant.models import TenantContext


# ---------------------------------------------------------------------------
# Singleton state (set during app lifespan)
# ---------------------------------------------------------------------------

_provider: Any = None
_guard: TenantGuard | None = None
_resolver: ConfigResolver | None = None


def set_provider(provider: Any) -> None:
    """Set the persistence provider (called during app startup)."""
    global _provider, _guard, _resolver
    _provider = provider
    _guard = TenantGuard(provider)
    _resolver = ConfigResolver(provider)


def get_provider() -> Any:
    """Get the raw persistence provider."""
    if _provider is None:
        raise RuntimeError("Provider not initialized — app not started?")
    return _provider


def get_guard() -> TenantGuard:
    """Get the tenant guard (validates all persistence calls)."""
    if _guard is None:
        raise RuntimeError("Guard not initialized — app not started?")
    return _guard


def get_resolver() -> ConfigResolver:
    """Get the config resolver."""
    if _resolver is None:
        raise RuntimeError("Resolver not initialized — app not started?")
    return _resolver


def get_tenant_context(user: AuthUser) -> TenantContext:
    """Build TenantContext from authenticated user."""
    return TenantContext(
        tenant_id=user.tenant_id,
        tenant_slug=user.tenant_slug or "unknown",
    )
