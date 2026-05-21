"""Archetype API routes — CRUD + deploy."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser, Role, require_role
from aria_core.archetypes.models import Archetype, ArchetypeCategory
from aria_core.archetypes.registry import ArchetypeRegistry


# Singleton registry (set during app startup)
_registry: ArchetypeRegistry | None = None


def get_registry() -> ArchetypeRegistry:
    global _registry
    if _registry is None:
        _registry = ArchetypeRegistry()
    return _registry


async def list_archetypes(
    user: AuthUser,
    category: str | None = None,
) -> list[dict[str, Any]]:
    registry = get_registry()
    archetypes = await registry.list(user.tenant_id, category=category)
    if not archetypes:
        await registry.seed_defaults(user.tenant_id)
        archetypes = await registry.list(user.tenant_id, category=category)
    return [a.model_dump(mode="json") for a in archetypes]


async def get_archetype(
    archetype_id: UUID,
    user: AuthUser,
) -> dict[str, Any] | None:
    registry = get_registry()
    a = await registry.get(user.tenant_id, archetype_id)
    return a.model_dump(mode="json") if a else None


async def create_archetype(
    data: dict[str, Any],
    user: AuthUser,
) -> dict[str, Any]:
    require_role(user, Role.OPERATOR)
    registry = get_registry()
    archetype = Archetype(
        name=data["name"],
        slug=data.get("slug", data["name"].lower().replace(" ", "-")),
        description=data.get("description", ""),
        category=ArchetypeCategory(data.get("category", "custom")),
        icon=data.get("icon", "⬢"),
        model=data.get("model", "gpt-4"),
        system_prompt=data.get("system_prompt", ""),
        temperature=data.get("temperature", 0.7),
        max_steps=data.get("max_steps", 10),
        allowed_skills=data.get("allowed_skills", []),
        tags=data.get("tags", []),
        created_by=user.user_id,
    )
    saved = await registry.save(user.tenant_id, archetype)
    return saved.model_dump(mode="json")


async def delete_archetype(
    archetype_id: UUID,
    user: AuthUser,
) -> dict[str, Any]:
    require_role(user, Role.OPERATOR)
    registry = get_registry()
    deleted = await registry.delete(user.tenant_id, archetype_id)
    return {"deleted": deleted, "archetype_id": str(archetype_id)}


async def deploy_archetype(
    archetype_id: UUID,
    user: AuthUser,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an agent from an archetype template."""
    require_role(user, Role.OPERATOR)
    registry = get_registry()
    config = await registry.create_from_archetype(
        user.tenant_id, archetype_id, overrides
    )
    return config


async def seed_defaults(user: AuthUser) -> dict[str, Any]:
    """Seed built-in archetypes for the tenant."""
    require_role(user, Role.ADMIN)
    registry = get_registry()
    count = await registry.seed_defaults(user.tenant_id)
    return {"seeded": count, "tenant_id": str(user.tenant_id)}
