"""Agent archetypes — reusable agent configuration templates.

Provides:
- Archetype model: name, model, system prompt, skills, risk policy
- ArchetypeRegistry: CRUD + built-in defaults
- Per-tenant archetype storage via persistence provider

Usage:
    from aria_core.archetypes import ArchetypeRegistry, Archetype

    registry = ArchetypeRegistry(provider)
    await registry.seed_defaults(tenant_id)
    archetypes = await registry.list(tenant_id)
"""

from aria_core.archetypes.models import Archetype, ArchetypeCategory
from aria_core.archetypes.registry import ArchetypeRegistry

__all__ = ["Archetype", "ArchetypeCategory", "ArchetypeRegistry"]
