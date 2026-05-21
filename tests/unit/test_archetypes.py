"""Tests for archetype registry."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.archetypes.models import Archetype, ArchetypeCategory
from aria_core.archetypes.registry import ArchetypeRegistry, BUILTIN_ARCHETYPES


class TestArchetypeRegistry:
    async def test_seed_defaults(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        count = await registry.seed_defaults(tid)
        assert count == len(BUILTIN_ARCHETYPES)

        # Seeding again should add 0
        again = await registry.seed_defaults(tid)
        assert again == 0

    async def test_list_archetypes(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        await registry.seed_defaults(tid)

        all_archetypes = await registry.list(tid)
        assert len(all_archetypes) == len(BUILTIN_ARCHETYPES)

    async def test_list_by_category(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        await registry.seed_defaults(tid)

        research = await registry.list(tid, category="research")
        assert len(research) >= 1
        assert all(a.category == ArchetypeCategory.RESEARCH for a in research)

    async def test_save_custom_archetype(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()

        custom = Archetype(
            name="Custom Bot",
            slug="custom-bot",
            description="My custom agent",
            category=ArchetypeCategory.CUSTOM,
            model="gpt-4o",
        )
        saved = await registry.save(tid, custom)
        assert saved.tenant_id == tid

        retrieved = await registry.get(tid, saved.id)
        assert retrieved is not None
        assert retrieved.name == "Custom Bot"

    async def test_get_by_slug(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        await registry.seed_defaults(tid)

        analyst = await registry.get_by_slug(tid, "research-analyst")
        assert analyst is not None
        assert analyst.name == "Research Analyst"

    async def test_delete_archetype(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        await registry.seed_defaults(tid)

        archetypes = await registry.list(tid)
        first_id = archetypes[0].id

        deleted = await registry.delete(tid, first_id)
        assert deleted is True

        gone = await registry.get(tid, first_id)
        assert gone is None

    async def test_tenant_isolation(self) -> None:
        registry = ArchetypeRegistry()
        t1, t2 = uuid4(), uuid4()
        await registry.seed_defaults(t1)

        t2_list = await registry.list(t2)
        assert len(t2_list) == 0

    async def test_create_from_archetype(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        await registry.seed_defaults(tid)

        analyst = await registry.get_by_slug(tid, "research-analyst")
        assert analyst is not None

        config = await registry.create_from_archetype(tid, analyst.id)
        assert config["name"] == "Research Analyst"
        assert config["model"] == "claude-sonnet-4-20250514"
        assert "web_search" in config["allowed_skills"]

    async def test_create_from_archetype_with_overrides(self) -> None:
        registry = ArchetypeRegistry()
        tid = uuid4()
        await registry.seed_defaults(tid)

        analyst = await registry.get_by_slug(tid, "code-assistant")
        assert analyst is not None

        config = await registry.create_from_archetype(
            tid, analyst.id, overrides={"model": "gpt-4o", "temperature": 0.0}
        )
        assert config["model"] == "gpt-4o"
        assert config["temperature"] == 0.0

    async def test_builtin_archetypes_have_correct_fields(self) -> None:
        for a in BUILTIN_ARCHETYPES:
            assert a.is_builtin is True
            assert len(a.name) > 0
            assert len(a.slug) > 0
            assert len(a.system_prompt) > 0
            assert len(a.allowed_skills) > 0
            assert len(a.tags) > 0
