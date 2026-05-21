"""Tests for agent memory — cross-session persistence (ARIA-301)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from aria_core.memory.store import (
    EntityProfile,
    EntityType,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    as_event_callback,
)


TENANT = uuid4()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seeded_store() -> MemoryStore:
    """Return a MemoryStore pre-loaded with diverse memories."""
    store = MemoryStore(tenant_id=TENANT)
    await store.add("u1", "User prefers dark mode", memory_type="preference", importance=0.8)
    await store.add("u1", "User works at Acme Corp", memory_type="fact", importance=0.7)
    await store.add("u1", "User asked about billing on Monday", memory_type="episode", importance=0.4)
    await store.add("u2", "User speaks French", memory_type="fact", importance=0.6)
    return store


# ---------------------------------------------------------------------------
# MemoryEntry model
# ---------------------------------------------------------------------------


class TestMemoryEntryModel:
    def test_defaults(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.importance == 0.5
        assert entry.access_count == 0
        assert entry.memory_type == MemoryType.FACT
        assert entry.expires_at is None

    def test_memory_type_enum(self) -> None:
        for mt in ("fact", "preference", "behavior", "entity", "episode"):
            assert MemoryType(mt).value == mt

    def test_entity_type_enum(self) -> None:
        for et in ("user", "company", "topic", "agent"):
            assert EntityType(et).value == et


# ---------------------------------------------------------------------------
# EntityProfile model
# ---------------------------------------------------------------------------


class TestEntityProfileModel:
    def test_defaults(self) -> None:
        ep = EntityProfile(name="Acme")
        assert ep.entity_type == EntityType.USER
        assert ep.attributes == {}
        assert ep.memory_ids == []


# ---------------------------------------------------------------------------
# MemoryStore — add / recall / forget
# ---------------------------------------------------------------------------


class TestMemoryStoreBasic:
    async def test_add_returns_entry(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "likes sushi", memory_type="preference")
        assert isinstance(entry, MemoryEntry)
        assert entry.tenant_id == TENANT
        assert entry.user_id == "u1"
        assert entry.content == "likes sushi"
        assert entry.memory_type == MemoryType.PREFERENCE
        assert len(entry.embedding) > 0

    async def test_add_with_metadata(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "test", metadata={"src": "chat"})
        assert entry.metadata["src"] == "chat"

    async def test_recall_empty_store(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        results = await store.recall("anything")
        assert results == []

    async def test_recall_returns_relevant(self) -> None:
        store = await _seeded_store()
        results = await store.recall("dark mode UI preferences", user_id="u1")
        assert len(results) >= 1
        # The preference about dark mode should rank highly
        contents = [r.content for r in results]
        assert any("dark mode" in c for c in contents)

    async def test_recall_filters_by_user(self) -> None:
        store = await _seeded_store()
        results = await store.recall("language", user_id="u2")
        assert all(r.user_id == "u2" for r in results)

    async def test_recall_filters_by_type(self) -> None:
        store = await _seeded_store()
        results = await store.recall("user info", memory_type="preference")
        assert all(r.memory_type == MemoryType.PREFERENCE for r in results)

    async def test_recall_min_importance(self) -> None:
        store = await _seeded_store()
        results = await store.recall("billing", user_id="u1", min_importance=0.5)
        assert all(r.importance >= 0.5 for r in results)

    async def test_recall_increments_access_count(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "test fact")
        assert entry.access_count == 0
        await store.recall("test")
        assert entry.access_count == 1
        await store.recall("test")
        assert entry.access_count == 2

    async def test_recall_top_k(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        for i in range(20):
            await store.add("u1", f"memory number {i}")
        results = await store.recall("memory", top_k=5)
        assert len(results) == 5

    async def test_forget_existing(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "to be deleted")
        assert await store.forget(entry.id) is True
        assert store.stats()["memory_count"] == 0

    async def test_forget_nonexistent(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        assert await store.forget(uuid4()) is False

    async def test_expired_memories_excluded(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "ephemeral note")
        # Manually expire
        entry.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        results = await store.recall("ephemeral")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


class TestDecay:
    async def test_decay_reduces_importance(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "test", importance=1.0)
        store.decay(factor=0.5)
        assert entry.importance == pytest.approx(0.5)

    async def test_decay_never_below_zero(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "test", importance=0.01)
        for _ in range(100):
            store.decay(factor=0.5)
        assert entry.importance >= 0.0


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


class TestConsolidate:
    async def test_consolidate_deduplicates(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add("u1", "user prefers dark mode", importance=0.9)
        await store.add("u1", "user prefers dark mode", importance=0.3)
        removed = await store.consolidate("u1", similarity_threshold=0.9)
        assert removed >= 1
        assert store.stats()["memory_count"] == 1

    async def test_consolidate_keeps_higher_importance(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add("u1", "user prefers dark mode", importance=0.9)
        await store.add("u1", "user prefers dark mode", importance=0.3)
        await store.consolidate("u1", similarity_threshold=0.9)
        remaining = list(store._memories.values())
        assert remaining[0].importance == pytest.approx(0.9)

    async def test_consolidate_no_cross_user(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add("u1", "identical text", importance=0.5)
        await store.add("u2", "identical text", importance=0.5)
        removed = await store.consolidate("u1")
        assert removed == 0
        assert store.stats()["memory_count"] == 2

    async def test_consolidate_dissimilar_kept(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add("u1", "python machine learning AI", importance=0.5)
        await store.add("u1", "cooking pasta recipes italian food", importance=0.5)
        removed = await store.consolidate("u1", similarity_threshold=0.9)
        assert removed == 0


# ---------------------------------------------------------------------------
# Context Injection
# ---------------------------------------------------------------------------


class TestContextInjection:
    async def test_basic_injection(self) -> None:
        store = await _seeded_store()
        ctx = await store.get_context_injection("u1", "preferences")
        assert ctx.startswith("[Agent Memory]")
        assert "dark mode" in ctx

    async def test_empty_injection(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        ctx = await store.get_context_injection("u1", "anything")
        assert ctx == ""

    async def test_max_tokens_respected(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        for i in range(50):
            await store.add("u1", f"Memory entry number {i} with some padding text")
        ctx = await store.get_context_injection("u1", "memory", max_tokens=200)
        assert len(ctx) <= 200


# ---------------------------------------------------------------------------
# Entity Management
# ---------------------------------------------------------------------------


class TestEntityManagement:
    async def test_add_entity(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entity = await store.add_entity("Acme Corp", entity_type="company", attributes={"industry": "tech"})
        assert isinstance(entity, EntityProfile)
        assert entity.name == "Acme Corp"
        assert entity.entity_type == EntityType.COMPANY
        assert entity.attributes["industry"] == "tech"

    async def test_add_entity_update_existing(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add_entity("Acme", attributes={"a": 1})
        updated = await store.add_entity("Acme", attributes={"b": 2})
        assert updated.attributes == {"a": 1, "b": 2}

    async def test_get_entity(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add_entity("Acme")
        assert store.get_entity("Acme") is not None
        assert store.get_entity("Missing") is None

    async def test_link_memory_to_entity(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "Acme ships fast")
        await store.add_entity("Acme")
        assert await store.link_memory_to_entity(entry.id, "Acme") is True

    async def test_link_nonexistent_entity(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "test")
        assert await store.link_memory_to_entity(entry.id, "NoSuchEntity") is False

    async def test_link_nonexistent_memory(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        await store.add_entity("Acme")
        assert await store.link_memory_to_entity(uuid4(), "Acme") is False

    async def test_get_entity_memories(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        e1 = await store.add("u1", "Acme ships fast")
        e2 = await store.add("u1", "Acme has 100 employees")
        await store.add_entity("Acme")
        await store.link_memory_to_entity(e1.id, "Acme")
        await store.link_memory_to_entity(e2.id, "Acme")
        memories = store.get_entity_memories("Acme")
        assert len(memories) == 2

    async def test_get_entity_memories_empty(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        assert store.get_entity_memories("Missing") == []

    async def test_forget_unlinks_from_entity(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        entry = await store.add("u1", "linked fact")
        await store.add_entity("Acme")
        await store.link_memory_to_entity(entry.id, "Acme")
        await store.forget(entry.id)
        assert store.get_entity_memories("Acme") == []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_stats_empty(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        s = store.stats()
        assert s["memory_count"] == 0
        assert s["entity_count"] == 0
        assert s["avg_importance"] == 0.0

    async def test_stats_populated(self) -> None:
        store = await _seeded_store()
        await store.add_entity("Acme")
        s = store.stats()
        assert s["memory_count"] == 4
        assert s["entity_count"] == 1
        assert s["avg_importance"] > 0
        assert "fact" in s["type_counts"]


# ---------------------------------------------------------------------------
# Event Callback
# ---------------------------------------------------------------------------


class TestEventCallback:
    async def test_callback_stores_episode(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        callback = as_event_callback(store, user_id="u1")
        await callback({
            "type": "agent.complete",
            "result": {"content": "The weather in Detroit is 72F and sunny today."},
        })
        assert store.stats()["memory_count"] == 1
        mem = list(store._memories.values())[0]
        assert mem.memory_type == MemoryType.EPISODE
        assert mem.metadata["auto_extracted"] is True

    async def test_callback_ignores_non_complete(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        callback = as_event_callback(store, user_id="u1")
        await callback({"type": "agent.start", "result": {"content": "something"}})
        assert store.stats()["memory_count"] == 0

    async def test_callback_ignores_short_content(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        callback = as_event_callback(store, user_id="u1")
        await callback({"type": "agent.complete", "result": {"content": "ok"}})
        assert store.stats()["memory_count"] == 0

    async def test_callback_truncates_long_content(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        callback = as_event_callback(store, user_id="u1")
        long_text = "A" * 1000
        await callback({"type": "agent.complete", "result": {"content": long_text}})
        mem = list(store._memories.values())[0]
        assert len(mem.content) == 500

    async def test_callback_string_result(self) -> None:
        store = MemoryStore(tenant_id=TENANT)
        callback = as_event_callback(store, user_id="u1")
        await callback({
            "type": "agent.complete",
            "result": "This is a plain text result from the agent.",
        })
        assert store.stats()["memory_count"] == 1
