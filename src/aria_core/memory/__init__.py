"""Agent Memory — cross-session persistence with Mem0-style architecture.

Provides three tiers of memory:
- **Short-term** — current conversation context (AgentContext.messages)
- **Long-term** — facts, preferences, learned behaviors persisted across sessions
- **Entity** — structured profiles for users, companies, topics with linked memories

Usage:
    from aria_core.memory import MemoryStore, MemoryEntry, EntityProfile

    store = MemoryStore(tenant_id=tid)
    entry = await store.add("user-1", "Prefers dark mode", memory_type="preference")
    results = await store.recall("UI preferences", user_id="user-1")
"""

from aria_core.memory.store import (
    EntityProfile,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    EntityType,
    as_event_callback,
)

__all__ = [
    "EntityProfile",
    "EntityType",
    "MemoryEntry",
    "MemoryStore",
    "MemoryType",
    "as_event_callback",
]
