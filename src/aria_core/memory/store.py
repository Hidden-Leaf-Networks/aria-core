"""MemoryStore — Mem0-style cross-session agent memory.

In-memory implementation with cosine-similarity retrieval.
For production, swap the backing store with pgvector, Redis, etc.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.knowledge.base import SimpleEmbedding, EmbeddingProvider, _cosine_similarity
from aria_core.runtime.models import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    BEHAVIOR = "behavior"
    ENTITY = "entity"
    EPISODE = "episode"


class EntityType(StrEnum):
    USER = "user"
    COMPANY = "company"
    TOPIC = "topic"
    AGENT = "agent"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """A single persisted memory."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID = Field(default_factory=uuid4)
    agent_id: UUID | None = None
    user_id: str = ""
    session_id: str | None = None
    memory_type: MemoryType = MemoryType.FACT
    content: str = ""
    embedding: list[float] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None


class EntityProfile(BaseModel):
    """Structured profile for a user, company, topic, or agent."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType = EntityType.USER
    name: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    memory_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """Tenant-scoped memory store with semantic recall and decay.

    In-memory implementation. For production, back with pgvector / Redis.
    """

    def __init__(
        self,
        tenant_id: UUID,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self._embedder = embedding_provider or SimpleEmbedding()
        self._memories: dict[UUID, MemoryEntry] = {}
        self._entities: dict[str, EntityProfile] = {}  # keyed by name

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def add(
        self,
        user_id: str,
        content: str,
        memory_type: str | MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        *,
        agent_id: UUID | None = None,
        session_id: str | None = None,
    ) -> MemoryEntry:
        """Embed and store a new memory."""
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        embedding = (await self._embedder.embed([content]))[0]

        entry = MemoryEntry(
            tenant_id=self.tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {},
        )
        self._memories[entry.id] = entry
        return entry

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        memory_type: str | MemoryType | None = None,
        top_k: int = 10,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Semantic search over memories.

        Returns up to *top_k* entries ranked by cosine similarity,
        optionally filtered by user_id, memory_type, and min_importance.
        """
        if not self._memories:
            return []

        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        query_embedding = (await self._embedder.embed([query]))[0]

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._memories.values():
            # Filters
            if user_id is not None and entry.user_id != user_id:
                continue
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            if entry.importance < min_importance:
                continue
            if entry.expires_at is not None and entry.expires_at < now:
                continue
            if not entry.embedding:
                continue

            score = _cosine_similarity(query_embedding, entry.embedding)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[MemoryEntry] = []
        for _score, entry in scored[:top_k]:
            # Update access metadata
            entry.access_count += 1
            entry.last_accessed_at = now
            results.append(entry)

        return results

    async def forget(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        if memory_id in self._memories:
            # Also unlink from any entities
            for entity in self._entities.values():
                if memory_id in entity.memory_ids:
                    entity.memory_ids.remove(memory_id)
            del self._memories[memory_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Decay & Consolidation
    # ------------------------------------------------------------------

    def decay(self, factor: float = 0.95) -> None:
        """Reduce importance of all memories by *factor* (simulates forgetting)."""
        for entry in self._memories.values():
            entry.importance = max(0.0, entry.importance * factor)

    async def consolidate(self, user_id: str, similarity_threshold: float = 0.92) -> int:
        """Merge similar memories for *user_id* (dedup by high cosine similarity).

        When two memories are very similar, the one with higher importance
        is kept and the other is deleted. Returns the number of memories removed.
        """
        user_memories = [
            m for m in self._memories.values() if m.user_id == user_id
        ]
        if len(user_memories) < 2:
            return 0

        to_remove: set[UUID] = set()
        for i, a in enumerate(user_memories):
            if a.id in to_remove:
                continue
            for b in user_memories[i + 1 :]:
                if b.id in to_remove:
                    continue
                if not a.embedding or not b.embedding:
                    continue
                sim = _cosine_similarity(a.embedding, b.embedding)
                if sim >= similarity_threshold:
                    # Keep the more important one
                    loser = b if a.importance >= b.importance else a
                    to_remove.add(loser.id)

        for mid in to_remove:
            await self.forget(mid)

        return len(to_remove)

    # ------------------------------------------------------------------
    # Context Injection
    # ------------------------------------------------------------------

    async def get_context_injection(
        self,
        user_id: str,
        query: str,
        max_tokens: int = 500,
    ) -> str:
        """Format recalled memories as a context string for LLM injection.

        Returns a human-readable block of relevant memories, truncated
        to approximately *max_tokens* characters.
        """
        entries = await self.recall(query, user_id=user_id, top_k=10)
        if not entries:
            return ""

        lines: list[str] = ["[Agent Memory]"]
        total_len = len(lines[0])
        for entry in entries:
            line = f"- [{entry.memory_type.value}] {entry.content}"
            if total_len + len(line) + 1 > max_tokens:
                break
            lines.append(line)
            total_len += len(line) + 1

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Entity Management
    # ------------------------------------------------------------------

    async def add_entity(
        self,
        name: str,
        entity_type: str | EntityType = EntityType.USER,
        attributes: dict[str, Any] | None = None,
    ) -> EntityProfile:
        """Create or update an entity profile."""
        if isinstance(entity_type, str):
            entity_type = EntityType(entity_type)

        if name in self._entities:
            existing = self._entities[name]
            existing.attributes.update(attributes or {})
            existing.updated_at = datetime.now(timezone.utc)
            return existing

        profile = EntityProfile(
            tenant_id=self.tenant_id,
            entity_type=entity_type,
            name=name,
            attributes=attributes or {},
        )
        self._entities[name] = profile
        return profile

    def get_entity(self, name: str) -> EntityProfile | None:
        """Retrieve an entity profile by name."""
        return self._entities.get(name)

    async def link_memory_to_entity(self, memory_id: UUID, entity_name: str) -> bool:
        """Link a memory to an entity profile."""
        entity = self._entities.get(entity_name)
        if entity is None or memory_id not in self._memories:
            return False
        if memory_id not in entity.memory_ids:
            entity.memory_ids.append(memory_id)
            entity.updated_at = datetime.now(timezone.utc)
        return True

    def get_entity_memories(self, entity_name: str) -> list[MemoryEntry]:
        """Get all memories linked to an entity."""
        entity = self._entities.get(entity_name)
        if entity is None:
            return []
        return [
            self._memories[mid]
            for mid in entity.memory_ids
            if mid in self._memories
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the memory store."""
        memories = list(self._memories.values())
        avg_importance = (
            sum(m.importance for m in memories) / len(memories)
            if memories
            else 0.0
        )
        type_counts: dict[str, int] = {}
        for m in memories:
            key = m.memory_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "memory_count": len(memories),
            "entity_count": len(self._entities),
            "avg_importance": round(avg_importance, 4),
            "type_counts": type_counts,
        }


# ---------------------------------------------------------------------------
# Event callback for automatic memory extraction
# ---------------------------------------------------------------------------


def as_event_callback(
    store: MemoryStore,
    user_id: str,
    *,
    importance: float = 0.5,
) -> Callable:
    """Return an async callback that extracts facts from agent execution events.

    Intended to be wired into the agent lifecycle:

        store = MemoryStore(tenant_id=tid)
        callback = as_event_callback(store, user_id="user-1")
        # agent.on("complete", callback)

    The callback inspects the event for assistant messages and stores
    each one as an ``episode`` memory.
    """

    async def _on_event(event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type != "agent.complete":
            return

        # Extract content from the event payload
        result = event.get("result", {})
        content: str = ""
        if isinstance(result, dict):
            content = result.get("content", "") or result.get("text", "")
        elif isinstance(result, str):
            content = result

        if not content or len(content) < 10:
            return

        # Truncate very long responses to a summary-sized excerpt
        if len(content) > 500:
            content = content[:497] + "..."

        await store.add(
            user_id=user_id,
            content=content,
            memory_type=MemoryType.EPISODE,
            importance=importance,
            metadata={"source": "agent.complete", "auto_extracted": True},
        )

    return _on_event
