"""Event and context API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from aria_core.api.auth import AuthUser
from aria_core.api.deps import get_guard
from aria_core.persistence.event_store import EventStore


async def list_events(
    user: AuthUser,
    event_type: str | None = None,
    agent_id: UUID | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List events for the authenticated tenant."""
    guard = get_guard()
    events = await guard.list_events(
        user.tenant_id,
        event_type=event_type,
        agent_id=agent_id,
        limit=limit,
        offset=offset,
    )
    # Serialize UUIDs and datetimes for JSON
    return [_serialize_event(e) for e in events]


async def replay_events(
    user: AuthUser,
    event_type: str | None = None,
    agent_id: UUID | None = None,
    limit: int = 10000,
) -> dict[str, Any]:
    """Replay events in chronological order for state reconstruction."""
    guard = get_guard()
    store = EventStore(guard, user.tenant_id)
    events = await store.replay(
        event_type=event_type,
        agent_id=agent_id,
        limit=limit,
    )
    return {
        "count": len(events),
        "events": [_serialize_event(e) for e in events],
    }


async def count_events(
    user: AuthUser,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Count events for the authenticated tenant."""
    guard = get_guard()
    count = await guard.count_events(user.tenant_id, event_type=event_type)
    return {"count": count, "event_type": event_type}


async def get_context(
    context_id: UUID,
    user: AuthUser,
) -> dict[str, Any] | None:
    """Get an agent context by ID."""
    guard = get_guard()
    ctx = await guard.get_context(user.tenant_id, context_id)
    return ctx.model_dump(mode="json") if ctx else None


async def list_contexts(
    user: AuthUser,
    conversation_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List agent contexts for the authenticated tenant."""
    guard = get_guard()
    contexts = await guard.list_contexts(
        user.tenant_id,
        conversation_id=conversation_id,
        limit=limit,
        offset=offset,
    )
    return [c.model_dump(mode="json") for c in contexts]


def _serialize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Serialize event dict for JSON response."""
    result = {}
    for k, v in event.items():
        if isinstance(v, UUID):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result
