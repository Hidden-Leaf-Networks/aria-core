"""Event store — durable, append-only audit trail with replay capability.

Replaces the callback-based audit system with a persistent event store
that supports full event sourcing: rebuild state from events.

Usage:
    from aria_core.persistence.event_store import EventStore

    store = EventStore(provider)
    await store.emit(tenant_id, "plan.created", {"plan_id": "..."})

    # Replay events to rebuild state
    events = await store.replay(tenant_id, after=checkpoint)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable
from uuid import UUID


# Type for event subscribers
EventHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class EventStore:
    """Durable event store with subscriber support and replay capability.

    Wraps a PersistenceProvider's event methods with:
    - Subscriber pattern (fan-out to multiple handlers)
    - Event replay for state reconstruction
    - Aggregate projections from event streams
    - Backward-compatible event_callback interface
    """

    def __init__(self, provider: Any, tenant_id: UUID) -> None:
        self._provider = provider
        self._tenant_id = tenant_id
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Register an event handler. Called on every emit."""
        self._subscribers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove an event handler."""
        self._subscribers.remove(handler)

    async def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        agent_id: UUID | None = None,
        context_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Persist an event and notify all subscribers."""
        event = await self._provider.save_event(
            self._tenant_id,
            event_type,
            payload,
            agent_id=agent_id,
            context_id=context_id,
        )

        # Fan out to subscribers
        for handler in self._subscribers:
            result = handler(event_type, payload)
            if result is not None:
                await result

        return event

    async def replay(
        self,
        event_type: str | None = None,
        agent_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Replay events from the store for state reconstruction.

        Returns events in chronological order (oldest first).
        """
        events = await self._provider.list_events(
            self._tenant_id,
            event_type=event_type,
            agent_id=agent_id,
            after=after,
            before=before,
            limit=limit,
        )
        # list_events returns newest-first, replay needs oldest-first
        return list(reversed(events))

    async def replay_with_handler(
        self,
        handler: EventHandler,
        event_type: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> int:
        """Replay events through a handler for projection rebuilding.

        Returns the number of events replayed.
        """
        events = await self.replay(
            event_type=event_type,
            after=after,
            before=before,
        )
        count = 0
        for event in events:
            result = handler(event["event_type"], event["payload"])
            if result is not None:
                await result
            count += 1
        return count

    async def count(self, event_type: str | None = None) -> int:
        """Count events in the store."""
        return await self._provider.count_events(
            self._tenant_id,
            event_type=event_type,
        )

    def as_callback(self) -> Callable[[str, dict[str, Any]], Any]:
        """Return an event_callback compatible with legacy PlanEngine/ApprovalEngine.

        Bridges the old callback interface to the new event store.
        Note: This returns a sync wrapper that schedules the async emit.
        For fully async contexts, use emit() directly.
        """
        import asyncio

        store = self

        def callback(event_type: str, payload: dict[str, Any]) -> None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(store.emit(event_type, payload))
            except RuntimeError:
                # No running loop — skip persistence (test/sync context)
                pass

        return callback

    async def get_stream(
        self,
        agent_id: UUID,
        after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get the event stream for a specific agent (chronological)."""
        return await self.replay(agent_id=agent_id, after=after)

    async def get_latest_checkpoint(
        self,
        event_type: str = "checkpoint.created",
    ) -> dict[str, Any] | None:
        """Get the most recent checkpoint event for resumable replay."""
        events = await self._provider.list_events(
            self._tenant_id,
            event_type=event_type,
            limit=1,
        )
        return events[0] if events else None
