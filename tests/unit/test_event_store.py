"""Tests for the event store — durable audit trail with replay."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.persistence.event_store import EventStore
from aria_core.persistence.memory import InMemoryProvider
from aria_core.tenant.models import Tenant


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def tenant() -> Tenant:
    return Tenant(slug="test-co", name="Test Co")


@pytest.fixture
async def store(provider: InMemoryProvider, tenant: Tenant) -> EventStore:
    await provider.save_tenant(tenant)
    return EventStore(provider, tenant.id)


class TestEventStoreEmit:
    async def test_emit_persists_event(self, store: EventStore) -> None:
        event = await store.emit("plan.created", {"plan_id": "abc"})
        assert event["event_type"] == "plan.created"
        assert event["payload"]["plan_id"] == "abc"
        assert "id" in event
        assert "timestamp" in event

    async def test_emit_with_agent_id(self, store: EventStore) -> None:
        agent_id = uuid4()
        event = await store.emit("step.complete", {"step": 1}, agent_id=agent_id)
        assert event["agent_id"] == agent_id

    async def test_emit_notifies_subscribers(self, store: EventStore) -> None:
        received: list[tuple[str, dict]] = []

        async def handler(event_type: str, payload: dict) -> None:
            received.append((event_type, payload))

        store.subscribe(handler)
        await store.emit("test.event", {"val": 42})

        assert len(received) == 1
        assert received[0] == ("test.event", {"val": 42})

    async def test_multiple_subscribers(self, store: EventStore) -> None:
        counts = {"a": 0, "b": 0}

        async def handler_a(et: str, p: dict) -> None:
            counts["a"] += 1

        async def handler_b(et: str, p: dict) -> None:
            counts["b"] += 1

        store.subscribe(handler_a)
        store.subscribe(handler_b)
        await store.emit("test.event", {})

        assert counts["a"] == 1
        assert counts["b"] == 1

    async def test_unsubscribe(self, store: EventStore) -> None:
        calls = 0

        async def handler(et: str, p: dict) -> None:
            nonlocal calls
            calls += 1

        store.subscribe(handler)
        await store.emit("a", {})
        assert calls == 1

        store.unsubscribe(handler)
        await store.emit("b", {})
        assert calls == 1  # not called again

    async def test_sync_handler_supported(self, store: EventStore) -> None:
        received = []

        def sync_handler(et: str, p: dict) -> None:
            received.append(et)

        store.subscribe(sync_handler)
        await store.emit("sync.test", {})
        assert "sync.test" in received


class TestEventStoreReplay:
    async def test_replay_returns_chronological(self, store: EventStore) -> None:
        await store.emit("event.1", {"order": 1})
        await store.emit("event.2", {"order": 2})
        await store.emit("event.3", {"order": 3})

        events = await store.replay()
        assert len(events) == 3
        assert events[0]["payload"]["order"] == 1
        assert events[2]["payload"]["order"] == 3

    async def test_replay_filter_by_type(self, store: EventStore) -> None:
        await store.emit("plan.created", {"id": "p1"})
        await store.emit("agent.start", {"id": "a1"})
        await store.emit("plan.completed", {"id": "p1"})

        plan_events = await store.replay(event_type="plan.created")
        assert len(plan_events) == 1

    async def test_replay_filter_by_agent(self, store: EventStore) -> None:
        agent_1 = uuid4()
        agent_2 = uuid4()
        await store.emit("step", {}, agent_id=agent_1)
        await store.emit("step", {}, agent_id=agent_2)
        await store.emit("step", {}, agent_id=agent_1)

        stream = await store.replay(agent_id=agent_1)
        assert len(stream) == 2

    async def test_replay_with_handler(self, store: EventStore) -> None:
        """Replay through a handler for projection rebuilding."""
        await store.emit("counter.increment", {"delta": 1})
        await store.emit("counter.increment", {"delta": 5})
        await store.emit("counter.increment", {"delta": 3})

        total = 0

        async def accumulator(et: str, payload: dict) -> None:
            nonlocal total
            total += payload.get("delta", 0)

        count = await store.replay_with_handler(
            accumulator, event_type="counter.increment"
        )
        assert count == 3
        assert total == 9


class TestEventStoreStream:
    async def test_get_agent_stream(self, store: EventStore) -> None:
        agent_id = uuid4()
        await store.emit("agent.start", {}, agent_id=agent_id)
        await store.emit("routing.complete", {"intent": "direct"}, agent_id=agent_id)
        await store.emit("responding.complete", {"tokens": 150}, agent_id=agent_id)

        stream = await store.get_stream(agent_id)
        assert len(stream) == 3
        assert stream[0]["event_type"] == "agent.start"
        assert stream[-1]["event_type"] == "responding.complete"


class TestEventStoreCount:
    async def test_count_all(self, store: EventStore) -> None:
        await store.emit("a", {})
        await store.emit("b", {})
        await store.emit("a", {})

        assert await store.count() == 3
        assert await store.count(event_type="a") == 2
        assert await store.count(event_type="b") == 1


class TestEventStoreCallback:
    async def test_as_callback_compatibility(self, store: EventStore) -> None:
        """The as_callback() method returns a legacy-compatible callback."""
        callback = store.as_callback()
        assert callable(callback)
        # Callback is sync — it schedules async work
        # In test context without running loop, it should not crash
        callback("test.event", {"data": "value"})


class TestEventStoreCheckpoint:
    async def test_get_latest_checkpoint(self, store: EventStore) -> None:
        await store.emit("checkpoint.created", {"position": 0})
        await store.emit("plan.created", {})
        await store.emit("checkpoint.created", {"position": 42})

        cp = await store.get_latest_checkpoint()
        assert cp is not None
        assert cp["payload"]["position"] == 42

    async def test_no_checkpoint_returns_none(self, store: EventStore) -> None:
        await store.emit("plan.created", {})
        cp = await store.get_latest_checkpoint()
        assert cp is None
