"""Tests for the WebSocket manager and real-time event streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

from aria_core.api.ws import WebSocketManager


class MockWebSocket:
    """Mock WebSocket for unit testing."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("WebSocket closed")
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True

    @property
    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.sent]


class TestWebSocketManager:
    async def test_connect_and_disconnect(self) -> None:
        manager = WebSocketManager()
        ws = MockWebSocket()
        tid = uuid4()

        await manager.connect(ws, tid)
        assert manager.connection_count(tid) == 1

        await manager.disconnect(ws, tid)
        assert manager.connection_count(tid) == 0

    async def test_broadcast_to_tenant(self) -> None:
        manager = WebSocketManager()
        tid = uuid4()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, tid)
        await manager.connect(ws2, tid)

        sent = await manager.broadcast(tid, "plan.created", {"plan_id": "abc"})
        assert sent == 2

        # Both received the message
        assert len(ws1.messages) == 1
        assert ws1.messages[0]["event_type"] == "plan.created"
        assert ws1.messages[0]["payload"]["plan_id"] == "abc"
        assert len(ws2.messages) == 1

    async def test_broadcast_tenant_isolation(self) -> None:
        manager = WebSocketManager()
        tid_a = uuid4()
        tid_b = uuid4()

        ws_a = MockWebSocket()
        ws_b = MockWebSocket()

        await manager.connect(ws_a, tid_a)
        await manager.connect(ws_b, tid_b)

        await manager.broadcast(tid_a, "secret.event", {"data": "classified"})

        assert len(ws_a.messages) == 1
        assert len(ws_b.messages) == 0  # Tenant B gets nothing

    async def test_broadcast_no_connections(self) -> None:
        manager = WebSocketManager()
        sent = await manager.broadcast(uuid4(), "test", {})
        assert sent == 0

    async def test_dead_connections_cleaned_up(self) -> None:
        manager = WebSocketManager()
        tid = uuid4()

        ws_alive = MockWebSocket()
        ws_dead = MockWebSocket()
        ws_dead.close()  # Simulate disconnected client

        await manager.connect(ws_alive, tid)
        await manager.connect(ws_dead, tid)
        assert manager.connection_count(tid) == 2

        sent = await manager.broadcast(tid, "test", {})
        assert sent == 1  # Only alive one received
        assert manager.connection_count(tid) == 1  # Dead one removed

    async def test_connection_count(self) -> None:
        manager = WebSocketManager()
        tid_a = uuid4()
        tid_b = uuid4()

        await manager.connect(MockWebSocket(), tid_a)
        await manager.connect(MockWebSocket(), tid_a)
        await manager.connect(MockWebSocket(), tid_b)

        assert manager.connection_count(tid_a) == 2
        assert manager.connection_count(tid_b) == 1
        assert manager.connection_count() == 3

    async def test_as_event_handler(self) -> None:
        """WebSocket manager can act as an EventStore subscriber."""
        manager = WebSocketManager()
        tid = uuid4()
        ws = MockWebSocket()
        await manager.connect(ws, tid)

        handler = manager.as_event_handler(tid)
        await handler("agent.complete", {"response": "Done"})

        assert len(ws.messages) == 1
        assert ws.messages[0]["event_type"] == "agent.complete"
        assert ws.messages[0]["payload"]["response"] == "Done"

    async def test_uuid_serialization_in_broadcast(self) -> None:
        manager = WebSocketManager()
        tid = uuid4()
        ws = MockWebSocket()
        await manager.connect(ws, tid)

        agent_id = uuid4()
        await manager.broadcast(tid, "step.complete", {"agent_id": agent_id})

        msg = ws.messages[0]
        assert msg["payload"]["agent_id"] == str(agent_id)
