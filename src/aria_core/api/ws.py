"""WebSocket manager for real-time agent execution streaming.

Provides tenant-scoped channels with event fan-out to connected clients.

Usage (in app factory):
    ws_manager = WebSocketManager()

    @app.websocket("/ws/events")
    async def ws_events(websocket):
        await ws_manager.handle_connection(websocket, tenant_id)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import UUID


class WebSocketManager:
    """Manages tenant-scoped WebSocket connections.

    Each tenant has its own channel. Events emitted to a tenant's
    channel are broadcast to all connected clients for that tenant.
    """

    def __init__(self) -> None:
        # {tenant_id: set of websocket connections}
        self._channels: dict[UUID, set[Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: Any, tenant_id: UUID) -> None:
        """Register a WebSocket connection for a tenant."""
        async with self._lock:
            if tenant_id not in self._channels:
                self._channels[tenant_id] = set()
            self._channels[tenant_id].add(websocket)

    async def disconnect(self, websocket: Any, tenant_id: UUID) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if tenant_id in self._channels:
                self._channels[tenant_id].discard(websocket)
                if not self._channels[tenant_id]:
                    del self._channels[tenant_id]

    async def broadcast(
        self, tenant_id: UUID, event_type: str, payload: dict[str, Any]
    ) -> int:
        """Broadcast an event to all connections in a tenant's channel.

        Returns the number of clients that received the message.
        """
        connections = self._channels.get(tenant_id, set()).copy()
        if not connections:
            return 0

        message = json.dumps(
            {
                "event_type": event_type,
                "payload": _serialize_payload(payload),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

        sent = 0
        dead: list[Any] = []
        for ws in connections:
            try:
                await ws.send_text(message)
                sent += 1
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        if dead:
            async with self._lock:
                channel = self._channels.get(tenant_id)
                if channel:
                    for ws in dead:
                        channel.discard(ws)

        return sent

    def connection_count(self, tenant_id: UUID | None = None) -> int:
        """Get the number of active connections."""
        if tenant_id:
            return len(self._channels.get(tenant_id, set()))
        return sum(len(conns) for conns in self._channels.values())

    def as_event_handler(self, tenant_id: UUID) -> Any:
        """Return an EventStore-compatible handler that broadcasts events."""
        manager = self

        async def handler(event_type: str, payload: dict[str, Any]) -> None:
            await manager.broadcast(tenant_id, event_type, payload)

        return handler


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Serialize UUIDs and datetimes for JSON."""
    result = {}
    for k, v in payload.items():
        if isinstance(v, UUID):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _serialize_payload(v)
        else:
            result[k] = v
    return result
