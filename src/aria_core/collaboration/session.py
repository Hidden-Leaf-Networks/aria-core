"""Collaboration session — presence, locking, comments, and broadcast."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from aria_core.runtime.models import BaseModel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Cursor(BaseModel):
    """A user's cursor position on the canvas."""

    user_id: UUID
    position: tuple[float, float]
    color: str = "#3b82f6"
    timestamp: float = 0.0


class Presence(BaseModel):
    """A user's presence in a collaboration session."""

    user_id: UUID
    tenant_id: UUID
    page: str
    status: Literal["active", "idle", "away"] = "active"
    cursor: Optional[Cursor] = None
    connected_at: datetime


class NodeLock(BaseModel):
    """An exclusive edit lock on a workflow node."""

    node_id: str
    locked_by: UUID
    locked_at: datetime
    expires_at: datetime


class Comment(BaseModel):
    """A comment attached to a node or edge."""

    id: UUID
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    user_id: UUID
    text: str
    created_at: datetime
    resolved: bool = False


class BroadcastEvent(BaseModel):
    """An event queued for delivery to session participants."""

    event_type: str
    payload: dict[str, Any]
    timestamp: float


# ---------------------------------------------------------------------------
# CollaborationSession
# ---------------------------------------------------------------------------

class CollaborationSession:
    """Manages presence, locks, comments, and broadcasts for one session."""

    def __init__(self, session_id: UUID, tenant_id: UUID) -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id

        self._presence: dict[UUID, Presence] = {}
        self._locks: dict[str, NodeLock] = {}
        self._comments: list[Comment] = []
        self._pending: dict[UUID, list[BroadcastEvent]] = defaultdict(list)

    # -- Presence -----------------------------------------------------------

    def join(self, user_id: UUID, page: str) -> Presence:
        """Add a user to the session."""
        presence = Presence(
            user_id=user_id,
            tenant_id=self.tenant_id,
            page=page,
            status="active",
            connected_at=datetime.now(timezone.utc),
        )
        self._presence[user_id] = presence
        self.broadcast("user_joined", {"user_id": str(user_id), "page": page})
        return presence

    def leave(self, user_id: UUID) -> None:
        """Remove a user from the session."""
        self._presence.pop(user_id, None)
        # Release any locks held by this user.
        to_release = [nid for nid, lk in self._locks.items() if lk.locked_by == user_id]
        for nid in to_release:
            del self._locks[nid]
        self.broadcast("user_left", {"user_id": str(user_id)})

    def update_cursor(self, user_id: UUID, x: float, y: float) -> Cursor:
        """Update a user's cursor position."""
        if user_id not in self._presence:
            raise ValueError(f"User {user_id} is not in the session")
        cursor = Cursor(user_id=user_id, position=(x, y), timestamp=time.time())
        self._presence[user_id].cursor = cursor
        self.broadcast("cursor_moved", {"user_id": str(user_id), "x": x, "y": y})
        return cursor

    def update_status(self, user_id: UUID, status: Literal["active", "idle", "away"]) -> None:
        """Update a user's status."""
        if user_id not in self._presence:
            raise ValueError(f"User {user_id} is not in the session")
        self._presence[user_id].status = status
        self.broadcast("status_changed", {"user_id": str(user_id), "status": status})

    def get_presence(self) -> list[Presence]:
        """Return all present users."""
        return list(self._presence.values())

    # -- Locking ------------------------------------------------------------

    def lock_node(self, node_id: str, user_id: UUID, duration_seconds: int = 300) -> NodeLock:
        """Lock a node for exclusive editing."""
        existing = self._locks.get(node_id)
        now = datetime.now(timezone.utc)
        if existing is not None:
            if existing.locked_by != user_id and existing.expires_at > now:
                raise PermissionError(
                    f"Node {node_id} is locked by {existing.locked_by}"
                )
        lock = NodeLock(
            node_id=node_id,
            locked_by=user_id,
            locked_at=now,
            expires_at=datetime.fromtimestamp(
                now.timestamp() + duration_seconds, tz=timezone.utc
            ),
        )
        self._locks[node_id] = lock
        self.broadcast("node_locked", {"node_id": node_id, "locked_by": str(user_id)})
        return lock

    def unlock_node(self, node_id: str, user_id: UUID) -> None:
        """Unlock a node. Only the lock holder may unlock."""
        existing = self._locks.get(node_id)
        if existing is None:
            return
        if existing.locked_by != user_id:
            raise PermissionError(
                f"Node {node_id} is locked by {existing.locked_by}, not {user_id}"
            )
        del self._locks[node_id]
        self.broadcast("node_unlocked", {"node_id": node_id, "unlocked_by": str(user_id)})

    def is_locked(self, node_id: str) -> Optional[NodeLock]:
        """Check if a node is locked. Returns the lock or None."""
        lock = self._locks.get(node_id)
        if lock is None:
            return None
        if lock.expires_at <= datetime.now(timezone.utc):
            del self._locks[node_id]
            return None
        return lock

    # -- Comments -----------------------------------------------------------

    def add_comment(self, node_id: str, user_id: UUID, text: str) -> Comment:
        """Add a comment on a node."""
        comment = Comment(
            id=uuid4(),
            node_id=node_id,
            user_id=user_id,
            text=text,
            created_at=datetime.now(timezone.utc),
        )
        self._comments.append(comment)
        self.broadcast("comment_added", {"comment_id": str(comment.id), "node_id": node_id})
        return comment

    def resolve_comment(self, comment_id: UUID) -> None:
        """Mark a comment as resolved."""
        for c in self._comments:
            if c.id == comment_id:
                c.resolved = True
                self.broadcast("comment_resolved", {"comment_id": str(comment_id)})
                return
        raise ValueError(f"Comment {comment_id} not found")

    def get_comments(self, node_id: str | None = None) -> list[Comment]:
        """List comments, optionally filtered by node."""
        if node_id is None:
            return list(self._comments)
        return [c for c in self._comments if c.node_id == node_id]

    # -- Broadcast ----------------------------------------------------------

    def broadcast(self, event_type: str, payload: dict[str, Any]) -> list[BroadcastEvent]:
        """Queue an event for all current participants."""
        event = BroadcastEvent(event_type=event_type, payload=payload, timestamp=time.time())
        events: list[BroadcastEvent] = []
        for uid in self._presence:
            self._pending[uid].append(event)
            events.append(event)
        return events

    def get_pending_broadcasts(self, user_id: UUID) -> list[BroadcastEvent]:
        """Get and drain queued events for a user."""
        events = list(self._pending.get(user_id, []))
        self._pending[user_id] = []
        return events


# ---------------------------------------------------------------------------
# CollaborationManager
# ---------------------------------------------------------------------------

class CollaborationManager:
    """Manages multiple collaboration sessions."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, CollaborationSession] = {}

    def create_session(self, tenant_id: UUID) -> CollaborationSession:
        """Create and register a new session."""
        session_id = uuid4()
        session = CollaborationSession(session_id=session_id, tenant_id=tenant_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: UUID) -> Optional[CollaborationSession]:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self, tenant_id: UUID) -> list[CollaborationSession]:
        """List all sessions for a tenant."""
        return [s for s in self._sessions.values() if s.tenant_id == tenant_id]

    def close_session(self, session_id: UUID) -> None:
        """Close and remove a session."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
