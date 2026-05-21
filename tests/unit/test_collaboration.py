"""Tests for real-time collaboration — presence, locking, comments, broadcast."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from aria_core.collaboration import (
    CollaborationManager,
    CollaborationSession,
    Comment,
    Cursor,
    NodeLock,
    Presence,
)


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.fixture
def session(tenant_id):
    return CollaborationSession(session_id=uuid4(), tenant_id=tenant_id)


@pytest.fixture
def manager():
    return CollaborationManager()


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------

class TestPresence:
    def test_join_adds_user(self, session) -> None:
        uid = uuid4()
        p = session.join(uid, "canvas")
        assert p.user_id == uid
        assert p.page == "canvas"
        assert p.status == "active"

    def test_leave_removes_user(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.leave(uid)
        assert session.get_presence() == []

    def test_get_presence_lists_all(self, session) -> None:
        u1, u2 = uuid4(), uuid4()
        session.join(u1, "page1")
        session.join(u2, "page2")
        presence = session.get_presence()
        assert len(presence) == 2

    def test_update_cursor(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        cursor = session.update_cursor(uid, 100.5, 200.3)
        assert cursor.position == (100.5, 200.3)
        assert cursor.user_id == uid

    def test_update_cursor_not_in_session_raises(self, session) -> None:
        with pytest.raises(ValueError):
            session.update_cursor(uuid4(), 0, 0)

    def test_update_status(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.update_status(uid, "idle")
        p = session.get_presence()[0]
        assert p.status == "idle"

    def test_update_status_not_in_session_raises(self, session) -> None:
        with pytest.raises(ValueError):
            session.update_status(uuid4(), "away")

    def test_connected_at_set(self, session) -> None:
        uid = uuid4()
        p = session.join(uid, "canvas")
        assert isinstance(p.connected_at, datetime)
        assert p.connected_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class TestLocking:
    def test_lock_node(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        lock = session.lock_node("node-1", uid)
        assert lock.node_id == "node-1"
        assert lock.locked_by == uid

    def test_lock_prevents_other_user(self, session) -> None:
        u1, u2 = uuid4(), uuid4()
        session.join(u1, "canvas")
        session.join(u2, "canvas")
        session.lock_node("node-1", u1)
        with pytest.raises(PermissionError):
            session.lock_node("node-1", u2)

    def test_unlock_node(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.lock_node("node-1", uid)
        session.unlock_node("node-1", uid)
        assert session.is_locked("node-1") is None

    def test_unlock_by_wrong_user_raises(self, session) -> None:
        u1, u2 = uuid4(), uuid4()
        session.join(u1, "canvas")
        session.join(u2, "canvas")
        session.lock_node("node-1", u1)
        with pytest.raises(PermissionError):
            session.unlock_node("node-1", u2)

    def test_is_locked_returns_lock(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.lock_node("node-1", uid, duration_seconds=600)
        lock = session.is_locked("node-1")
        assert lock is not None
        assert lock.locked_by == uid

    def test_is_locked_returns_none_when_not_locked(self, session) -> None:
        assert session.is_locked("node-99") is None

    def test_leave_releases_locks(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.lock_node("node-1", uid)
        session.leave(uid)
        assert session.is_locked("node-1") is None

    def test_same_user_can_relock(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.lock_node("node-1", uid)
        # Same user can re-lock (extend)
        lock = session.lock_node("node-1", uid, duration_seconds=600)
        assert lock.locked_by == uid


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

class TestComments:
    def test_add_comment(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        comment = session.add_comment("node-1", uid, "looks good")
        assert comment.text == "looks good"
        assert comment.node_id == "node-1"
        assert comment.resolved is False

    def test_resolve_comment(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        comment = session.add_comment("node-1", uid, "fix this")
        session.resolve_comment(comment.id)
        comments = session.get_comments("node-1")
        assert comments[0].resolved is True

    def test_resolve_nonexistent_raises(self, session) -> None:
        with pytest.raises(ValueError):
            session.resolve_comment(uuid4())

    def test_get_comments_all(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.add_comment("node-1", uid, "a")
        session.add_comment("node-2", uid, "b")
        assert len(session.get_comments()) == 2

    def test_get_comments_filtered(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.add_comment("node-1", uid, "a")
        session.add_comment("node-2", uid, "b")
        assert len(session.get_comments("node-1")) == 1


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

class TestBroadcast:
    def test_broadcast_queues_for_all(self, session) -> None:
        u1, u2 = uuid4(), uuid4()
        session.join(u1, "canvas")
        session.join(u2, "canvas")
        # drain join events
        session.get_pending_broadcasts(u1)
        session.get_pending_broadcasts(u2)
        events = session.broadcast("test_event", {"data": 1})
        assert len(events) == 2

    def test_get_pending_broadcasts_drains(self, session) -> None:
        uid = uuid4()
        session.join(uid, "canvas")
        session.broadcast("ping", {})
        events = session.get_pending_broadcasts(uid)
        assert len(events) > 0
        # Second call should be empty
        assert session.get_pending_broadcasts(uid) == []

    def test_broadcast_no_participants(self, session) -> None:
        events = session.broadcast("orphan", {})
        assert events == []


# ---------------------------------------------------------------------------
# CollaborationManager
# ---------------------------------------------------------------------------

class TestCollaborationManager:
    def test_create_session(self, manager, tenant_id) -> None:
        s = manager.create_session(tenant_id)
        assert s.tenant_id == tenant_id

    def test_get_session(self, manager, tenant_id) -> None:
        s = manager.create_session(tenant_id)
        found = manager.get_session(s.session_id)
        assert found is s

    def test_get_session_not_found(self, manager) -> None:
        assert manager.get_session(uuid4()) is None

    def test_list_sessions(self, manager, tenant_id) -> None:
        t2 = uuid4()
        manager.create_session(tenant_id)
        manager.create_session(tenant_id)
        manager.create_session(t2)
        assert len(manager.list_sessions(tenant_id)) == 2
        assert len(manager.list_sessions(t2)) == 1

    def test_close_session(self, manager, tenant_id) -> None:
        s = manager.create_session(tenant_id)
        manager.close_session(s.session_id)
        assert manager.get_session(s.session_id) is None

    def test_close_nonexistent_raises(self, manager) -> None:
        with pytest.raises(ValueError):
            manager.close_session(uuid4())
