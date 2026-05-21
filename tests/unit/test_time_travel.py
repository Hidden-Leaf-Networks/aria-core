"""Tests for time-travel debugging of the agent FSM."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from aria_core.runtime.models import (
    AgentConfig,
    AgentContext,
    AgentStateEnum,
    ChatMessage,
    MessageRole,
)
from aria_core.runtime.time_travel import Checkpoint, CheckpointDiff, TimeTravel
from aria_core.api.routes.time_travel import (
    fork_checkpoint,
    get_checkpoint,
    get_time_travel,
    list_checkpoints,
    rewind_checkpoint,
    set_time_travel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(**kwargs: Any) -> AgentContext:
    """Create a minimal AgentContext with optional overrides."""
    defaults: dict[str, Any] = {
        "messages": [ChatMessage(role=MessageRole.USER, content="hello")],
        "metadata": {},
    }
    defaults.update(kwargs)
    return AgentContext(**defaults)


def _make_tt_with_checkpoints(
    ctx: AgentContext | None = None,
    states: list[AgentStateEnum] | None = None,
) -> tuple[TimeTravel, AgentContext, list[Checkpoint]]:
    """Create a TimeTravel instance pre-loaded with checkpoints."""
    tt = TimeTravel()
    ctx = ctx or _make_context()
    states = states or [
        AgentStateEnum.IDLE,
        AgentStateEnum.ROUTING,
        AgentStateEnum.RESPONDING,
        AgentStateEnum.COMPLETE,
    ]
    cps: list[Checkpoint] = []
    for i, state in enumerate(states):
        cp = tt.capture(
            context=ctx,
            state=state,
            event_type=f"test.event.{i}",
            event_payload={"index": i},
        )
        cps.append(cp)
    return tt, ctx, cps


# ---------------------------------------------------------------------------
# Checkpoint creation
# ---------------------------------------------------------------------------


class TestCheckpointCreation:
    def test_capture_creates_checkpoint(self) -> None:
        """capture() stores a Checkpoint with correct fields."""
        tt = TimeTravel()
        ctx = _make_context()
        cp = tt.capture(
            context=ctx,
            state=AgentStateEnum.IDLE,
            event_type="agent.start",
            event_payload={"context_id": str(ctx.id)},
        )
        assert isinstance(cp, Checkpoint)
        assert cp.context_id == ctx.id
        assert cp.state == AgentStateEnum.IDLE
        assert cp.event_type == "agent.start"

    def test_capture_on_multiple_transitions(self) -> None:
        """Each transition produces a distinct checkpoint."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        assert len(cps) == 4
        ids = {cp.checkpoint_id for cp in cps}
        assert len(ids) == 4  # all unique

    def test_checkpoint_contains_full_context_snapshot(self) -> None:
        """Snapshot includes all AgentContext fields."""
        tt = TimeTravel()
        ctx = _make_context(step_count=3, metadata={"key": "value"})
        cp = tt.capture(ctx, AgentStateEnum.EXECUTING_STEP, "test", {})
        snap = cp.context_snapshot
        assert snap["step_count"] == 3
        assert snap["metadata"]["key"] == "value"


# ---------------------------------------------------------------------------
# Rewind
# ---------------------------------------------------------------------------


class TestRewind:
    def test_rewind_restores_context(self) -> None:
        """rewind() returns an AgentContext matching the checkpoint snapshot."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        restored = tt.rewind(cps[0].checkpoint_id)
        assert isinstance(restored, AgentContext)
        assert restored.id == ctx.id
        assert restored.messages[0].content == "hello"

    def test_rewind_returns_independent_copy(self) -> None:
        """Rewound context is a distinct object, not a reference."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        restored = tt.rewind(cps[0].checkpoint_id)
        restored.metadata["mutated"] = True
        # Original snapshot must be untouched
        snap = cps[0].context_snapshot
        assert "mutated" not in snap.get("metadata", {})

    def test_rewind_unknown_id_raises(self) -> None:
        """rewind() raises KeyError for non-existent checkpoint."""
        tt = TimeTravel()
        with pytest.raises(KeyError):
            tt.rewind(uuid4())


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


class TestFork:
    def test_fork_creates_new_context(self) -> None:
        """fork() returns a context with a fresh id."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        forked = tt.fork(cps[1].checkpoint_id)
        assert forked.id != ctx.id  # new identity

    def test_fork_with_modifications(self) -> None:
        """fork() applies the provided modifications."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        forked = tt.fork(cps[0].checkpoint_id, modifications={"step_count": 99})
        assert forked.step_count == 99

    def test_fork_preserves_unmodified_fields(self) -> None:
        """Fields not in modifications stay as they were in the checkpoint."""
        tt = TimeTravel()
        ctx = _make_context(step_count=5, metadata={"original": True})
        cp = tt.capture(ctx, AgentStateEnum.IDLE, "test", {})
        forked = tt.fork(cp.checkpoint_id, modifications={"step_count": 0})
        assert forked.step_count == 0
        assert forked.metadata["original"] is True

    def test_fork_unknown_id_raises(self) -> None:
        """fork() raises KeyError for non-existent checkpoint."""
        tt = TimeTravel()
        with pytest.raises(KeyError):
            tt.fork(uuid4())


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_detects_state_change(self) -> None:
        """diff() sets state_changed=True when states differ."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        result = tt.diff(cps[0].checkpoint_id, cps[1].checkpoint_id)
        assert isinstance(result, CheckpointDiff)
        assert result.state_changed is True
        assert result.from_state == AgentStateEnum.IDLE
        assert result.to_state == AgentStateEnum.ROUTING

    def test_diff_no_state_change(self) -> None:
        """diff() sets state_changed=False when states are the same."""
        tt = TimeTravel()
        ctx = _make_context()
        cp1 = tt.capture(ctx, AgentStateEnum.IDLE, "a", {})
        cp2 = tt.capture(ctx, AgentStateEnum.IDLE, "b", {})
        result = tt.diff(cp1.checkpoint_id, cp2.checkpoint_id)
        assert result.state_changed is False

    def test_diff_detects_context_changes(self) -> None:
        """diff() reports changed context fields."""
        tt = TimeTravel()
        ctx1 = _make_context(step_count=0)
        ctx2 = _make_context(step_count=3)
        # Use same id so we can compare meaningfully
        ctx2 = ctx2.model_copy(update={"id": ctx1.id})
        cp1 = tt.capture(ctx1, AgentStateEnum.IDLE, "a", {})
        cp2 = tt.capture(ctx2, AgentStateEnum.EXECUTING_STEP, "b", {})
        result = tt.diff(cp1.checkpoint_id, cp2.checkpoint_id)
        assert "step_count" in result.context_changes
        assert result.context_changes["step_count"]["before"] == 0
        assert result.context_changes["step_count"]["after"] == 3

    def test_diff_unknown_id_raises(self) -> None:
        """diff() raises KeyError for non-existent checkpoint."""
        tt = TimeTravel()
        with pytest.raises(KeyError):
            tt.diff(uuid4(), uuid4())


# ---------------------------------------------------------------------------
# list / get queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_list_checkpoints_all(self) -> None:
        """list_checkpoints() with no filter returns everything."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        result = tt.list_checkpoints()
        assert len(result) == 4

    def test_list_checkpoints_by_context(self) -> None:
        """list_checkpoints(context_id=...) filters correctly."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        # Add a checkpoint for a *different* context
        other = _make_context()
        tt.capture(other, AgentStateEnum.IDLE, "other", {})

        result = tt.list_checkpoints(context_id=ctx.id)
        assert len(result) == 4  # only the original 4

    def test_get_checkpoint_found(self) -> None:
        tt, ctx, cps = _make_tt_with_checkpoints()
        found = tt.get_checkpoint(cps[2].checkpoint_id)
        assert found is not None
        assert found.checkpoint_id == cps[2].checkpoint_id

    def test_get_checkpoint_not_found(self) -> None:
        tt = TimeTravel()
        assert tt.get_checkpoint(uuid4()) is None


# ---------------------------------------------------------------------------
# Full lifecycle capture
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_full_lifecycle_states_captured(self) -> None:
        """Capture across a full IDLE -> ROUTING -> RESPONDING -> COMPLETE cycle."""
        tt, ctx, cps = _make_tt_with_checkpoints(
            states=[
                AgentStateEnum.IDLE,
                AgentStateEnum.ROUTING,
                AgentStateEnum.RESPONDING,
                AgentStateEnum.COMPLETE,
            ]
        )
        states_captured = [cp.state for cp in cps]
        assert states_captured == [
            AgentStateEnum.IDLE,
            AgentStateEnum.ROUTING,
            AgentStateEnum.RESPONDING,
            AgentStateEnum.COMPLETE,
        ]

    def test_export_import_round_trip(self) -> None:
        """Checkpoints survive export -> import cycle."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        exported = tt.export_checkpoints(context_id=ctx.id)
        assert len(exported) == 4

        tt2 = TimeTravel()
        count = tt2.import_checkpoints(exported)
        assert count == 4
        assert len(tt2.list_checkpoints()) == 4

    def test_clear_removes_checkpoints(self) -> None:
        """clear() empties the store."""
        tt, ctx, cps = _make_tt_with_checkpoints()
        removed = tt.clear()
        assert removed == 4
        assert len(tt.list_checkpoints()) == 0


# ---------------------------------------------------------------------------
# Event callback integration
# ---------------------------------------------------------------------------


class TestEventCallback:
    async def test_callback_captures_on_transition_complete(self) -> None:
        """The event callback captures a checkpoint for transition.complete events."""
        tt = TimeTravel()
        ctx = _make_context()
        cb = tt.as_event_callback()

        # Simulate the events the FSM would fire
        await cb("transition.complete", {
            "from": "idle",
            "to": "routing",
            "_context": ctx,
        })
        await cb("transition.complete", {
            "from": "routing",
            "to": "responding",
            "_context": ctx,
        })

        checkpoints = tt.list_checkpoints(context_id=ctx.id)
        assert len(checkpoints) == 2
        assert checkpoints[0].state == AgentStateEnum.ROUTING
        assert checkpoints[1].state == AgentStateEnum.RESPONDING

    async def test_callback_captures_agent_start(self) -> None:
        """agent.start event is captured as an IDLE checkpoint."""
        tt = TimeTravel()
        ctx = _make_context()
        cb = tt.as_event_callback()

        await cb("agent.start", {"context_id": str(ctx.id), "_context": ctx})

        checkpoints = tt.list_checkpoints(context_id=ctx.id)
        assert len(checkpoints) == 1
        assert checkpoints[0].state == AgentStateEnum.IDLE

    async def test_callback_ignores_unknown_events(self) -> None:
        """Events that aren't transition.complete or agent.start are skipped."""
        tt = TimeTravel()
        cb = tt.as_event_callback()

        await cb("some.other.event", {"data": 123})
        assert len(tt.list_checkpoints()) == 0


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    async def test_list_checkpoints_route(self) -> None:
        tt, ctx, cps = _make_tt_with_checkpoints()
        set_time_travel(tt)

        result = await list_checkpoints(ctx.id)
        assert len(result) == 4
        assert all(isinstance(r, dict) for r in result)

    async def test_get_checkpoint_route(self) -> None:
        tt, ctx, cps = _make_tt_with_checkpoints()
        set_time_travel(tt)

        result = await get_checkpoint(cps[0].checkpoint_id)
        assert result is not None
        assert result["checkpoint_id"] == str(cps[0].checkpoint_id)

    async def test_get_checkpoint_not_found_route(self) -> None:
        set_time_travel(TimeTravel())
        result = await get_checkpoint(uuid4())
        assert result is None

    async def test_rewind_route(self) -> None:
        tt, ctx, cps = _make_tt_with_checkpoints()
        set_time_travel(tt)

        result = await rewind_checkpoint(cps[1].checkpoint_id)
        assert result["success"] is True
        assert "context" in result

    async def test_rewind_not_found_route(self) -> None:
        set_time_travel(TimeTravel())
        result = await rewind_checkpoint(uuid4())
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_fork_route(self) -> None:
        tt, ctx, cps = _make_tt_with_checkpoints()
        set_time_travel(tt)

        result = await fork_checkpoint(
            cps[0].checkpoint_id,
            modifications={"step_count": 42},
        )
        assert result["success"] is True
        assert result["context"]["step_count"] == 42

    async def test_fork_not_found_route(self) -> None:
        set_time_travel(TimeTravel())
        result = await fork_checkpoint(uuid4())
        assert result["success"] is False
        assert "not found" in result["error"]
