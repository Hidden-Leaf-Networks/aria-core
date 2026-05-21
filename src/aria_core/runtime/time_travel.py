"""Time-travel debugging for the agent FSM state machine.

Captures snapshots of AgentContext at every state transition, enabling
rewind, fork, and diff operations for debugging and replay.

Usage:
    from aria_core.runtime.time_travel import TimeTravel

    tt = TimeTravel()
    machine = AgentStateMachine(
        router=my_router,
        planner=my_planner,
        executor=my_executor,
        adapter=my_adapter,
        event_callback=tt.as_event_callback(),
    )
    result = await machine.run(context)

    # Inspect captured checkpoints
    checkpoints = tt.list_checkpoints(context.id)
    old_ctx = tt.rewind(checkpoints[0].checkpoint_id)
    forked = tt.fork(checkpoints[0].checkpoint_id, {"step_count": 0})
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from aria_core.runtime.models import AgentContext, AgentStateEnum, BaseModel

logger = logging.getLogger("aria_core.runtime.time_travel")

# EventCallback type (mirrors state_machine.py)
EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class Checkpoint(BaseModel):
    """Snapshot of AgentContext + FSM state at a point in time."""

    checkpoint_id: UUID
    context_id: UUID
    state: AgentStateEnum
    context_snapshot: dict[str, Any]
    timestamp: datetime
    event_type: str
    event_payload: dict[str, Any]


class CheckpointDiff(BaseModel):
    """Differences between two checkpoints."""

    checkpoint_a_id: UUID
    checkpoint_b_id: UUID
    state_changed: bool
    from_state: AgentStateEnum
    to_state: AgentStateEnum
    context_changes: dict[str, dict[str, Any]]


class TimeTravel:
    """Time-travel debugger for the agent FSM.

    Hooks into the state machine event system to capture a checkpoint
    on every state transition. Supports rewind, fork, diff, and
    persistence integration.
    """

    def __init__(self) -> None:
        # All checkpoints indexed by checkpoint_id
        self._checkpoints: dict[UUID, Checkpoint] = {}
        # Checkpoints grouped by context_id for fast lookup
        self._by_context: dict[UUID, list[UUID]] = {}

    # -------------------------------------------------------------------
    # Core capture
    # -------------------------------------------------------------------

    def capture(
        self,
        context: AgentContext,
        state: AgentStateEnum,
        event_type: str,
        event_payload: dict[str, Any],
    ) -> Checkpoint:
        """Create and store a checkpoint from the current context + state."""
        cp = Checkpoint(
            checkpoint_id=uuid4(),
            context_id=context.id,
            state=state,
            context_snapshot=context.model_dump(mode="json"),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            event_payload=event_payload,
        )
        self._checkpoints[cp.checkpoint_id] = cp
        self._by_context.setdefault(context.id, []).append(cp.checkpoint_id)
        logger.debug(
            "Checkpoint %s captured: state=%s event=%s",
            cp.checkpoint_id,
            state.value,
            event_type,
        )
        return cp

    # -------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------

    def list_checkpoints(self, context_id: UUID | None = None) -> list[Checkpoint]:
        """Return checkpoints, optionally filtered by context_id.

        Results are sorted chronologically (oldest first).
        """
        if context_id is not None:
            ids = self._by_context.get(context_id, [])
            cps = [self._checkpoints[cid] for cid in ids if cid in self._checkpoints]
        else:
            cps = list(self._checkpoints.values())
        return sorted(cps, key=lambda c: c.timestamp)

    def get_checkpoint(self, checkpoint_id: UUID) -> Checkpoint | None:
        """Get a specific checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    # -------------------------------------------------------------------
    # Rewind
    # -------------------------------------------------------------------

    def rewind(self, checkpoint_id: UUID) -> AgentContext:
        """Restore the AgentContext from a checkpoint.

        Returns a *new* AgentContext instance (deep copy from snapshot).

        Raises:
            KeyError: If checkpoint_id is not found.
        """
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")
        return AgentContext(**cp.context_snapshot)

    # -------------------------------------------------------------------
    # Fork
    # -------------------------------------------------------------------

    def fork(
        self,
        checkpoint_id: UUID,
        modifications: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Create a new context from a checkpoint with optional modifications.

        The forked context gets a new id and created_at timestamp so it
        is distinguishable from the original.

        Raises:
            KeyError: If checkpoint_id is not found.
        """
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")

        snapshot = dict(cp.context_snapshot)
        # Apply modifications to the raw snapshot before parsing
        if modifications:
            snapshot.update(modifications)

        # Always assign a fresh id so the fork is a distinct entity
        snapshot["id"] = str(uuid4())
        snapshot["created_at"] = datetime.now(timezone.utc).isoformat()

        return AgentContext(**snapshot)

    # -------------------------------------------------------------------
    # Diff
    # -------------------------------------------------------------------

    def diff(self, checkpoint_a_id: UUID, checkpoint_b_id: UUID) -> CheckpointDiff:
        """Show what changed between two checkpoints.

        Raises:
            KeyError: If either checkpoint_id is not found.
        """
        cp_a = self._checkpoints.get(checkpoint_a_id)
        cp_b = self._checkpoints.get(checkpoint_b_id)
        if cp_a is None:
            raise KeyError(f"Checkpoint {checkpoint_a_id} not found")
        if cp_b is None:
            raise KeyError(f"Checkpoint {checkpoint_b_id} not found")

        changes: dict[str, dict[str, Any]] = {}
        snap_a = cp_a.context_snapshot
        snap_b = cp_b.context_snapshot

        all_keys = set(snap_a.keys()) | set(snap_b.keys())
        for key in all_keys:
            val_a = snap_a.get(key)
            val_b = snap_b.get(key)
            if val_a != val_b:
                changes[key] = {"before": val_a, "after": val_b}

        return CheckpointDiff(
            checkpoint_a_id=checkpoint_a_id,
            checkpoint_b_id=checkpoint_b_id,
            state_changed=cp_a.state != cp_b.state,
            from_state=cp_a.state,
            to_state=cp_b.state,
            context_changes=changes,
        )

    # -------------------------------------------------------------------
    # Event callback integration
    # -------------------------------------------------------------------

    def as_event_callback(self) -> EventCallback:
        """Return an EventCallback compatible with AgentStateMachine.

        Listens for ``transition.complete`` events and captures a
        checkpoint for each one. Also captures ``agent.start`` so the
        initial state is recorded.
        """
        # We keep a mapping of context_id -> latest known context
        # so we can reconstruct the snapshot from events.
        _contexts: dict[str, AgentContext] = {}

        async def _callback(event_type: str, payload: dict[str, Any]) -> None:
            # On agent.start, stash the context reference
            if event_type == "agent.start":
                ctx = payload.get("_context")
                if ctx is not None and isinstance(ctx, AgentContext):
                    _contexts[str(ctx.id)] = ctx
                    self.capture(
                        context=ctx,
                        state=AgentStateEnum.IDLE,
                        event_type=event_type,
                        event_payload={k: v for k, v in payload.items() if k != "_context"},
                    )
                return

            if event_type == "transition.complete":
                to_state_str = payload.get("to", "")
                from_state_str = payload.get("from", "")
                context_id = payload.get("context_id")
                ctx = payload.get("_context")

                if ctx is not None and isinstance(ctx, AgentContext):
                    _contexts[str(ctx.id)] = ctx

                # Try to resolve state enum
                try:
                    to_state = AgentStateEnum(to_state_str)
                except ValueError:
                    return

                # Find the context
                resolved_ctx: AgentContext | None = None
                if ctx is not None and isinstance(ctx, AgentContext):
                    resolved_ctx = ctx
                elif context_id:
                    resolved_ctx = _contexts.get(str(context_id))

                if resolved_ctx is None:
                    # Fallback: look for any known context
                    for cid, c in _contexts.items():
                        resolved_ctx = c
                        break

                if resolved_ctx is not None:
                    self.capture(
                        context=resolved_ctx,
                        state=to_state,
                        event_type=event_type,
                        event_payload={
                            k: v for k, v in payload.items() if k != "_context"
                        },
                    )

        return _callback

    # -------------------------------------------------------------------
    # Persistence helpers
    # -------------------------------------------------------------------

    def export_checkpoints(self, context_id: UUID | None = None) -> list[dict[str, Any]]:
        """Serialize checkpoints for persistence.

        Returns a list of dicts ready for JSON storage / provider save.
        """
        return [cp.model_dump(mode="json") for cp in self.list_checkpoints(context_id)]

    def import_checkpoints(self, data: list[dict[str, Any]]) -> int:
        """Load checkpoints from serialized data (e.g., from persistence).

        Returns the number of checkpoints imported.
        """
        count = 0
        for item in data:
            cp = Checkpoint(**item)
            self._checkpoints[cp.checkpoint_id] = cp
            self._by_context.setdefault(cp.context_id, []).append(cp.checkpoint_id)
            count += 1
        return count

    def clear(self, context_id: UUID | None = None) -> int:
        """Remove checkpoints. If context_id given, only that context's checkpoints.

        Returns the number of checkpoints removed.
        """
        if context_id is not None:
            ids = self._by_context.pop(context_id, [])
            for cid in ids:
                self._checkpoints.pop(cid, None)
            return len(ids)
        count = len(self._checkpoints)
        self._checkpoints.clear()
        self._by_context.clear()
        return count
