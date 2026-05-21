"""Time-travel debugging API routes.

Exposes checkpoint inspection, rewind, and fork operations over HTTP.
All endpoints are scoped to a context_id or checkpoint_id.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from aria_core.runtime.time_travel import TimeTravel

# Module-level TimeTravel instance (shared across routes).
# In production this would be injected via dependency, similar to get_guard().
_time_travel: TimeTravel | None = None


def get_time_travel() -> TimeTravel:
    """Get or create the shared TimeTravel instance."""
    global _time_travel
    if _time_travel is None:
        _time_travel = TimeTravel()
    return _time_travel


def set_time_travel(tt: TimeTravel) -> None:
    """Replace the shared TimeTravel instance (useful for testing)."""
    global _time_travel
    _time_travel = tt


# -------------------------------------------------------------------
# GET /api/v1/time-travel/{context_id}/checkpoints
# -------------------------------------------------------------------


async def list_checkpoints(context_id: UUID) -> list[dict[str, Any]]:
    """List all checkpoints for a given context run."""
    tt = get_time_travel()
    checkpoints = tt.list_checkpoints(context_id=context_id)
    return [cp.model_dump(mode="json") for cp in checkpoints]


# -------------------------------------------------------------------
# GET /api/v1/time-travel/checkpoints/{checkpoint_id}
# -------------------------------------------------------------------


async def get_checkpoint(checkpoint_id: UUID) -> dict[str, Any] | None:
    """Get a specific checkpoint by ID."""
    tt = get_time_travel()
    cp = tt.get_checkpoint(checkpoint_id)
    if cp is None:
        return None
    return cp.model_dump(mode="json")


# -------------------------------------------------------------------
# POST /api/v1/time-travel/checkpoints/{checkpoint_id}/rewind
# -------------------------------------------------------------------


async def rewind_checkpoint(checkpoint_id: UUID) -> dict[str, Any]:
    """Rewind to a checkpoint — returns the restored AgentContext."""
    tt = get_time_travel()
    try:
        ctx = tt.rewind(checkpoint_id)
        return {"success": True, "context": ctx.model_dump(mode="json")}
    except KeyError:
        return {"success": False, "error": f"Checkpoint {checkpoint_id} not found"}


# -------------------------------------------------------------------
# POST /api/v1/time-travel/checkpoints/{checkpoint_id}/fork
# -------------------------------------------------------------------


async def fork_checkpoint(
    checkpoint_id: UUID,
    modifications: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fork a new context from a checkpoint with optional modifications."""
    tt = get_time_travel()
    try:
        ctx = tt.fork(checkpoint_id, modifications=modifications)
        return {"success": True, "context": ctx.model_dump(mode="json")}
    except KeyError:
        return {"success": False, "error": f"Checkpoint {checkpoint_id} not found"}
