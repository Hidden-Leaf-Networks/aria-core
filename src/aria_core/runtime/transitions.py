"""State transition definitions and validation.

Defines the valid state graph for the agent FSM. Every transition
is validated before execution — no implicit state changes allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from aria_core.runtime.models import AgentStateEnum


@dataclass
class TransitionResult:
    """Result of a state transition attempt."""

    from_state: AgentStateEnum
    to_state: AgentStateEnum
    timestamp: datetime
    success: bool
    error: str | None = None


# Valid state transitions (from → allowed to states)
VALID_TRANSITIONS: dict[AgentStateEnum, set[AgentStateEnum]] = {
    AgentStateEnum.IDLE: {
        AgentStateEnum.ROUTING,
        AgentStateEnum.IDLE,
    },
    AgentStateEnum.ROUTING: {
        AgentStateEnum.PLANNING,
        AgentStateEnum.RESPONDING,  # Direct response for simple queries
        AgentStateEnum.ERROR,
    },
    AgentStateEnum.PLANNING: {
        AgentStateEnum.EXECUTING_STEP,
        AgentStateEnum.RESPONDING,  # No steps needed
        AgentStateEnum.ERROR,
    },
    AgentStateEnum.EXECUTING_STEP: {
        AgentStateEnum.EXECUTING_STEP,  # Next step
        AgentStateEnum.AWAITING_APPROVAL,
        AgentStateEnum.RESPONDING,  # All steps done
        AgentStateEnum.ERROR,
    },
    AgentStateEnum.AWAITING_APPROVAL: {
        AgentStateEnum.EXECUTING_STEP,  # Approved, continue
        AgentStateEnum.RESPONDING,  # Denied, finish with partial results
        AgentStateEnum.AWAITING_APPROVAL,  # Still waiting
        AgentStateEnum.ERROR,
    },
    AgentStateEnum.RESPONDING: {
        AgentStateEnum.COMPLETE,
        AgentStateEnum.ERROR,
    },
    # Terminal states — no outgoing transitions
    AgentStateEnum.COMPLETE: set(),
    AgentStateEnum.ERROR: set(),
}


class Transition:
    """State transition logic."""

    @staticmethod
    def is_valid(from_state: AgentStateEnum, to_state: AgentStateEnum) -> bool:
        """Check if a transition is valid."""
        allowed = VALID_TRANSITIONS.get(from_state, set())
        return to_state in allowed

    @staticmethod
    def validate(from_state: AgentStateEnum, to_state: AgentStateEnum) -> TransitionResult:
        """Validate and record a transition."""
        if Transition.is_valid(from_state, to_state):
            return TransitionResult(
                from_state=from_state,
                to_state=to_state,
                timestamp=datetime.now(timezone.utc),
                success=True,
            )
        return TransitionResult(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(timezone.utc),
            success=False,
            error=f"Invalid transition: {from_state.value} -> {to_state.value}",
        )

    @staticmethod
    def get_allowed_transitions(state: AgentStateEnum) -> set[AgentStateEnum]:
        """Get all allowed transitions from a state."""
        return VALID_TRANSITIONS.get(state, set())

    @staticmethod
    def is_terminal(state: AgentStateEnum) -> bool:
        """Check if state is terminal (no outgoing transitions)."""
        return len(VALID_TRANSITIONS.get(state, set())) == 0
