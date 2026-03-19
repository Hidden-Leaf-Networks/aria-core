"""Deterministic agent state machine.

The core execution engine for Aria Core agents. Guarantees:
- No uncontrolled loops (max_steps enforced)
- Explicit state transitions (validated against transition graph)
- Full observability (events emitted for every transition)
- Timeout protection (configurable)
- Auto-recovery from stuck states

Usage:
    from aria_core.runtime import AgentStateMachine
    from aria_core.runtime.models import AgentConfig

    machine = AgentStateMachine(
        router=my_router,
        planner=my_planner,
        executor=my_executor,
        adapter=my_llm_adapter,
    )
    result = await machine.process_message("Hello, world!")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID, uuid4

from aria_core.runtime.models import (
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentStateEnum,
    ChatMessage,
    MessageRole,
)
from aria_core.runtime.states import State, STATE_REGISTRY
from aria_core.runtime.transitions import Transition, TransitionResult

logger = logging.getLogger("aria_core.runtime")

# Event callback type
EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class RouterProtocol(Protocol):
    """Protocol for intent routers."""

    async def route(self, context: AgentContext) -> dict[str, Any]: ...


class PlannerProtocol(Protocol):
    """Protocol for plan generators."""

    async def create_plan(self, context: AgentContext) -> Any: ...


class ExecutorProtocol(Protocol):
    """Protocol for step executors."""

    async def execute_step(self, context: AgentContext) -> dict[str, Any]: ...


class AdapterProtocol(Protocol):
    """Protocol for LLM adapters."""

    async def generate_response(self, context: AgentContext) -> str: ...


class AgentStateMachine:
    """Deterministic finite state machine for agent execution.

    Guarantees:
    - No uncontrolled loops (max_steps enforced)
    - Explicit state transitions (validated)
    - Full observability (events emitted for each transition)
    """

    def __init__(
        self,
        router: RouterProtocol,
        planner: PlannerProtocol,
        executor: ExecutorProtocol,
        adapter: AdapterProtocol,
        config: AgentConfig | None = None,
        event_callback: EventCallback | None = None,
        states: dict[AgentStateEnum, type[State]] | None = None,
    ) -> None:
        self.router = router
        self.planner = planner
        self.executor = executor
        self.adapter = adapter
        self.config = config or AgentConfig()
        self.event_callback = event_callback

        # Initialize state instances (allow custom state overrides)
        registry = states or STATE_REGISTRY
        self._states: dict[AgentStateEnum, State] = {
            state_enum: state_class() for state_enum, state_class in registry.items()
        }

        # Execution state
        self._current_state: AgentStateEnum = AgentStateEnum.IDLE
        self._transition_history: list[TransitionResult] = []
        self._running: bool = False

    @property
    def current_state(self) -> AgentStateEnum:
        """Get current state."""
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return Transition.is_terminal(self._current_state)

    @property
    def transition_history(self) -> list[TransitionResult]:
        """Get transition history."""
        return self._transition_history.copy()

    async def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event if callback is registered."""
        if self.event_callback:
            await self.event_callback(event_type, payload)

    async def transition_to(
        self, next_state: AgentStateEnum, context: AgentContext
    ) -> bool:
        """Attempt to transition to a new state.

        Returns True if transition succeeded, False otherwise.
        """
        result = Transition.validate(self._current_state, next_state)
        self._transition_history.append(result)

        if not result.success:
            await self.emit_event(
                "transition.invalid",
                {
                    "from": self._current_state.value,
                    "to": next_state.value,
                    "error": result.error,
                },
            )
            return False

        # Exit current state
        current = self._states[self._current_state]
        await current.exit(self, context)

        # Update state
        prev_state = self._current_state
        self._current_state = next_state

        await self.emit_event(
            "transition.complete",
            {"from": prev_state.value, "to": next_state.value},
        )

        # Enter new state
        new_state = self._states[next_state]
        await new_state.enter(self, context)

        return True

    async def run(self, context: AgentContext) -> AgentResult:
        """Run the state machine until terminal state.

        Args:
            context: Agent execution context.

        Returns:
            Final agent result.
        """
        if self._running:
            logger.warning(
                "State machine was still marked running — resetting from previous interrupted run"
            )
            self.reset()

        self._running = True
        self._current_state = AgentStateEnum.IDLE
        self._transition_history = []

        await self.emit_event("agent.start", {"context_id": str(context.id)})

        try:
            await self._states[self._current_state].enter(self, context)

            iteration = 0
            max_iterations = self.config.max_steps * 3  # Safety limit

            while not self.is_terminal and iteration < max_iterations:
                iteration += 1

                current = self._states[self._current_state]
                next_state = await current.execute(self, context)

                if next_state != self._current_state:
                    success = await self.transition_to(next_state, context)
                    if not success:
                        context.metadata["error"] = (
                            f"Invalid transition: {self._current_state} -> {next_state}"
                        )
                        await self.transition_to(AgentStateEnum.ERROR, context)
                        break

                await asyncio.sleep(0)

            if iteration >= max_iterations:
                context.metadata["error"] = "Maximum iterations exceeded"
                if not self.is_terminal:
                    await self.transition_to(AgentStateEnum.ERROR, context)

        except Exception as e:
            context.metadata["error"] = str(e)
            if not self.is_terminal:
                try:
                    await self.transition_to(AgentStateEnum.ERROR, context)
                except Exception:
                    self._current_state = AgentStateEnum.ERROR

        finally:
            self._running = False

        return AgentResult(
            state=self._current_state,
            context=context,
            error=context.metadata.get("error"),
        )

    async def process_message(
        self,
        message: str,
        conversation_id: UUID | None = None,
        config: AgentConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Process a user message.

        Convenience method that creates context and runs the machine.
        """
        config = config or self.config
        context = AgentContext(
            conversation_id=conversation_id or uuid4(),
            config=config,
            messages=[ChatMessage(role=MessageRole.USER, content=message)],
            metadata=metadata or {},
        )
        return await self.run(context)

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self._current_state = AgentStateEnum.IDLE
        self._transition_history = []
        self._running = False
