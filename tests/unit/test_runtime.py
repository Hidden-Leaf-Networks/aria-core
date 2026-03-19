"""Tests for the FSM agent runtime."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from aria_core.runtime.models import (
    AgentConfig,
    AgentContext,
    AgentStateEnum,
    ChatMessage,
    MessageRole,
)
from aria_core.runtime.state_machine import AgentStateMachine
from aria_core.runtime.transitions import Transition, VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# Stub implementations of the Protocol interfaces
# ---------------------------------------------------------------------------


class StubRouter:
    """Routes everything to 'direct' strategy."""

    def __init__(self, strategy: str = "direct") -> None:
        self.strategy = strategy
        self.call_count = 0

    async def route(self, context: AgentContext) -> dict[str, Any]:
        self.call_count += 1
        return {"strategy": self.strategy}


class StubPlanner:
    """Returns a plan with the given number of steps."""

    def __init__(self, num_steps: int = 2) -> None:
        self.num_steps = num_steps

    async def create_plan(self, context: AgentContext) -> Any:
        from pydantic import BaseModel, Field
        from uuid import uuid4 as _uuid4

        class Step(BaseModel):
            id: str = Field(default_factory=lambda: str(_uuid4()))
            action: str = "test_action"

        class Plan(BaseModel):
            id: Any = Field(default_factory=_uuid4)
            steps: list[Step] = Field(default_factory=list)

            def model_dump(self, **kwargs: Any) -> dict[str, Any]:
                return {
                    "id": str(self.id),
                    "steps": [s.model_dump() for s in self.steps],
                }

        steps = [Step(action=f"step_{i}") for i in range(self.num_steps)]
        return Plan(steps=steps)


class StubExecutor:
    """Succeeds on every step."""

    def __init__(self, require_approval: bool = False) -> None:
        self.require_approval = require_approval
        self.executed_steps: list[int] = []

    async def execute_step(self, context: AgentContext) -> dict[str, Any]:
        self.executed_steps.append(context.current_step_index)
        return {"success": True, "requires_approval": self.require_approval}


class StubAdapter:
    """Returns a fixed response."""

    def __init__(self, response: str = "Hello from Aria Core!") -> None:
        self.response = response

    async def generate_response(self, context: AgentContext) -> str:
        return self.response


# ---------------------------------------------------------------------------
# Transition validation tests
# ---------------------------------------------------------------------------


class TestTransitions:
    def test_valid_idle_to_routing(self) -> None:
        assert Transition.is_valid(AgentStateEnum.IDLE, AgentStateEnum.ROUTING)

    def test_invalid_idle_to_complete(self) -> None:
        assert not Transition.is_valid(AgentStateEnum.IDLE, AgentStateEnum.COMPLETE)

    def test_terminal_states_have_no_transitions(self) -> None:
        assert len(VALID_TRANSITIONS[AgentStateEnum.COMPLETE]) == 0
        assert len(VALID_TRANSITIONS[AgentStateEnum.ERROR]) == 0

    def test_validate_returns_result(self) -> None:
        result = Transition.validate(AgentStateEnum.IDLE, AgentStateEnum.ROUTING)
        assert result.success is True
        assert result.error is None

    def test_validate_invalid_returns_error(self) -> None:
        result = Transition.validate(AgentStateEnum.IDLE, AgentStateEnum.COMPLETE)
        assert result.success is False
        assert result.error is not None

    def test_all_non_terminal_states_can_reach_error(self) -> None:
        for state, allowed in VALID_TRANSITIONS.items():
            if state in (AgentStateEnum.IDLE, AgentStateEnum.COMPLETE, AgentStateEnum.ERROR):
                continue
            assert AgentStateEnum.ERROR in allowed, f"{state} cannot transition to ERROR"


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------


def _make_machine(**kwargs: Any) -> AgentStateMachine:
    defaults = {
        "router": StubRouter(),
        "planner": StubPlanner(),
        "executor": StubExecutor(),
        "adapter": StubAdapter(),
    }
    defaults.update(kwargs)
    return AgentStateMachine(**defaults)


class TestStateMachine:
    async def test_direct_response_flow(self) -> None:
        """IDLE → ROUTING → RESPONDING → COMPLETE (direct strategy)."""
        machine = _make_machine(router=StubRouter(strategy="direct"))
        result = await machine.process_message("Hello")

        assert result.state == AgentStateEnum.COMPLETE
        assert result.response == "Hello from Aria Core!"
        assert result.error is None

    async def test_plan_execution_flow(self) -> None:
        """IDLE → ROUTING → PLANNING → EXECUTING_STEP(×2) → RESPONDING → COMPLETE."""
        executor = StubExecutor()
        machine = _make_machine(
            router=StubRouter(strategy="plan"),
            planner=StubPlanner(num_steps=2),
            executor=executor,
        )
        result = await machine.process_message("Do something complex")

        assert result.state == AgentStateEnum.COMPLETE
        assert result.error is None
        assert len(executor.executed_steps) == 2

    async def test_empty_plan_skips_to_responding(self) -> None:
        """Plan with 0 steps goes straight to RESPONDING."""
        machine = _make_machine(
            router=StubRouter(strategy="plan"),
            planner=StubPlanner(num_steps=0),
        )
        result = await machine.process_message("Simple plan")

        assert result.state == AgentStateEnum.COMPLETE

    async def test_max_steps_enforced(self) -> None:
        """Exceeding max_steps produces an error."""
        config = AgentConfig(max_steps=1)
        machine = _make_machine(
            router=StubRouter(strategy="plan"),
            planner=StubPlanner(num_steps=5),
            config=config,
        )
        result = await machine.process_message("Too many steps")

        assert result.state == AgentStateEnum.ERROR
        assert "Maximum" in (result.error or "")

    async def test_event_callback_fires(self) -> None:
        """Event callback receives events during execution."""
        events: list[tuple[str, dict[str, Any]]] = []

        async def capture(event_type: str, payload: dict[str, Any]) -> None:
            events.append((event_type, payload))

        machine = _make_machine(
            router=StubRouter(strategy="direct"),
            event_callback=capture,
        )
        await machine.process_message("Fire events")

        event_types = [e[0] for e in events]
        assert "agent.start" in event_types
        assert "routing.start" in event_types
        assert "agent.complete" in event_types

    async def test_reset_clears_state(self) -> None:
        machine = _make_machine()
        await machine.process_message("Run once")
        machine.reset()

        assert machine.current_state == AgentStateEnum.IDLE
        assert machine.transition_history == []

    async def test_adapter_error_transitions_to_error(self) -> None:
        """Exception in adapter lands in ERROR state."""

        class BrokenAdapter:
            async def generate_response(self, context: AgentContext) -> str:
                raise RuntimeError("LLM is down")

        machine = _make_machine(
            router=StubRouter(strategy="direct"),
            adapter=BrokenAdapter(),
        )
        result = await machine.process_message("Fail please")

        assert result.state == AgentStateEnum.ERROR
        assert "LLM is down" in (result.error or "")

    async def test_process_message_creates_context(self) -> None:
        machine = _make_machine()
        result = await machine.process_message("Hello", conversation_id=uuid4())

        assert result.context.messages[0].content == "Hello"
        assert result.context.messages[0].role == MessageRole.USER


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_agent_config_defaults(self) -> None:
        config = AgentConfig()
        assert config.max_steps == 10
        assert config.timeout_seconds == 300

    def test_agent_config_validation(self) -> None:
        with pytest.raises(Exception):
            AgentConfig(max_steps=0)

    def test_agent_result_response_property(self) -> None:
        from aria_core.runtime.models import AgentResult

        ctx = AgentContext(metadata={"response": "test"})
        result = AgentResult(context=ctx, state=AgentStateEnum.COMPLETE)
        assert result.response == "test"

    def test_chat_message(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        assert msg.role == MessageRole.USER
        assert msg.content == "hi"
