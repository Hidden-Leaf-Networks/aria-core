"""Agent state definitions.

Each state implements enter/execute/exit lifecycle hooks.
The state machine calls these in order as it transitions.

To add custom states, subclass State and register in STATE_REGISTRY.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aria_core.runtime.models import AgentStateEnum, AgentContext

if TYPE_CHECKING:
    from aria_core.runtime.state_machine import AgentStateMachine


class State(ABC):
    """Base class for agent states.

    Subclass this to define custom state behavior.
    """

    name: AgentStateEnum
    is_terminal: bool = False

    @abstractmethod
    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        """Called when entering this state."""
        ...

    @abstractmethod
    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        """Execute state logic and return next state."""
        ...

    async def exit(self, machine: AgentStateMachine, context: AgentContext) -> None:
        """Called when exiting this state. Override if cleanup needed."""
        pass


class IdleState(State):
    """Initial idle state — waiting for input."""

    name = AgentStateEnum.IDLE

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        pass

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        if context.messages:
            return AgentStateEnum.ROUTING
        return AgentStateEnum.IDLE


class RoutingState(State):
    """Classify intent and select execution strategy."""

    name = AgentStateEnum.ROUTING

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        await machine.emit_event("routing.start", {"context_id": str(context.id)})

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        route_result = await machine.router.route(context)
        context.metadata["route"] = route_result

        await machine.emit_event("routing.complete", {"route": route_result})

        if route_result.get("strategy") == "direct":
            return AgentStateEnum.RESPONDING
        return AgentStateEnum.PLANNING


class PlanningState(State):
    """Generate execution plan."""

    name = AgentStateEnum.PLANNING

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        await machine.emit_event("planning.start", {"context_id": str(context.id)})

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        plan = await machine.planner.create_plan(context)
        context.current_plan_id = plan.id
        context.metadata["plan"] = plan.model_dump()

        await machine.emit_event(
            "planning.complete", {"plan_id": str(plan.id), "steps": len(plan.steps)}
        )

        if plan.steps:
            return AgentStateEnum.EXECUTING_STEP
        return AgentStateEnum.RESPONDING


class ExecutingStepState(State):
    """Execute a single plan step."""

    name = AgentStateEnum.EXECUTING_STEP

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        await machine.emit_event(
            "step.start",
            {
                "step_index": context.current_step_index,
                "total_steps": context.step_count,
            },
        )

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        if context.step_count >= context.config.max_steps:
            context.metadata["error"] = "Maximum steps exceeded"
            return AgentStateEnum.ERROR

        result = await machine.executor.execute_step(context)
        context.step_count += 1

        if result.get("requires_approval"):
            return AgentStateEnum.AWAITING_APPROVAL

        await machine.emit_event(
            "step.complete",
            {
                "step_index": context.current_step_index,
                "success": result.get("success", True),
            },
        )

        context.current_step_index += 1

        plan_data = context.metadata.get("plan", {})
        if context.current_step_index < len(plan_data.get("steps", [])):
            return AgentStateEnum.EXECUTING_STEP

        return AgentStateEnum.RESPONDING


class AwaitingApprovalState(State):
    """Waiting for human approval of restricted action."""

    name = AgentStateEnum.AWAITING_APPROVAL

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        await machine.emit_event(
            "approval.required",
            {
                "step_index": context.current_step_index,
                "skill": context.metadata.get("pending_skill"),
            },
        )

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        approval = context.metadata.get("approval_granted")

        if approval is None:
            approval = False
            context.metadata["approval_granted"] = False

        if approval:
            await machine.emit_event(
                "approval.granted", {"step_index": context.current_step_index}
            )
            return AgentStateEnum.EXECUTING_STEP

        await machine.emit_event(
            "approval.denied", {"step_index": context.current_step_index}
        )
        context.current_step_index += 1
        context.metadata.pop("approval_granted", None)

        plan_data = context.metadata.get("plan", {})
        if context.current_step_index < len(plan_data.get("steps", [])):
            return AgentStateEnum.EXECUTING_STEP
        return AgentStateEnum.RESPONDING


class RespondingState(State):
    """Generate final response."""

    name = AgentStateEnum.RESPONDING

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        await machine.emit_event("responding.start", {"context_id": str(context.id)})

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        response = await machine.adapter.generate_response(context)
        context.metadata["response"] = response

        await machine.emit_event(
            "responding.complete", {"response_length": len(response)}
        )
        return AgentStateEnum.COMPLETE


class CompleteState(State):
    """Terminal state — execution finished successfully."""

    name = AgentStateEnum.COMPLETE
    is_terminal = True

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        await machine.emit_event(
            "agent.complete",
            {
                "context_id": str(context.id),
                "steps_executed": context.step_count,
            },
        )

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        return AgentStateEnum.COMPLETE


class ErrorState(State):
    """Terminal state — execution failed."""

    name = AgentStateEnum.ERROR
    is_terminal = True

    async def enter(self, machine: AgentStateMachine, context: AgentContext) -> None:
        error = context.metadata.get("error", "Unknown error")
        await machine.emit_event(
            "agent.error",
            {"context_id": str(context.id), "error": error},
        )

    async def execute(
        self, machine: AgentStateMachine, context: AgentContext
    ) -> AgentStateEnum:
        return AgentStateEnum.ERROR


# Registry of all built-in states
STATE_REGISTRY: dict[AgentStateEnum, type[State]] = {
    AgentStateEnum.IDLE: IdleState,
    AgentStateEnum.ROUTING: RoutingState,
    AgentStateEnum.PLANNING: PlanningState,
    AgentStateEnum.EXECUTING_STEP: ExecutingStepState,
    AgentStateEnum.AWAITING_APPROVAL: AwaitingApprovalState,
    AgentStateEnum.RESPONDING: RespondingState,
    AgentStateEnum.COMPLETE: CompleteState,
    AgentStateEnum.ERROR: ErrorState,
}
