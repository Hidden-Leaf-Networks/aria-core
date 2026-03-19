"""Plan engine — orchestrates plan lifecycle and action execution.

Manages plan creation, validation, dependency resolution, risk assessment
integration, and sequential/parallel action execution.

Usage:
    from aria_core.planning import PlanEngine

    engine = PlanEngine()
    plan = engine.create_plan(name="Deploy", actions=[
        {"name": "Build", "skill_name": "build_project"},
        {"name": "Test", "skill_name": "run_tests", "dependencies": [0]},
        {"name": "Deploy", "skill_name": "deploy", "dependencies": [1]},
    ])
    plan = engine.validate_plan(plan.id)
    plan = engine.start_plan(plan.id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from aria_core.planning.models import (
    ActionState,
    Plan,
    PlanAction,
    PlanState,
    PlanVersion,
    VALID_ACTION_TRANSITIONS,
    VALID_PLAN_TRANSITIONS,
)

logger = logging.getLogger("aria_core.planning")


class PlanEngineError(Exception):
    pass


class PlanNotFoundError(PlanEngineError):
    def __init__(self, plan_id: UUID):
        self.plan_id = plan_id
        super().__init__(f"Plan {plan_id} not found")


class PlanStateError(PlanEngineError):
    def __init__(self, plan_id: UUID, state: str, operation: str):
        super().__init__(f"Cannot {operation} plan {plan_id} in state '{state}'")


class DependencyError(PlanEngineError):
    def __init__(self, action_id: UUID, missing_deps: list[int]):
        super().__init__(f"Action {action_id} has unsatisfied dependencies: {missing_deps}")


class ExecutionResult:
    """Result of a single action execution."""

    def __init__(
        self,
        success: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        execution_time_ms: int | None = None,
    ):
        self.success = success
        self.result = result
        self.error = error
        self.execution_time_ms = execution_time_ms


# Callback type for executing skills
SkillExecutor = Callable[[str, dict[str, Any] | None], Awaitable[ExecutionResult]]


class PlanEngine:
    """Orchestrator for plan lifecycle and action execution.

    Manages: plan CRUD, dependency validation, state transitions,
    risk integration, and action execution through skill callbacks.
    """

    def __init__(
        self,
        skill_executor: SkillExecutor | None = None,
        event_callback: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._skill_executor = skill_executor
        self._event_callback = event_callback
        self._plans: dict[UUID, Plan] = {}

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback:
            self._event_callback(event_type, payload)

    # -----------------------------------------------------------------------
    # Plan lifecycle
    # -----------------------------------------------------------------------

    def create_plan(
        self,
        name: str,
        actions: list[dict[str, Any]],
        description: str = "",
        conversation_id: UUID | None = None,
        prompt: str | None = None,
        created_by: str = "system",
    ) -> Plan:
        """Create a plan in DRAFT state."""
        plan_id = uuid4()
        now = datetime.now(timezone.utc)

        plan_actions: list[PlanAction] = []
        for idx, action_data in enumerate(actions):
            action = PlanAction(
                plan_id=plan_id,
                index=idx,
                name=action_data.get("name", f"Action {idx + 1}"),
                description=action_data.get("description", ""),
                skill_name=action_data.get("skill_name"),
                skill_args=action_data.get("skill_args"),
                dependencies=action_data.get("dependencies", []),
                state=ActionState.PENDING,
                created_at=now,
            )
            plan_actions.append(action)

        plan = Plan(
            id=plan_id,
            name=name,
            description=description,
            conversation_id=conversation_id,
            prompt=prompt,
            state=PlanState.DRAFT,
            actions=plan_actions,
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

        self._plans[plan_id] = plan
        self._emit("plan.created", {"plan_id": str(plan_id), "name": name})
        return plan

    def get_plan(self, plan_id: UUID) -> Plan | None:
        return self._plans.get(plan_id)

    def validate_plan(self, plan_id: UUID) -> Plan:
        """Validate and transition plan from DRAFT to PLANNED."""
        plan = self._require_plan(plan_id)

        if plan.state != PlanState.DRAFT:
            raise PlanStateError(plan_id, plan.state.value, "validate")

        if not plan.actions:
            raise PlanEngineError(f"Plan {plan_id} has no actions")

        self._validate_dependencies(plan)

        updated = plan.model_copy(update={
            "state": PlanState.PLANNED,
            "planned_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
        self._plans[plan_id] = updated
        self._emit("plan.validated", {"plan_id": str(plan_id)})
        return updated

    def start_plan(self, plan_id: UUID) -> Plan:
        """Transition plan to EXECUTING (queued → executing)."""
        plan = self._require_plan(plan_id)

        if plan.state == PlanState.PLANNED:
            plan = plan.model_copy(update={"state": PlanState.QUEUED, "updated_at": datetime.now(timezone.utc)})

        if plan.state != PlanState.QUEUED:
            raise PlanStateError(plan_id, plan.state.value, "start")

        updated = plan.model_copy(update={
            "state": PlanState.EXECUTING,
            "started_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
        self._plans[plan_id] = updated
        self._emit("plan.started", {"plan_id": str(plan_id)})
        return updated

    async def execute_next_action(self, plan_id: UUID) -> Plan:
        """Execute the next ready action in the plan.

        An action is ready when all its dependencies are completed.
        """
        plan = self._require_plan(plan_id)

        if plan.state != PlanState.EXECUTING:
            raise PlanStateError(plan_id, plan.state.value, "execute action")

        # Find next ready action
        action_idx = self._find_next_ready_action(plan)
        if action_idx is None:
            # Check if all done
            if all(a.state in (ActionState.COMPLETED, ActionState.SKIPPED) for a in plan.actions):
                return self._complete_plan(plan)
            if any(a.state == ActionState.FAILED for a in plan.actions):
                return self._fail_plan(plan, "Action failed")
            raise PlanEngineError(f"No ready actions in plan {plan_id}")

        action = plan.actions[action_idx]

        if not self._skill_executor:
            raise PlanEngineError("No skill executor configured")

        # Execute
        now = datetime.now(timezone.utc)
        action = action.model_copy(update={
            "state": ActionState.EXECUTING,
            "started_at": now,
        })

        try:
            result = await self._skill_executor(
                action.skill_name or action.name,
                action.skill_args,
            )

            if result.success:
                action = action.model_copy(update={
                    "state": ActionState.COMPLETED,
                    "result": result.result,
                    "completed_at": datetime.now(timezone.utc),
                    "execution_time_ms": result.execution_time_ms,
                })
            else:
                action = action.model_copy(update={
                    "state": ActionState.FAILED,
                    "error": result.error,
                    "completed_at": datetime.now(timezone.utc),
                    "execution_time_ms": result.execution_time_ms,
                })

        except Exception as e:
            action = action.model_copy(update={
                "state": ActionState.FAILED,
                "error": str(e),
                "completed_at": datetime.now(timezone.utc),
            })

        # Update plan with modified action
        actions = list(plan.actions)
        actions[action_idx] = action
        plan = plan.model_copy(update={
            "actions": actions,
            "current_action_index": action_idx,
            "updated_at": datetime.now(timezone.utc),
        })
        self._plans[plan_id] = plan

        self._emit("action.completed", {
            "plan_id": str(plan_id),
            "action_id": str(action.id),
            "success": action.state == ActionState.COMPLETED,
        })

        # Auto-complete if all done
        if all(a.state in (ActionState.COMPLETED, ActionState.SKIPPED) for a in plan.actions):
            plan = self._complete_plan(plan)

        return plan

    async def execute_all(self, plan_id: UUID) -> Plan:
        """Execute all actions in dependency order until done or failed."""
        plan = self._require_plan(plan_id)

        if plan.state in (PlanState.DRAFT, PlanState.PLANNED):
            plan = self.validate_plan(plan_id) if plan.state == PlanState.DRAFT else plan
            plan = self.start_plan(plan_id)

        while plan.state == PlanState.EXECUTING:
            next_idx = self._find_next_ready_action(plan)
            if next_idx is None:
                break
            plan = await self.execute_next_action(plan_id)

        # Final state check
        if plan.state == PlanState.EXECUTING:
            if all(a.state in (ActionState.COMPLETED, ActionState.SKIPPED) for a in plan.actions):
                plan = self._complete_plan(plan)
            elif any(a.state == ActionState.FAILED for a in plan.actions):
                plan = self._fail_plan(plan, "One or more actions failed")

        return plan

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _require_plan(self, plan_id: UUID) -> Plan:
        plan = self.get_plan(plan_id)
        if not plan:
            raise PlanNotFoundError(plan_id)
        return plan

    def _validate_dependencies(self, plan: Plan) -> None:
        """Validate dependency graph: no cycles, valid indices."""
        n = len(plan.actions)
        for action in plan.actions:
            for dep in action.dependencies:
                if dep < 0 or dep >= n:
                    raise DependencyError(action.id, [dep])
                if dep == action.index:
                    raise DependencyError(action.id, [dep])

        # Check for cycles using DFS
        visited = [0] * n  # 0=unvisited, 1=in-progress, 2=done
        def dfs(idx: int) -> bool:
            if visited[idx] == 1:
                return True  # cycle
            if visited[idx] == 2:
                return False
            visited[idx] = 1
            for dep in plan.actions[idx].dependencies:
                if dfs(dep):
                    return True
            visited[idx] = 2
            return False

        for i in range(n):
            if dfs(i):
                raise PlanEngineError(f"Circular dependency detected in plan {plan.id}")

    def _find_next_ready_action(self, plan: Plan) -> int | None:
        """Find the next action whose dependencies are all completed."""
        for i, action in enumerate(plan.actions):
            if action.state != ActionState.PENDING:
                continue
            deps_met = all(
                plan.actions[dep].state in (ActionState.COMPLETED, ActionState.SKIPPED)
                for dep in action.dependencies
            )
            if deps_met:
                return i
        return None

    def _complete_plan(self, plan: Plan) -> Plan:
        updated = plan.model_copy(update={
            "state": PlanState.COMPLETED,
            "completed_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
        self._plans[plan.id] = updated
        self._emit("plan.completed", {"plan_id": str(plan.id)})
        return updated

    def _fail_plan(self, plan: Plan, error: str) -> Plan:
        updated = plan.model_copy(update={
            "state": PlanState.FAILED,
            "updated_at": datetime.now(timezone.utc),
            "metadata": {**plan.metadata, "error": error},
        })
        self._plans[plan.id] = updated
        self._emit("plan.failed", {"plan_id": str(plan.id), "error": error})
        return updated
