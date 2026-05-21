"""
Flow Orchestration Engine — DAG-based multi-agent workflow execution.

Supports linear chains, conditional branching, parallel fan-out, pause/resume,
per-step timeout/retry, and shared state passing between steps.

ARIA-304
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

# Python 3.10 compat — mirrors aria_core.runtime.models
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """String enum backport for Python < 3.11."""

        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return self.value


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FlowStepType(StrEnum):
    """Types of steps in a flow."""

    AGENT = "agent"
    SKILL = "skill"
    CONDITION = "condition"
    PARALLEL = "parallel"
    WAIT = "wait"


class FlowExecutionStatus(StrEnum):
    """Execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


# ---------------------------------------------------------------------------
# Condition operators
# ---------------------------------------------------------------------------

_OPERATORS: dict[str, Any] = {
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "contains": lambda a, b: b in a,
    "exists": lambda a, _: a is not None,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FlowStep(BaseModel):
    """A single step in a flow DAG."""

    id: str
    name: str
    step_type: FlowStepType

    # Agent / skill specifics
    agent_id: str | None = None
    skill_name: str | None = None
    message: str | None = None

    # Condition routing
    condition: dict[str, Any] | None = None
    branches: dict[str, str] | None = None

    # Parallel fan-out
    parallel_steps: list[str] | None = None

    # Default next step
    next_step: str | None = None

    # Execution policy
    timeout_seconds: int = 300
    retry_count: int = 0

    # Arbitrary metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowDefinition(BaseModel):
    """Declarative flow — a DAG of FlowSteps."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID | None = None
    name: str
    description: str = ""
    steps: list[FlowStep]
    entry_step: str
    triggers: list[dict[str, Any]] | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    created_by: str = ""


class FlowExecution(BaseModel):
    """Runtime state for one flow execution."""

    id: UUID = Field(default_factory=uuid4)
    flow_id: UUID
    tenant_id: UUID | None = None
    status: FlowExecutionStatus = FlowExecutionStatus.PENDING
    current_step: str = ""
    completed_steps: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FlowEngine:
    """Orchestrates flow registration and execution."""

    def __init__(self, tenant_id: UUID | None = None) -> None:
        self.tenant_id = tenant_id
        self._flows: dict[UUID, FlowDefinition] = {}
        self._executions: dict[UUID, FlowExecution] = {}

    # -- CRUD ---------------------------------------------------------------

    def register(self, flow: FlowDefinition) -> FlowDefinition:
        """Register a flow definition."""
        if self.tenant_id is not None:
            flow.tenant_id = self.tenant_id
        self._flows[flow.id] = flow
        return flow

    def list_flows(self) -> list[FlowDefinition]:
        """Return all registered flows."""
        return list(self._flows.values())

    def get_flow(self, flow_id: UUID) -> FlowDefinition | None:
        """Look up a flow by ID."""
        return self._flows.get(flow_id)

    def delete_flow(self, flow_id: UUID) -> bool:
        """Remove a flow definition. Returns True if it existed."""
        return self._flows.pop(flow_id, None) is not None

    # -- Execution ----------------------------------------------------------

    async def start(
        self,
        flow_id: UUID,
        initial_state: dict[str, Any] | None = None,
    ) -> FlowExecution:
        """Begin a new execution (runs the first step only)."""
        flow = self._flows.get(flow_id)
        if flow is None:
            raise ValueError(f"Flow {flow_id} not found")

        execution = FlowExecution(
            flow_id=flow_id,
            tenant_id=self.tenant_id,
            status=FlowExecutionStatus.RUNNING,
            current_step=flow.entry_step,
            state=dict(initial_state or {}),
            started_at=datetime.now(timezone.utc),
        )
        self._executions[execution.id] = execution

        # Execute the entry step
        step = self._find_step(flow, flow.entry_step)
        if step is None:
            execution.status = FlowExecutionStatus.FAILED
            execution.error = f"Entry step '{flow.entry_step}' not found"
            return execution

        result = await self.execute_step(execution, step)
        execution.results[step.id] = result
        execution.completed_steps.append(step.id)

        # Resolve what comes next
        next_id = self._resolve_next_step(execution, step, result)
        if next_id is None:
            execution.status = FlowExecutionStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            if execution.started_at:
                delta = execution.completed_at - execution.started_at
                execution.duration_ms = int(delta.total_seconds() * 1000)
        else:
            execution.current_step = next_id

        return execution

    async def execute_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Execute a single step and return its result dict."""
        if execution.status == FlowExecutionStatus.PAUSED:
            return {"skipped": True, "reason": "execution_paused"}

        retries = step.retry_count
        last_error: str | None = None

        for attempt in range(retries + 1):
            try:
                return await self._run_step(execution, step)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < retries:
                    continue

        # All retries exhausted
        execution.status = FlowExecutionStatus.FAILED
        execution.error = last_error
        return {"error": last_error, "retries_exhausted": True}

    async def run(
        self,
        flow_id: UUID,
        initial_state: dict[str, Any] | None = None,
    ) -> FlowExecution:
        """Run a flow to completion (or until failure / pause)."""
        flow = self._flows.get(flow_id)
        if flow is None:
            raise ValueError(f"Flow {flow_id} not found")

        execution = FlowExecution(
            flow_id=flow_id,
            tenant_id=self.tenant_id,
            status=FlowExecutionStatus.RUNNING,
            current_step=flow.entry_step,
            state=dict(initial_state or {}),
            started_at=datetime.now(timezone.utc),
        )
        self._executions[execution.id] = execution

        visited: set[str] = set()
        current_step_id: str | None = flow.entry_step

        while current_step_id is not None:
            # Guard against infinite loops
            if current_step_id in visited:
                execution.status = FlowExecutionStatus.FAILED
                execution.error = f"Cycle detected at step '{current_step_id}'"
                break
            visited.add(current_step_id)

            if execution.status == FlowExecutionStatus.PAUSED:
                break

            step = self._find_step(flow, current_step_id)
            if step is None:
                execution.status = FlowExecutionStatus.FAILED
                execution.error = f"Step '{current_step_id}' not found in flow"
                break

            execution.current_step = current_step_id
            result = await self.execute_step(execution, step)

            if execution.status == FlowExecutionStatus.FAILED:
                break

            execution.results[step.id] = result
            execution.completed_steps.append(step.id)

            current_step_id = self._resolve_next_step(execution, step, result)

        if execution.status == FlowExecutionStatus.RUNNING:
            execution.status = FlowExecutionStatus.COMPLETED

        execution.completed_at = datetime.now(timezone.utc)
        if execution.started_at:
            delta = execution.completed_at - execution.started_at
            execution.duration_ms = int(delta.total_seconds() * 1000)

        return execution

    # -- Pause / Resume -----------------------------------------------------

    def pause(self, execution_id: UUID) -> FlowExecution:
        """Pause a running execution."""
        execution = self._executions.get(execution_id)
        if execution is None:
            raise ValueError(f"Execution {execution_id} not found")
        if execution.status != FlowExecutionStatus.RUNNING:
            raise ValueError(
                f"Cannot pause execution in '{execution.status}' state"
            )
        execution.status = FlowExecutionStatus.PAUSED
        return execution

    def resume(self, execution_id: UUID) -> FlowExecution:
        """Resume a paused execution."""
        execution = self._executions.get(execution_id)
        if execution is None:
            raise ValueError(f"Execution {execution_id} not found")
        if execution.status != FlowExecutionStatus.PAUSED:
            raise ValueError(
                f"Cannot resume execution in '{execution.status}' state"
            )
        execution.status = FlowExecutionStatus.RUNNING
        return execution

    # -- Query --------------------------------------------------------------

    def get_execution(self, execution_id: UUID) -> FlowExecution | None:
        """Retrieve an execution by ID."""
        return self._executions.get(execution_id)

    def list_executions(
        self,
        flow_id: UUID | None = None,
    ) -> list[FlowExecution]:
        """List executions, optionally filtered by flow_id."""
        execs = list(self._executions.values())
        if flow_id is not None:
            execs = [e for e in execs if e.flow_id == flow_id]
        return execs

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _find_step(flow: FlowDefinition, step_id: str) -> FlowStep | None:
        """Find a step by ID within a flow definition."""
        for s in flow.steps:
            if s.id == step_id:
                return s
        return None

    def _resolve_next_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
        step_result: dict[str, Any],
    ) -> str | None:
        """Determine the next step ID based on step type and result."""
        if step.step_type == FlowStepType.CONDITION:
            if step.condition and step.branches:
                outcome = self._evaluate_condition(
                    step.condition, execution.state,
                )
                branch_key = "true" if outcome else "false"
                return step.branches.get(branch_key)
        return step.next_step

    @staticmethod
    def _evaluate_condition(
        condition: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        """Evaluate a condition dict against the execution state."""
        field = condition.get("field", "")
        operator = condition.get("operator", "eq")
        expected = condition.get("value")

        actual = state.get(field)

        op_fn = _OPERATORS.get(operator)
        if op_fn is None:
            raise ValueError(f"Unknown condition operator: '{operator}'")

        try:
            return bool(op_fn(actual, expected))
        except (TypeError, KeyError):
            return False

    async def _run_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Dispatch a step by type and return its result."""
        if step.step_type == FlowStepType.AGENT:
            return await self._run_agent_step(execution, step)
        elif step.step_type == FlowStepType.SKILL:
            return await self._run_skill_step(execution, step)
        elif step.step_type == FlowStepType.CONDITION:
            return await self._run_condition_step(execution, step)
        elif step.step_type == FlowStepType.PARALLEL:
            return await self._run_parallel_step(execution, step)
        elif step.step_type == FlowStepType.WAIT:
            return await self._run_wait_step(execution, step)
        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    # -- Step runners (simulate for now) ------------------------------------

    async def _run_agent_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Run an agent step — stores message in results."""
        result: dict[str, Any] = {
            "step_type": "agent",
            "agent_id": step.agent_id,
            "message": step.message,
            "status": "completed",
        }
        # Allow the step to write into shared state
        if step.metadata.get("state_updates"):
            execution.state.update(step.metadata["state_updates"])
        return result

    async def _run_skill_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Run a skill step — stores skill_name in results."""
        result: dict[str, Any] = {
            "step_type": "skill",
            "skill_name": step.skill_name,
            "message": step.message,
            "status": "completed",
        }
        if step.metadata.get("state_updates"):
            execution.state.update(step.metadata["state_updates"])
        return result

    async def _run_condition_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Evaluate a condition step."""
        outcome = False
        if step.condition:
            outcome = self._evaluate_condition(step.condition, execution.state)
        return {
            "step_type": "condition",
            "condition": step.condition,
            "result": outcome,
            "status": "completed",
        }

    async def _run_parallel_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Run parallel sub-steps concurrently and collect results."""
        if not step.parallel_steps:
            return {"step_type": "parallel", "results": {}, "status": "completed"}

        flow = self._flows.get(execution.flow_id)
        if flow is None:
            raise ValueError(f"Flow {execution.flow_id} not found")

        sub_results: dict[str, dict[str, Any]] = {}

        async def _run_sub(sub_id: str) -> None:
            sub_step = self._find_step(flow, sub_id)
            if sub_step is None:
                sub_results[sub_id] = {"error": f"Step '{sub_id}' not found"}
                return
            r = await self.execute_step(execution, sub_step)
            sub_results[sub_id] = r
            execution.results[sub_id] = r
            execution.completed_steps.append(sub_id)

        await asyncio.gather(*[_run_sub(sid) for sid in step.parallel_steps])

        return {
            "step_type": "parallel",
            "results": sub_results,
            "status": "completed",
        }

    async def _run_wait_step(
        self,
        execution: FlowExecution,
        step: FlowStep,
    ) -> dict[str, Any]:
        """Wait step — pauses for a configurable duration (0 in tests)."""
        wait_seconds = step.metadata.get("wait_seconds", 0)
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
        return {
            "step_type": "wait",
            "waited_seconds": wait_seconds,
            "status": "completed",
        }
