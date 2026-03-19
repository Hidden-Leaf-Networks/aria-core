"""Tests for the plan engine."""

from __future__ import annotations

from typing import Any

import pytest

from aria_core.planning import (
    ActionState,
    DependencyError,
    ExecutionResult,
    Plan,
    PlanEngine,
    PlanEngineError,
    PlanNotFoundError,
    PlanState,
    PlanStateError,
)


# ---------------------------------------------------------------------------
# Stub executor
# ---------------------------------------------------------------------------


class StubExecutor:
    def __init__(self, fail_actions: set[str] | None = None) -> None:
        self.fail_actions = fail_actions or set()
        self.executed: list[str] = []

    async def __call__(self, skill_name: str, args: dict[str, Any] | None) -> ExecutionResult:
        self.executed.append(skill_name)
        if skill_name in self.fail_actions:
            return ExecutionResult(success=False, error=f"{skill_name} failed")
        return ExecutionResult(success=True, result={"output": f"{skill_name} done"}, execution_time_ms=10)


# ---------------------------------------------------------------------------
# Plan lifecycle
# ---------------------------------------------------------------------------


class TestPlanLifecycle:
    def test_create_plan(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(
            name="Test Plan",
            actions=[{"name": "Step 1", "skill_name": "do_thing"}],
        )
        assert plan.state == PlanState.DRAFT
        assert len(plan.actions) == 1

    def test_validate_plan(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Test", actions=[{"name": "A"}])
        validated = engine.validate_plan(plan.id)
        assert validated.state == PlanState.PLANNED
        assert validated.planned_at is not None

    def test_validate_empty_plan_raises(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Empty", actions=[])
        with pytest.raises(PlanEngineError, match="no actions"):
            engine.validate_plan(plan.id)

    def test_start_plan(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Test", actions=[{"name": "A"}])
        engine.validate_plan(plan.id)
        started = engine.start_plan(plan.id)
        assert started.state == PlanState.EXECUTING
        assert started.started_at is not None

    def test_start_from_draft_raises(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Test", actions=[{"name": "A"}])
        with pytest.raises(PlanStateError):
            engine.start_plan(plan.id)

    def test_plan_not_found(self) -> None:
        from uuid import uuid4
        engine = PlanEngine()
        with pytest.raises(PlanNotFoundError):
            engine.validate_plan(uuid4())


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


class TestDependencies:
    def test_valid_linear_deps(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Linear", actions=[
            {"name": "A"},
            {"name": "B", "dependencies": [0]},
            {"name": "C", "dependencies": [1]},
        ])
        validated = engine.validate_plan(plan.id)
        assert validated.state == PlanState.PLANNED

    def test_circular_dependency_raises(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Circular", actions=[
            {"name": "A", "dependencies": [1]},
            {"name": "B", "dependencies": [0]},
        ])
        with pytest.raises(PlanEngineError, match="Circular"):
            engine.validate_plan(plan.id)

    def test_invalid_dep_index_raises(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Bad", actions=[
            {"name": "A", "dependencies": [5]},
        ])
        with pytest.raises(DependencyError):
            engine.validate_plan(plan.id)

    def test_self_dependency_raises(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="Self", actions=[
            {"name": "A", "dependencies": [0]},
        ])
        with pytest.raises(DependencyError):
            engine.validate_plan(plan.id)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    async def test_execute_single_action(self) -> None:
        executor = StubExecutor()
        engine = PlanEngine(skill_executor=executor)
        plan = engine.create_plan(name="Single", actions=[
            {"name": "A", "skill_name": "do_a"},
        ])
        result = await engine.execute_all(plan.id)

        assert result.state == PlanState.COMPLETED
        assert result.actions[0].state == ActionState.COMPLETED
        assert executor.executed == ["do_a"]

    async def test_execute_chain(self) -> None:
        executor = StubExecutor()
        engine = PlanEngine(skill_executor=executor)
        plan = engine.create_plan(name="Chain", actions=[
            {"name": "Build", "skill_name": "build"},
            {"name": "Test", "skill_name": "test", "dependencies": [0]},
            {"name": "Deploy", "skill_name": "deploy", "dependencies": [1]},
        ])
        result = await engine.execute_all(plan.id)

        assert result.state == PlanState.COMPLETED
        assert executor.executed == ["build", "test", "deploy"]

    async def test_parallel_actions(self) -> None:
        """Actions without deps can execute in order (no deps = ready)."""
        executor = StubExecutor()
        engine = PlanEngine(skill_executor=executor)
        plan = engine.create_plan(name="Parallel", actions=[
            {"name": "A", "skill_name": "a"},
            {"name": "B", "skill_name": "b"},
            {"name": "C", "skill_name": "c", "dependencies": [0, 1]},
        ])
        result = await engine.execute_all(plan.id)

        assert result.state == PlanState.COMPLETED
        # A and B run before C
        assert executor.executed.index("c") > executor.executed.index("a")
        assert executor.executed.index("c") > executor.executed.index("b")

    async def test_action_failure_fails_plan(self) -> None:
        executor = StubExecutor(fail_actions={"fail_step"})
        engine = PlanEngine(skill_executor=executor)
        plan = engine.create_plan(name="Fail", actions=[
            {"name": "OK", "skill_name": "ok_step"},
            {"name": "Fail", "skill_name": "fail_step", "dependencies": [0]},
        ])
        result = await engine.execute_all(plan.id)

        assert result.state == PlanState.FAILED
        assert result.actions[1].state == ActionState.FAILED
        assert result.actions[1].error is not None

    async def test_no_executor_raises(self) -> None:
        engine = PlanEngine()
        plan = engine.create_plan(name="No Exec", actions=[
            {"name": "A", "skill_name": "something"},
        ])
        with pytest.raises(PlanEngineError, match="No skill executor"):
            await engine.execute_all(plan.id)

    async def test_progress_tracking(self) -> None:
        executor = StubExecutor()
        engine = PlanEngine(skill_executor=executor)
        plan = engine.create_plan(name="Progress", actions=[
            {"name": "A", "skill_name": "a"},
            {"name": "B", "skill_name": "b"},
        ])
        assert plan.progress == 0.0

        result = await engine.execute_all(plan.id)
        assert result.progress == 1.0


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    async def test_event_callback_fires(self) -> None:
        events: list[tuple[str, dict]] = []
        executor = StubExecutor()
        engine = PlanEngine(
            skill_executor=executor,
            event_callback=lambda t, p: events.append((t, p)),
        )
        plan = engine.create_plan(name="Events", actions=[
            {"name": "A", "skill_name": "a"},
        ])
        await engine.execute_all(plan.id)

        event_types = [e[0] for e in events]
        assert "plan.created" in event_types
        assert "plan.completed" in event_types
        assert "action.completed" in event_types
