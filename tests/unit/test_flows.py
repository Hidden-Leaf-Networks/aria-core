"""
Tests for Flow Orchestration Engine — ARIA-304.

25+ tests covering: flow definition, linear execution, conditional branching,
parallel steps, state passing, pause/resume, error handling, CRUD, and
condition operators.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from aria_core.flows import (
    FlowDefinition,
    FlowEngine,
    FlowExecution,
    FlowExecutionStatus,
    FlowStep,
    FlowStepType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_step(
    id: str,
    name: str = "",
    message: str = "hello",
    next_step: str | None = None,
    **kw,
) -> FlowStep:
    return FlowStep(
        id=id,
        name=name or id,
        step_type=FlowStepType.AGENT,
        agent_id=f"agent-{id}",
        message=message,
        next_step=next_step,
        **kw,
    )


def _skill_step(
    id: str,
    skill_name: str = "echo",
    next_step: str | None = None,
    **kw,
) -> FlowStep:
    return FlowStep(
        id=id,
        name=id,
        step_type=FlowStepType.SKILL,
        skill_name=skill_name,
        next_step=next_step,
        **kw,
    )


def _condition_step(
    id: str,
    field: str,
    operator: str,
    value,
    true_step: str,
    false_step: str,
    **kw,
) -> FlowStep:
    return FlowStep(
        id=id,
        name=id,
        step_type=FlowStepType.CONDITION,
        condition={"field": field, "operator": operator, "value": value},
        branches={"true": true_step, "false": false_step},
        **kw,
    )


def _make_flow(steps: list[FlowStep], entry: str, name: str = "test") -> FlowDefinition:
    return FlowDefinition(name=name, steps=steps, entry_step=entry)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestFlowStepModel:
    def test_create_agent_step(self):
        s = _agent_step("s1")
        assert s.step_type == FlowStepType.AGENT
        assert s.agent_id == "agent-s1"
        assert s.timeout_seconds == 300
        assert s.retry_count == 0
        assert s.metadata == {}

    def test_create_skill_step(self):
        s = _skill_step("s1", skill_name="summarize")
        assert s.step_type == FlowStepType.SKILL
        assert s.skill_name == "summarize"

    def test_create_condition_step(self):
        s = _condition_step("c1", "status", "eq", "done", "a", "b")
        assert s.condition == {"field": "status", "operator": "eq", "value": "done"}
        assert s.branches == {"true": "a", "false": "b"}

    def test_step_metadata(self):
        s = _agent_step("s1", metadata={"key": "val"})
        assert s.metadata["key"] == "val"

    def test_parallel_step(self):
        s = FlowStep(
            id="p1",
            name="parallel",
            step_type=FlowStepType.PARALLEL,
            parallel_steps=["a", "b", "c"],
        )
        assert s.parallel_steps == ["a", "b", "c"]


class TestFlowDefinitionModel:
    def test_create_flow(self):
        flow = _make_flow([_agent_step("s1")], "s1", name="My Flow")
        assert isinstance(flow.id, UUID)
        assert flow.name == "My Flow"
        assert flow.entry_step == "s1"
        assert flow.created_at is not None

    def test_flow_default_values(self):
        flow = _make_flow([_agent_step("s1")], "s1")
        assert flow.description == ""
        assert flow.tenant_id is None
        assert flow.triggers is None
        assert flow.created_by == ""

    def test_flow_with_triggers(self):
        flow = FlowDefinition(
            name="triggered",
            steps=[_agent_step("s1")],
            entry_step="s1",
            triggers=[{"event": "on_message", "channel": "#general"}],
        )
        assert len(flow.triggers) == 1


class TestFlowExecutionModel:
    def test_default_status(self):
        ex = FlowExecution(flow_id=uuid4())
        assert ex.status == FlowExecutionStatus.PENDING
        assert ex.completed_steps == []
        assert ex.state == {}
        assert ex.results == {}


# ---------------------------------------------------------------------------
# Engine CRUD
# ---------------------------------------------------------------------------

class TestFlowEngineCRUD:
    def test_register_and_get(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        assert engine.get_flow(flow.id) is flow

    def test_list_flows(self):
        engine = FlowEngine()
        engine.register(_make_flow([_agent_step("s1")], "s1", name="A"))
        engine.register(_make_flow([_agent_step("s2")], "s2", name="B"))
        assert len(engine.list_flows()) == 2

    def test_delete_flow(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        assert engine.delete_flow(flow.id) is True
        assert engine.get_flow(flow.id) is None

    def test_delete_nonexistent(self):
        engine = FlowEngine()
        assert engine.delete_flow(uuid4()) is False

    def test_tenant_id_assigned(self):
        tid = uuid4()
        engine = FlowEngine(tenant_id=tid)
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        assert flow.tenant_id == tid


# ---------------------------------------------------------------------------
# Linear execution
# ---------------------------------------------------------------------------

class TestLinearExecution:
    @pytest.mark.asyncio
    async def test_single_step_flow(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.COMPLETED
        assert "s1" in ex.completed_steps
        assert ex.results["s1"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_two_step_chain(self):
        engine = FlowEngine()
        steps = [
            _agent_step("s1", next_step="s2"),
            _agent_step("s2"),
        ]
        flow = _make_flow(steps, "s1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.COMPLETED
        assert ex.completed_steps == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_three_step_chain(self):
        engine = FlowEngine()
        steps = [
            _agent_step("a", next_step="b"),
            _skill_step("b", next_step="c"),
            _agent_step("c"),
        ]
        flow = _make_flow(steps, "a")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.COMPLETED
        assert len(ex.completed_steps) == 3
        assert ex.results["b"]["step_type"] == "skill"

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.duration_ms is not None
        assert ex.duration_ms >= 0


# ---------------------------------------------------------------------------
# Conditional branching
# ---------------------------------------------------------------------------

class TestConditionalBranching:
    @pytest.mark.asyncio
    async def test_condition_true_branch(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "status", "eq", "ready", "yes", "no"),
            _agent_step("yes", message="took true path"),
            _agent_step("no", message="took false path"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"status": "ready"})
        assert ex.status == FlowExecutionStatus.COMPLETED
        assert "yes" in ex.completed_steps
        assert "no" not in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_false_branch(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "status", "eq", "ready", "yes", "no"),
            _agent_step("yes"),
            _agent_step("no"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"status": "not_ready"})
        assert "no" in ex.completed_steps
        assert "yes" not in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_neq(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "val", "neq", "x", "a", "b"),
            _agent_step("a"),
            _agent_step("b"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"val": "y"})
        assert "a" in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_gt(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "score", "gt", 50, "high", "low"),
            _agent_step("high"),
            _agent_step("low"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"score": 80})
        assert "high" in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_contains(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "tags", "contains", "vip", "a", "b"),
            _agent_step("a"),
            _agent_step("b"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"tags": ["vip", "new"]})
        assert "a" in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_exists_true(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "token", "exists", None, "a", "b"),
            _agent_step("a"),
            _agent_step("b"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"token": "abc"})
        assert "a" in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_exists_false(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "token", "exists", None, "a", "b"),
            _agent_step("a"),
            _agent_step("b"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={})
        assert "b" in ex.completed_steps

    @pytest.mark.asyncio
    async def test_condition_lte(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "count", "lte", 10, "ok", "over"),
            _agent_step("ok"),
            _agent_step("over"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"count": 10})
        assert "ok" in ex.completed_steps


# ---------------------------------------------------------------------------
# Parallel steps
# ---------------------------------------------------------------------------

class TestParallelSteps:
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        engine = FlowEngine()
        steps = [
            FlowStep(
                id="p1",
                name="parallel",
                step_type=FlowStepType.PARALLEL,
                parallel_steps=["a", "b"],
            ),
            _agent_step("a", message="agent a"),
            _skill_step("b", skill_name="echo"),
        ]
        flow = _make_flow(steps, "p1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.COMPLETED
        assert "a" in ex.results
        assert "b" in ex.results
        assert ex.results["a"]["step_type"] == "agent"
        assert ex.results["b"]["step_type"] == "skill"

    @pytest.mark.asyncio
    async def test_parallel_empty(self):
        engine = FlowEngine()
        steps = [
            FlowStep(
                id="p1",
                name="parallel-empty",
                step_type=FlowStepType.PARALLEL,
                parallel_steps=[],
            ),
        ]
        flow = _make_flow(steps, "p1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.COMPLETED


# ---------------------------------------------------------------------------
# State passing
# ---------------------------------------------------------------------------

class TestStatePassing:
    @pytest.mark.asyncio
    async def test_initial_state_available(self):
        engine = FlowEngine()
        steps = [
            _condition_step("c1", "x", "eq", 42, "yes", "no"),
            _agent_step("yes"),
            _agent_step("no"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id, initial_state={"x": 42})
        assert "yes" in ex.completed_steps

    @pytest.mark.asyncio
    async def test_state_updates_between_steps(self):
        engine = FlowEngine()
        steps = [
            _agent_step(
                "s1",
                next_step="c1",
                metadata={"state_updates": {"status": "done"}},
            ),
            _condition_step("c1", "status", "eq", "done", "s2", "s3"),
            _agent_step("s2"),
            _agent_step("s3"),
        ]
        flow = _make_flow(steps, "s1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert "s2" in ex.completed_steps
        assert ex.state["status"] == "done"


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------

class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_running_execution(self):
        engine = FlowEngine()
        flow = _make_flow(
            [_agent_step("s1", next_step="s2"), _agent_step("s2")],
            "s1",
        )
        engine.register(flow)
        ex = await engine.start(flow.id)
        # After start, first step done, execution still running
        assert ex.status == FlowExecutionStatus.RUNNING
        engine.pause(ex.id)
        assert ex.status == FlowExecutionStatus.PAUSED

    @pytest.mark.asyncio
    async def test_resume_paused_execution(self):
        engine = FlowEngine()
        flow = _make_flow(
            [_agent_step("s1", next_step="s2"), _agent_step("s2")],
            "s1",
        )
        engine.register(flow)
        ex = await engine.start(flow.id)
        engine.pause(ex.id)
        engine.resume(ex.id)
        assert ex.status == FlowExecutionStatus.RUNNING

    def test_pause_non_running_raises(self):
        engine = FlowEngine()
        ex = FlowExecution(flow_id=uuid4(), status=FlowExecutionStatus.COMPLETED)
        engine._executions[ex.id] = ex
        with pytest.raises(ValueError, match="Cannot pause"):
            engine.pause(ex.id)

    def test_resume_non_paused_raises(self):
        engine = FlowEngine()
        ex = FlowExecution(flow_id=uuid4(), status=FlowExecutionStatus.RUNNING)
        engine._executions[ex.id] = ex
        with pytest.raises(ValueError, match="Cannot resume"):
            engine.resume(ex.id)

    def test_pause_nonexistent_raises(self):
        engine = FlowEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.pause(uuid4())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_run_nonexistent_flow(self):
        engine = FlowEngine()
        with pytest.raises(ValueError, match="not found"):
            await engine.run(uuid4())

    @pytest.mark.asyncio
    async def test_start_nonexistent_flow(self):
        engine = FlowEngine()
        with pytest.raises(ValueError, match="not found"):
            await engine.start(uuid4())

    @pytest.mark.asyncio
    async def test_missing_entry_step(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "nonexistent")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.FAILED
        assert "not found" in ex.error

    @pytest.mark.asyncio
    async def test_missing_next_step(self):
        engine = FlowEngine()
        steps = [_agent_step("s1", next_step="missing")]
        flow = _make_flow(steps, "s1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.FAILED
        assert "not found" in ex.error

    @pytest.mark.asyncio
    async def test_invalid_condition_operator(self):
        engine = FlowEngine()
        steps = [
            FlowStep(
                id="c1",
                name="bad-cond",
                step_type=FlowStepType.CONDITION,
                condition={"field": "x", "operator": "bogus", "value": 1},
                branches={"true": "a", "false": "b"},
            ),
            _agent_step("a"),
            _agent_step("b"),
        ]
        flow = _make_flow(steps, "c1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.FAILED
        assert "bogus" in (ex.error or "")


# ---------------------------------------------------------------------------
# Execution queries
# ---------------------------------------------------------------------------

class TestExecutionQueries:
    @pytest.mark.asyncio
    async def test_get_execution(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        found = engine.get_execution(ex.id)
        assert found is not None
        assert found.id == ex.id

    @pytest.mark.asyncio
    async def test_list_executions(self):
        engine = FlowEngine()
        flow = _make_flow([_agent_step("s1")], "s1")
        engine.register(flow)
        await engine.run(flow.id)
        await engine.run(flow.id)
        assert len(engine.list_executions()) == 2

    @pytest.mark.asyncio
    async def test_list_executions_filtered(self):
        engine = FlowEngine()
        f1 = _make_flow([_agent_step("s1")], "s1", name="A")
        f2 = _make_flow([_agent_step("s2")], "s2", name="B")
        engine.register(f1)
        engine.register(f2)
        await engine.run(f1.id)
        await engine.run(f2.id)
        assert len(engine.list_executions(flow_id=f1.id)) == 1


# ---------------------------------------------------------------------------
# Wait step
# ---------------------------------------------------------------------------

class TestWaitStep:
    @pytest.mark.asyncio
    async def test_wait_step_completes(self):
        engine = FlowEngine()
        steps = [
            FlowStep(id="w1", name="wait", step_type=FlowStepType.WAIT),
        ]
        flow = _make_flow(steps, "w1")
        engine.register(flow)
        ex = await engine.run(flow.id)
        assert ex.status == FlowExecutionStatus.COMPLETED
        assert ex.results["w1"]["waited_seconds"] == 0
