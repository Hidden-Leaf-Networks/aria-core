"""Tests for computer-use agent (ARIA-313)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.computer_use.agent import (
    ActionResult,
    ComputerAction,
    ComputerUseAgent,
    ScreenState,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _agent(**kw) -> ComputerUseAgent:
    return ComputerUseAgent(tenant_id=uuid4(), **kw)


# ── ScreenState ───────────────────────────────────────────────────────

class TestScreenState:
    def test_defaults(self) -> None:
        s = ScreenState()
        assert s.width == 1920
        assert s.height == 1080
        assert s.active_window == "Desktop"
        assert s.cursor_position == (0, 0)
        assert s.screenshot_data is None

    def test_custom(self) -> None:
        s = ScreenState(width=800, height=600, active_window="Terminal")
        assert s.width == 800
        assert s.active_window == "Terminal"


# ── ComputerAction ────────────────────────────────────────────────────

class TestComputerAction:
    def test_click(self) -> None:
        a = ComputerAction(action_type="click", x=100, y=200)
        assert a.requires_approval is True

    def test_type_action(self) -> None:
        a = ComputerAction(action_type="type", text="hello")
        assert a.text == "hello"

    def test_all_action_types(self) -> None:
        types = [
            "click", "double_click", "right_click", "type",
            "key_press", "scroll", "screenshot", "move",
            "launch_app", "wait",
        ]
        for t in types:
            a = ComputerAction(action_type=t)
            assert a.action_type == t

    def test_invalid_action_type(self) -> None:
        with pytest.raises(Exception):
            ComputerAction(action_type="invalid")


# ── Safety Gate ───────────────────────────────────────────────────────

class TestSafetyGate:
    def test_destructive_blocked_by_default(self) -> None:
        agent = _agent(require_approval=True)
        action = ComputerAction(action_type="click", x=10, y=10)
        assert agent.is_approved(action) is False

    def test_non_destructive_always_approved(self) -> None:
        agent = _agent(require_approval=True)
        for t in ("screenshot", "scroll", "move", "wait"):
            action = ComputerAction(action_type=t)
            assert agent.is_approved(action) is True

    def test_approval_gate_disabled(self) -> None:
        agent = _agent(require_approval=False)
        action = ComputerAction(action_type="click", x=10, y=10)
        assert agent.is_approved(action) is True

    def test_explicit_approve(self) -> None:
        agent = _agent(require_approval=True)
        action = ComputerAction(action_type="click", x=10, y=10)
        agent.approve(action)
        assert agent.is_approved(action) is True

    def test_action_level_no_approval(self) -> None:
        agent = _agent(require_approval=True)
        action = ComputerAction(action_type="type", text="hi", requires_approval=False)
        assert agent.is_approved(action) is True


# ── Execution ─────────────────────────────────────────────────────────

class TestExecution:
    async def test_execute_unapproved_fails(self) -> None:
        agent = _agent()
        action = ComputerAction(action_type="click", x=10, y=20)
        result = await agent.execute_action(action)
        assert result.success is False
        assert "approval" in result.error

    async def test_execute_click_moves_cursor(self) -> None:
        agent = _agent(require_approval=False)
        action = ComputerAction(action_type="click", x=50, y=75)
        result = await agent.execute_action(action)
        assert result.success is True
        assert agent.get_screen_state().cursor_position == (50, 75)

    async def test_execute_type_echoes_text(self) -> None:
        agent = _agent(require_approval=False)
        action = ComputerAction(action_type="type", text="aria")
        result = await agent.execute_action(action)
        assert result.success is True
        assert result.ocr_text == "aria"

    async def test_execute_launch_app(self) -> None:
        agent = _agent(require_approval=False)
        action = ComputerAction(action_type="launch_app", app_name="Firefox")
        result = await agent.execute_action(action)
        assert result.success is True
        assert agent.get_screen_state().active_window == "Firefox"

    async def test_execute_launch_app_no_name(self) -> None:
        agent = _agent(require_approval=False)
        action = ComputerAction(action_type="launch_app")
        result = await agent.execute_action(action)
        assert result.success is False

    async def test_execute_click_no_coords(self) -> None:
        agent = _agent(require_approval=False)
        action = ComputerAction(action_type="click")
        result = await agent.execute_action(action)
        assert result.success is False

    async def test_execute_sequence(self) -> None:
        agent = _agent(require_approval=False)
        actions = [
            ComputerAction(action_type="launch_app", app_name="Terminal"),
            ComputerAction(action_type="type", text="ls"),
            ComputerAction(action_type="key_press", key="Enter"),
        ]
        results = await agent.execute_sequence(actions)
        assert len(results) == 3
        assert all(r.success for r in results)

    async def test_execute_sequence_stops_on_failure(self) -> None:
        agent = _agent(require_approval=False)
        actions = [
            ComputerAction(action_type="click"),  # no coords — fails
            ComputerAction(action_type="type", text="after"),
        ]
        results = await agent.execute_sequence(actions)
        assert len(results) == 1
        assert results[0].success is False


# ── Screenshot / Screen State ─────────────────────────────────────────

class TestScreenCapture:
    async def test_take_screenshot(self) -> None:
        agent = _agent()
        state = await agent.take_screenshot()
        assert state.screenshot_data is not None

    def test_get_screen_state(self) -> None:
        agent = _agent()
        state = agent.get_screen_state()
        assert state.width == 1920


# ── Find Element ──────────────────────────────────────────────────────

class TestFindElement:
    async def test_find_element(self) -> None:
        agent = _agent()
        pos = await agent.find_element("Submit button")
        assert pos is not None
        assert 0 <= pos[0] < 1920
        assert 0 <= pos[1] < 1080

    async def test_find_element_empty(self) -> None:
        agent = _agent()
        pos = await agent.find_element("")
        assert pos is None

    async def test_find_element_deterministic(self) -> None:
        agent = _agent()
        a = await agent.find_element("OK")
        b = await agent.find_element("OK")
        assert a == b


# ── History ───────────────────────────────────────────────────────────

class TestHistory:
    async def test_action_history(self) -> None:
        agent = _agent(require_approval=False)
        await agent.execute_action(ComputerAction(action_type="screenshot"))
        await agent.execute_action(ComputerAction(action_type="scroll"))
        history = agent.get_action_history()
        assert len(history) == 2

    async def test_action_history_limit(self) -> None:
        agent = _agent(require_approval=False)
        for _ in range(5):
            await agent.execute_action(ComputerAction(action_type="screenshot"))
        assert len(agent.get_action_history(limit=3)) == 3
