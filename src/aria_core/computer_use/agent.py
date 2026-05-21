"""Computer-use agent — simulated desktop automation.

All actions are simulated: no real mouse/keyboard events are sent.
A safety gate ensures destructive actions require explicit approval.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ScreenState(BaseModel):
    """Snapshot of the virtual desktop."""

    screenshot_data: Optional[bytes] = None
    width: int = 1920
    height: int = 1080
    active_window: str = "Desktop"
    cursor_position: tuple[int, int] = (0, 0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ComputerAction(BaseModel):
    """A single desktop action to execute."""

    action_type: Literal[
        "click",
        "double_click",
        "right_click",
        "type",
        "key_press",
        "scroll",
        "screenshot",
        "move",
        "launch_app",
        "wait",
    ]
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    key: Optional[str] = None
    app_name: Optional[str] = None
    duration_ms: Optional[int] = None
    requires_approval: bool = True


class ActionResult(BaseModel):
    """Outcome of an executed action."""

    action: ComputerAction
    success: bool
    screenshot_after: Optional[bytes] = None
    ocr_text: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0


# Actions considered destructive — always require approval when the
# agent-level approval gate is active.
_DESTRUCTIVE_ACTIONS: set[str] = {
    "click",
    "double_click",
    "right_click",
    "type",
    "key_press",
    "launch_app",
}


class ComputerUseAgent:
    """Orchestrates simulated desktop actions for a tenant.

    Parameters
    ----------
    tenant_id:
        Owning tenant UUID.
    require_approval:
        When True (default), destructive actions must be explicitly
        approved before execution.
    """

    def __init__(self, tenant_id: UUID, require_approval: bool = True) -> None:
        self.tenant_id = tenant_id
        self.require_approval = require_approval
        self._screen = ScreenState()
        self._history: list[ActionResult] = []
        self._approved_actions: set[int] = set()  # ids of approved actions

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def approve(self, action: ComputerAction) -> None:
        """Grant approval for a specific action instance."""
        self._approved_actions.add(id(action))

    def is_approved(self, action: ComputerAction) -> bool:
        """Check whether *action* passes the safety gate.

        Non-destructive actions (screenshot, scroll, move, wait) are
        always approved.  Destructive actions require either that the
        agent-level gate is disabled **or** that the action was
        explicitly approved via :meth:`approve` / the action's own
        ``requires_approval`` flag is False.
        """
        if action.action_type not in _DESTRUCTIVE_ACTIONS:
            return True
        if not self.require_approval:
            return True
        if not action.requires_approval:
            return True
        return id(action) in self._approved_actions

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_action(self, action: ComputerAction) -> ActionResult:
        """Execute a single simulated action."""
        start = time.monotonic()

        if not self.is_approved(action):
            result = ActionResult(
                action=action,
                success=False,
                error="Action requires approval before execution",
                duration_ms=0,
            )
            self._history.append(result)
            return result

        # Simulate the action
        error: str | None = None
        ocr_text: str | None = None
        try:
            if action.action_type in ("click", "double_click", "right_click", "move"):
                if action.x is not None and action.y is not None:
                    self._screen = self._screen.model_copy(
                        update={"cursor_position": (action.x, action.y)}
                    )
                else:
                    error = "x and y coordinates required"

            elif action.action_type == "type":
                ocr_text = action.text  # echo typed text as OCR result

            elif action.action_type == "key_press":
                ocr_text = f"[key: {action.key}]"

            elif action.action_type == "scroll":
                pass  # no-op in simulation

            elif action.action_type == "screenshot":
                ocr_text = f"Screenshot of {self._screen.active_window}"

            elif action.action_type == "launch_app":
                if action.app_name:
                    self._screen = self._screen.model_copy(
                        update={"active_window": action.app_name}
                    )
                else:
                    error = "app_name required for launch_app"

            elif action.action_type == "wait":
                pass  # simulated wait

        except Exception as exc:  # pragma: no cover
            error = str(exc)

        elapsed = int((time.monotonic() - start) * 1000)
        result = ActionResult(
            action=action,
            success=error is None,
            ocr_text=ocr_text,
            error=error,
            duration_ms=elapsed,
        )
        self._history.append(result)
        return result

    async def execute_sequence(
        self, actions: list[ComputerAction]
    ) -> list[ActionResult]:
        """Execute a sequence of actions in order, stopping on failure."""
        results: list[ActionResult] = []
        for action in actions:
            result = await self.execute_action(action)
            results.append(result)
            if not result.success:
                break
        return results

    async def take_screenshot(self) -> ScreenState:
        """Capture the current simulated screen state."""
        self._screen = self._screen.model_copy(
            update={
                "timestamp": datetime.now(timezone.utc),
                "screenshot_data": b"simulated-screenshot-png",
            }
        )
        return self._screen

    def get_screen_state(self) -> ScreenState:
        """Return the current screen state without capturing."""
        return self._screen

    async def find_element(self, description: str) -> tuple[int, int] | None:
        """Simulate finding a UI element by description.

        Returns a deterministic position derived from the description
        hash so tests are reproducible.
        """
        if not description:
            return None
        h = hash(description)
        x = abs(h) % self._screen.width
        y = abs(h >> 16) % self._screen.height
        return (x, y)

    def get_action_history(self, limit: int = 50) -> list[ActionResult]:
        """Return the most recent action results."""
        return list(self._history[-limit:])
