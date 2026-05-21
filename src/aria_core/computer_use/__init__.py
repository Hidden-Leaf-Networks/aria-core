"""Computer-use agent — simulated desktop automation for ARIA agents.

Provides:
- ScreenState: snapshot of the virtual desktop
- ComputerAction: typed action (click, type, key_press, etc.)
- ActionResult: outcome of an executed action
- ComputerUseAgent: orchestrates actions with a safety-approval gate

Implements ARIA-313 computer-use agent capabilities.
"""

from aria_core.computer_use.agent import (
    ActionResult,
    ComputerAction,
    ComputerUseAgent,
    ScreenState,
)

__all__ = [
    "ActionResult",
    "ComputerAction",
    "ComputerUseAgent",
    "ScreenState",
]
