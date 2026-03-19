"""Routing strategies for intent classification.

Each strategy scores incoming messages and returns routing decisions.
The router selects the highest-scoring strategy.

Built-in strategies:
- DirectStrategy: Simple Q&A, knowledge lookups
- PlanStrategy: Complex multi-step tasks requiring tool use
- ClarifyStrategy: Ambiguous or empty input

Subclass RoutingStrategy to add custom routing logic.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aria_core.runtime.models import AgentContext

logger = logging.getLogger("aria_core.router")


@dataclass
class RouteResult:
    """Result of a routing decision."""

    strategy: str  # "direct", "plan", "clarify", or custom
    intent: str
    confidence: float
    tools_needed: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RoutingStrategy(ABC):
    """Base class for routing strategies.

    Subclass to implement custom routing logic. Each strategy needs:
    - name: Unique identifier for the strategy
    - should_use(): Return confidence score (0-1) that this strategy applies
    - get_route(): Return the full RouteResult if selected
    """

    name: str

    @abstractmethod
    async def should_use(self, context: AgentContext) -> float:
        """Return confidence score (0-1) that this strategy should handle the query."""
        ...

    @abstractmethod
    async def get_route(self, context: AgentContext) -> RouteResult:
        """Get the routing result for this strategy."""
        ...


class DirectStrategy(RoutingStrategy):
    """Direct response without planning — for simple queries."""

    name = "direct"

    DIRECT_PATTERNS = [
        "what is",
        "who is",
        "define",
        "explain",
        "tell me about",
        "how does",
        "what does",
        "why is",
        "when was",
        "where is",
    ]

    async def should_use(self, context: AgentContext) -> float:
        if not context.messages:
            return 0.0

        last_message = context.messages[-1].content.lower()

        for pattern in self.DIRECT_PATTERNS:
            if last_message.startswith(pattern):
                return 0.8

        if len(last_message.split()) < 10:
            return 0.5

        return 0.2

    async def get_route(self, context: AgentContext) -> RouteResult:
        return RouteResult(
            strategy="direct",
            intent="simple_query",
            confidence=0.8,
            tools_needed=[],
            metadata={"skip_planning": True},
        )


class PlanStrategy(RoutingStrategy):
    """Plan-based execution for complex, multi-step tasks."""

    name = "plan"

    PLAN_PATTERNS = [
        "search for",
        "find",
        "look up",
        "browse",
        "go to",
        "open",
        "create",
        "write",
        "build",
        "help me",
        "can you",
        "please",
        "i need",
        "i want",
    ]

    TOOL_KEYWORDS: dict[str, str] = {
        "search": "web_search",
        "browse": "web_browse",
        "website": "web_browse",
        "page": "web_browse",
        "file": "file_read",
        "read": "file_read",
        "code": "code_exec",
        "run": "code_exec",
        "execute": "code_exec",
        "fetch": "http_fetch",
        "http": "http_fetch",
        "https": "http_fetch",
        "summarize": "summarize_url",
        "summary": "summarize_url",
        "calculate": "calculator",
        "math": "calculator",
        "compute": "calculator",
        "how much": "calculator",
        "divide": "calculator",
        "multiply": "calculator",
        "subtract": "calculator",
        "percentage": "calculator",
        "percent": "calculator",
        "average": "calculator",
        "convert": "calculator",
        "note": "notes_memory",
        "remember": "notes_memory",
        "save": "file_writer",
        "export": "file_writer",
        "plan": "plan_manage",
        "draft": "plan_manage",
        "schedule": "plan_manage",
        "track": "plan_manage",
        "todo": "plan_manage",
        "task": "plan_manage",
        "goals": "plan_manage",
        "checklist": "plan_manage",
    }

    PLAN_CREATION_PATTERNS = [
        "create a plan",
        "make a plan",
        "set up a plan",
        "draft a plan",
        "build a plan",
        "create plan",
        "new plan",
        "plan to",
        "plan for",
        "track this",
        "track my",
        "schedule this",
        "add to my plan",
        "create a schedule",
        "set up a schedule",
        "create a todo",
        "make a todo",
    ]

    URL_PATTERN = r'https?://[^\s<>"{}|\\^`\[\]]+'

    async def should_use(self, context: AgentContext) -> float:
        if not context.messages:
            return 0.0

        last_message = context.messages[-1].content.lower()
        original_message = context.messages[-1].content

        for pattern in self.PLAN_CREATION_PATTERNS:
            if pattern in last_message:
                return 0.95

        if re.search(self.URL_PATTERN, original_message):
            return 0.95

        for pattern in self.PLAN_PATTERNS:
            if pattern in last_message:
                return 0.8

        for keyword in self.TOOL_KEYWORDS:
            if keyword in last_message:
                return 0.85

        if " and " in last_message or " then " in last_message:
            return 0.9

        if len(last_message.split()) > 20:
            return 0.7

        return 0.3

    async def get_route(self, context: AgentContext) -> RouteResult:
        last_message = context.messages[-1].content.lower() if context.messages else ""
        original_message = context.messages[-1].content if context.messages else ""

        is_plan_creation = any(
            pattern in last_message for pattern in self.PLAN_CREATION_PATTERNS
        )

        # Detect tools needed
        tools_needed: list[str] = []
        for keyword, tool in self.TOOL_KEYWORDS.items():
            if " " in keyword:
                match = keyword in last_message
            elif len(keyword) <= 5:
                match = bool(re.search(r"\b" + re.escape(keyword) + r"\b", last_message))
            else:
                match = keyword in last_message
            if match and tool not in tools_needed:
                tools_needed.append(tool)

        # Detect math expressions
        if "calculator" not in tools_needed:
            if re.search(r"\d+\s*[\+\-\*\/\^\%]\s*\d+", last_message):
                tools_needed.append("calculator")

        if is_plan_creation or "plan_manage" in tools_needed:
            intent = "plan_creation"
            if "plan_manage" not in tools_needed:
                tools_needed.append("plan_manage")
            metadata = {
                "requires_planning": True,
                "plan_creation": True,
                "user_prompt": original_message,
            }
        else:
            intent = "complex_task"
            metadata = {"requires_planning": True}

        return RouteResult(
            strategy="plan",
            intent=intent,
            confidence=0.7,
            tools_needed=tools_needed,
            metadata=metadata,
        )


class ClarifyStrategy(RoutingStrategy):
    """Ask for clarification when intent is unclear."""

    name = "clarify"

    async def should_use(self, context: AgentContext) -> float:
        if not context.messages:
            return 1.0

        last_message = context.messages[-1].content.strip()

        if len(last_message) < 5:
            return 0.9

        if not any(c.isalnum() for c in last_message):
            return 0.9

        return 0.1

    async def get_route(self, context: AgentContext) -> RouteResult:
        return RouteResult(
            strategy="clarify",
            intent="unclear",
            confidence=0.5,
            tools_needed=[],
            metadata={"needs_clarification": True},
        )
