"""Intent router — classifies queries and selects execution strategy.

The router scores all registered strategies against incoming messages
and selects the highest-scoring one. Strategies are pluggable.

Usage:
    from aria_core.router import Router, DirectStrategy, PlanStrategy

    router = Router()  # Uses default strategies
    result = await router.route(context)
    # result["strategy"] → "direct" | "plan" | "clarify" | custom
"""

from __future__ import annotations

import logging
from typing import Any

from aria_core.runtime.models import AgentContext
from aria_core.router.strategies import (
    ClarifyStrategy,
    DirectStrategy,
    PlanStrategy,
    RoutingStrategy,
)

logger = logging.getLogger("aria_core.router")


class Router:
    """Route incoming queries to the best execution strategy.

    Classifies user intent and decides whether to:
    - Respond directly (simple queries)
    - Create an execution plan (complex tasks)
    - Ask for clarification (ambiguous queries)
    - Defer to a custom strategy
    """

    def __init__(self, strategies: list[RoutingStrategy] | None = None) -> None:
        self.strategies = strategies or [
            DirectStrategy(),
            PlanStrategy(),
            ClarifyStrategy(),
        ]

    async def route(self, context: AgentContext) -> dict[str, Any]:
        """Route a query to the best strategy.

        Args:
            context: Agent context with messages.

        Returns:
            Dict with strategy, intent, confidence, tools_needed, metadata,
            and all_scores (every strategy's score for observability).
        """
        scores: list[tuple[float, RoutingStrategy]] = []
        for strategy in self.strategies:
            score = await strategy.should_use(context)
            scores.append((score, strategy))

        scores.sort(key=lambda x: x[0], reverse=True)

        best_score, best_strategy = scores[0]
        result = await best_strategy.get_route(context)

        return {
            "strategy": result.strategy,
            "intent": result.intent,
            "confidence": result.confidence,
            "tools_needed": result.tools_needed,
            "metadata": result.metadata,
            "all_scores": {s.name: score for score, s in scores},
        }

    def add_strategy(self, strategy: RoutingStrategy) -> None:
        """Add a custom routing strategy."""
        self.strategies.append(strategy)

    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy by name. Returns True if found."""
        for i, s in enumerate(self.strategies):
            if s.name == name:
                self.strategies.pop(i)
                return True
        return False


def create_default_router() -> Router:
    """Create a router with the default strategy set."""
    return Router()
