"""Tests for the intent router and strategies."""

from __future__ import annotations

from aria_core.router import (
    ClarifyStrategy,
    DirectStrategy,
    PlanStrategy,
    RouteResult,
    Router,
    RoutingStrategy,
)
from aria_core.runtime.models import AgentContext, ChatMessage, MessageRole


def _ctx(message: str) -> AgentContext:
    """Create a context with a single user message."""
    return AgentContext(
        messages=[ChatMessage(role=MessageRole.USER, content=message)]
    )


# ---------------------------------------------------------------------------
# DirectStrategy
# ---------------------------------------------------------------------------


class TestDirectStrategy:
    async def test_simple_question_high_score(self) -> None:
        s = DirectStrategy()
        assert await s.should_use(_ctx("what is Python?")) >= 0.8

    async def test_short_query_medium_score(self) -> None:
        s = DirectStrategy()
        score = await s.should_use(_ctx("hello"))
        assert 0.4 <= score <= 0.6

    async def test_long_query_low_score(self) -> None:
        s = DirectStrategy()
        score = await s.should_use(_ctx("I need you to search for data and then create a report"))
        assert score <= 0.3

    async def test_route_returns_direct(self) -> None:
        s = DirectStrategy()
        result = await s.get_route(_ctx("what is AI?"))
        assert result.strategy == "direct"


# ---------------------------------------------------------------------------
# PlanStrategy
# ---------------------------------------------------------------------------


class TestPlanStrategy:
    async def test_plan_creation_pattern(self) -> None:
        s = PlanStrategy()
        assert await s.should_use(_ctx("create a plan for the launch")) >= 0.9

    async def test_url_triggers_plan(self) -> None:
        s = PlanStrategy()
        assert await s.should_use(_ctx("summarize https://example.com")) >= 0.9

    async def test_multi_step_triggers_plan(self) -> None:
        s = PlanStrategy()
        assert await s.should_use(_ctx("search for data and then export it")) >= 0.8

    async def test_tool_detection(self) -> None:
        s = PlanStrategy()
        result = await s.get_route(_ctx("search for Python tutorials"))
        assert "web_search" in result.tools_needed

    async def test_math_detection(self) -> None:
        s = PlanStrategy()
        result = await s.get_route(_ctx("what is 15 + 27"))
        assert "calculator" in result.tools_needed

    async def test_plan_creation_intent(self) -> None:
        s = PlanStrategy()
        result = await s.get_route(_ctx("create a plan to learn Rust"))
        assert result.intent == "plan_creation"
        assert "plan_manage" in result.tools_needed

    async def test_simple_query_low_score(self) -> None:
        s = PlanStrategy()
        score = await s.should_use(_ctx("hi"))
        assert score <= 0.5


# ---------------------------------------------------------------------------
# ClarifyStrategy
# ---------------------------------------------------------------------------


class TestClarifyStrategy:
    async def test_empty_messages_high_score(self) -> None:
        s = ClarifyStrategy()
        ctx = AgentContext(messages=[])
        assert await s.should_use(ctx) >= 0.9

    async def test_very_short_high_score(self) -> None:
        s = ClarifyStrategy()
        assert await s.should_use(_ctx("hi")) >= 0.9

    async def test_punctuation_only_high_score(self) -> None:
        s = ClarifyStrategy()
        assert await s.should_use(_ctx("???")) >= 0.9

    async def test_normal_message_low_score(self) -> None:
        s = ClarifyStrategy()
        score = await s.should_use(_ctx("explain quantum computing"))
        assert score <= 0.2

    async def test_route_returns_clarify(self) -> None:
        s = ClarifyStrategy()
        result = await s.get_route(_ctx("?"))
        assert result.strategy == "clarify"
        assert result.metadata["needs_clarification"] is True


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TestRouter:
    async def test_simple_query_routes_direct(self) -> None:
        router = Router()
        result = await router.route(_ctx("what is machine learning?"))
        assert result["strategy"] == "direct"

    async def test_complex_query_routes_plan(self) -> None:
        router = Router()
        result = await router.route(_ctx("search for AI papers and create a summary report"))
        assert result["strategy"] == "plan"

    async def test_ambiguous_routes_clarify(self) -> None:
        router = Router()
        result = await router.route(_ctx("???"))
        assert result["strategy"] == "clarify"

    async def test_all_scores_present(self) -> None:
        router = Router()
        result = await router.route(_ctx("hello"))
        assert "direct" in result["all_scores"]
        assert "plan" in result["all_scores"]
        assert "clarify" in result["all_scores"]

    async def test_add_custom_strategy(self) -> None:
        class CustomStrategy(RoutingStrategy):
            name = "custom"

            async def should_use(self, context: AgentContext) -> float:
                return 0.99  # Always wins

            async def get_route(self, context: AgentContext) -> RouteResult:
                return RouteResult(strategy="custom", intent="custom", confidence=1.0)

        router = Router()
        router.add_strategy(CustomStrategy())
        result = await router.route(_ctx("anything"))
        assert result["strategy"] == "custom"

    async def test_remove_strategy(self) -> None:
        router = Router()
        assert router.remove_strategy("clarify") is True
        assert router.remove_strategy("nonexistent") is False
        assert len(router.strategies) == 2
