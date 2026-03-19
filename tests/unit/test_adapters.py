"""Tests for LLM provider adapters."""

from __future__ import annotations

from aria_core.adapters import (
    AnthropicAdapter,
    ModelAdapter,
    OpenAIAdapter,
    OpenAIAdapterStub,
    XAIAdapter,
)
from aria_core.runtime.models import AgentConfig, AgentContext, ChatMessage, MessageRole


def _ctx(message: str = "Hello", system_prompt: str | None = None) -> AgentContext:
    config = AgentConfig(system_prompt=system_prompt)
    return AgentContext(
        config=config,
        messages=[ChatMessage(role=MessageRole.USER, content=message)],
    )


# ---------------------------------------------------------------------------
# Stub adapter tests (no API keys needed)
# ---------------------------------------------------------------------------


class TestOpenAIAdapterStub:
    async def test_generate_response(self) -> None:
        adapter = OpenAIAdapterStub()
        response = await adapter.generate_response(_ctx("What is Python?"))
        assert "[STUB]" in response
        assert "What is Python?" in response

    async def test_stream_response(self) -> None:
        adapter = OpenAIAdapterStub()
        tokens = []
        async for token in adapter.stream_response(_ctx("hi")):
            tokens.append(token)
        assert len(tokens) > 0

    async def test_generate_with_tools(self) -> None:
        adapter = OpenAIAdapterStub()
        response, tool_calls = await adapter.generate_with_tools(
            _ctx("test"), [{"name": "search", "description": "Search", "parameters": {}}]
        )
        assert "[STUB]" in response
        assert tool_calls is None


# ---------------------------------------------------------------------------
# Base adapter formatting
# ---------------------------------------------------------------------------


class TestBaseFormatting:
    def test_format_messages_with_system_prompt(self) -> None:
        adapter = OpenAIAdapterStub()
        ctx = _ctx("Hello", system_prompt="You are a pirate.")
        messages = adapter.format_messages(ctx)

        assert messages[0]["role"] == "system"
        assert "pirate" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_format_messages_default_system(self) -> None:
        adapter = OpenAIAdapterStub()
        ctx = _ctx("Hello")
        messages = adapter.format_messages(ctx)

        assert "helpful" in messages[0]["content"].lower()

    def test_format_messages_with_skill_results(self) -> None:
        adapter = OpenAIAdapterStub()
        ctx = _ctx("What did you find?")
        ctx.skill_results = {
            "step_0": {"skill": "web_search", "success": True, "result": {"content": "Found it!"}},
            "step_1": {"skill": "failing", "success": False, "error": "Timeout"},
        }
        messages = adapter.format_messages(ctx)

        # Should have system, user message, and skill results
        assert len(messages) == 3
        assert "Found it!" in messages[2]["content"]
        assert "Timeout" in messages[2]["content"]


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_openai_adapter_is_model_adapter(self) -> None:
        adapter = OpenAIAdapter(api_key="fake")
        assert isinstance(adapter, ModelAdapter)
        assert adapter.name == "openai"

    def test_anthropic_adapter_is_model_adapter(self) -> None:
        adapter = AnthropicAdapter(api_key="fake")
        assert isinstance(adapter, ModelAdapter)
        assert adapter.name == "anthropic"

    def test_xai_adapter_extends_openai(self) -> None:
        adapter = XAIAdapter(api_key="fake")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.name == "xai"
        assert "x.ai" in adapter.base_url

    def test_openai_custom_base_url(self) -> None:
        adapter = OpenAIAdapter(api_key="fake", base_url="http://localhost:8080/v1")
        assert adapter.base_url == "http://localhost:8080/v1"

    def test_anthropic_default_model(self) -> None:
        adapter = AnthropicAdapter(api_key="fake")
        assert "claude" in adapter.model

    def test_xai_default_model(self) -> None:
        adapter = XAIAdapter(api_key="fake")
        assert "grok" in adapter.model

    def test_repr(self) -> None:
        adapter = OpenAIAdapterStub()
        assert "OpenAIAdapterStub" in repr(adapter)


# ---------------------------------------------------------------------------
# Anthropic message formatting
# ---------------------------------------------------------------------------


class TestAnthropicFormatting:
    def test_separates_system_prompt(self) -> None:
        adapter = AnthropicAdapter(api_key="fake")
        ctx = _ctx("Hello", system_prompt="Be concise.")
        system, messages = adapter.format_messages(ctx)

        assert "concise" in system
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_system_messages_merged(self) -> None:
        adapter = AnthropicAdapter(api_key="fake")
        ctx = AgentContext(
            config=AgentConfig(system_prompt="Base prompt."),
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="Extra context."),
                ChatMessage(role=MessageRole.USER, content="Hi"),
            ],
        )
        system, messages = adapter.format_messages(ctx)

        assert "Base prompt." in system
        assert "Extra context." in system
        # System message should not be in messages list
        assert len(messages) == 1

    def test_tool_role_becomes_user(self) -> None:
        adapter = AnthropicAdapter(api_key="fake")
        ctx = AgentContext(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Hi"),
                ChatMessage(role=MessageRole.TOOL, content="Tool result"),
            ],
        )
        _, messages = adapter.format_messages(ctx)
        assert all(m["role"] in ("user", "assistant") for m in messages)

    def test_extended_thinking_kwargs(self) -> None:
        adapter = AnthropicAdapter(api_key="fake")
        ctx = _ctx("Think hard")
        ctx.config.extended_thinking = True
        ctx.config.thinking_budget_tokens = 5000

        system, messages = adapter.format_messages(ctx)
        kwargs = adapter._build_api_kwargs(ctx, system, messages)

        assert "thinking" in kwargs
        assert kwargs["thinking"]["budget_tokens"] == 5000
        assert "temperature" not in kwargs  # Must not be set with thinking
