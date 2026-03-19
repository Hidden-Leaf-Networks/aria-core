"""Anthropic (Claude) model adapter."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from aria_core.adapters.base import ModelAdapter
from aria_core.runtime.models import AgentContext


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models.

    Supports extended thinking and tool calling.
    """

    name = "anthropic"
    supports_streaming = True
    supports_tools = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install aria-core[anthropic]")
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    def format_messages(self, context: AgentContext) -> tuple[str, list[dict[str, str]]]:  # type: ignore[override]
        """Format for Anthropic API (system prompt separate from messages)."""
        system = context.config.system_prompt or "You are a helpful AI assistant."
        messages: list[dict[str, str]] = []

        for msg in context.messages:
            role = msg.role.value
            if role == "system":
                system = f"{system}\n\n{msg.content}"
                continue
            if role == "tool":
                role = "user"
            messages.append({"role": role, "content": msg.content})

        # Skill results
        for step_key, result in context.skill_results.items():
            if result.get("result"):
                messages.append({
                    "role": "user",
                    "content": f"Skill result ({result.get('skill', 'unknown')}): {result['result']}",
                })

        return system, messages

    def _build_api_kwargs(
        self, context: AgentContext, system: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": context.config.model or self.model,
            "max_tokens": context.config.max_tokens,
            "system": system,
            "messages": messages,
        }

        if context.config.extended_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": context.config.thinking_budget_tokens,
            }
        else:
            if context.config.temperature is not None:
                kwargs["temperature"] = context.config.temperature

        return kwargs

    @staticmethod
    def _extract_text(response: Any, context: AgentContext) -> str:
        thinking_parts: list[str] = []
        text_parts: list[str] = []

        for block in response.content:
            if getattr(block, "type", None) == "thinking":
                thinking_parts.append(block.thinking)
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        if thinking_parts:
            context.metadata["thinking_trace"] = "\n\n".join(thinking_parts)

        return "".join(text_parts)

    async def generate_response(self, context: AgentContext) -> str:
        client = self._get_client()
        system, messages = self.format_messages(context)
        kwargs = self._build_api_kwargs(context, system, messages)
        response = await client.messages.create(**kwargs)
        return self._extract_text(response, context)

    async def stream_response(self, context: AgentContext) -> AsyncIterator[str]:
        if context.config.extended_thinking:
            response = await self.generate_response(context)
            yield response
            return

        client = self._get_client()
        system, messages = self.format_messages(context)
        kwargs = self._build_api_kwargs(context, system, messages)
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate_with_tools(
        self,
        context: AgentContext,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None]:
        client = self._get_client()
        system, messages = self.format_messages(context)
        kwargs = self._build_api_kwargs(context, system, messages)

        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**kwargs)

        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if getattr(block, "type", None) == "thinking":
                thinking_parts.append(block.thinking)
            elif hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        if thinking_parts:
            context.metadata["thinking_trace"] = "\n\n".join(thinking_parts)

        return "".join(text_parts), tool_calls if tool_calls else None
