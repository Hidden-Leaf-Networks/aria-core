"""OpenAI model adapter."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from aria_core.adapters.base import ModelAdapter
from aria_core.runtime.models import AgentContext


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models (GPT-4, GPT-4o, etc.)."""

    name = "openai"
    supports_streaming = True
    supports_tools = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install aria-core[openai]")
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def generate_response(self, context: AgentContext) -> str:
        client = self._get_client()
        messages = self.format_messages(context)

        response = await client.chat.completions.create(
            model=context.config.model or self.model,
            messages=messages,
            temperature=context.config.temperature if context.config.temperature is not None else 0.7,
            max_tokens=context.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_response(self, context: AgentContext) -> AsyncIterator[str]:
        client = self._get_client()
        messages = self.format_messages(context)

        stream = await client.chat.completions.create(
            model=context.config.model or self.model,
            messages=messages,
            temperature=context.config.temperature if context.config.temperature is not None else 0.7,
            max_tokens=context.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_with_tools(
        self,
        context: AgentContext,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None]:
        client = self._get_client()
        messages = self.format_messages(context)

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

        response = await client.chat.completions.create(
            model=context.config.model or self.model,
            messages=messages,
            temperature=context.config.temperature if context.config.temperature is not None else 0.7,
            max_tokens=context.config.max_tokens,
            tools=openai_tools if openai_tools else None,
        )

        message = response.choices[0].message
        content = message.content or ""

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        return content, tool_calls


class OpenAIAdapterStub(ModelAdapter):
    """Stub adapter for testing without API calls."""

    name = "openai-stub"
    supports_streaming = True
    supports_tools = True

    async def generate_response(self, context: AgentContext) -> str:
        last_message = context.messages[-1].content if context.messages else "empty"
        return f"[STUB] Response to: {last_message[:100]}"

    async def stream_response(self, context: AgentContext) -> AsyncIterator[str]:
        response = await self.generate_response(context)
        for word in response.split():
            yield word + " "

    async def generate_with_tools(
        self,
        context: AgentContext,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None]:
        response = await self.generate_response(context)
        return response, None
