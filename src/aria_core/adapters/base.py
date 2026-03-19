"""Base model adapter interface.

All LLM provider adapters implement this interface. The state machine
calls generate_response() during the RESPONDING state.

Subclass ModelAdapter to add new providers. Built-in adapters:
- OpenAIAdapter (GPT-4, GPT-3.5, etc.)
- AnthropicAdapter (Claude)
- XAIAdapter (Grok — OpenAI-compatible)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from aria_core.runtime.models import AgentContext


class ModelAdapter(ABC):
    """Abstract base class for LLM adapters.

    Provides a unified interface across providers for:
    - Response generation
    - Token streaming
    - Tool/function calling
    """

    name: str
    supports_streaming: bool = True
    supports_tools: bool = True

    @abstractmethod
    async def generate_response(self, context: AgentContext) -> str:
        """Generate a response given the agent context.

        Args:
            context: Agent context with messages, config, and skill results.

        Returns:
            Generated response text.
        """
        ...

    async def stream_response(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream response tokens. Override for streaming support.

        Default implementation falls back to generate_response.
        """
        response = await self.generate_response(context)
        yield response

    async def generate_with_tools(
        self,
        context: AgentContext,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Generate response with potential tool calls.

        Args:
            context: Agent context.
            tools: Tool definitions (OpenAI function-calling format).

        Returns:
            Tuple of (response_text, tool_calls or None).
        """
        # Default: no tool calling support, just generate
        response = await self.generate_response(context)
        return response, None

    def format_messages(self, context: AgentContext) -> list[dict[str, str]]:
        """Format context into API messages.

        Override in subclasses for provider-specific formatting.
        """
        messages: list[dict[str, str]] = []

        # System prompt
        system_prompt = context.config.system_prompt or "You are a helpful AI assistant."
        messages.append({"role": "system", "content": system_prompt})

        # Conversation messages
        for msg in context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Skill results as context
        if context.skill_results:
            parts = ["[Skill Execution Results]"]
            for step_key, result in context.skill_results.items():
                skill_name = result.get("skill", result.get("skill_name", "unknown"))
                if result.get("success"):
                    skill_result = result.get("result", {})
                    if isinstance(skill_result, dict):
                        content = skill_result.get("content", skill_result.get("summary", str(skill_result)))
                    else:
                        content = str(skill_result)
                    parts.append(f"\n{skill_name}: {content}")
                else:
                    parts.append(f"\n{skill_name} (failed): {result.get('error', 'Unknown error')}")
            messages.append({"role": "user", "content": "\n".join(parts)})

        return messages

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
