"""Runtime data models for the agent state machine."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field

# Python 3.10 compat
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """String enum backport for Python < 3.11."""

        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return self.value


class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class MessageRole(StrEnum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single chat message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class AgentStateEnum(StrEnum):
    """Agent FSM states."""

    IDLE = "idle"
    ROUTING = "routing"
    PLANNING = "planning"
    EXECUTING_STEP = "executing_step"
    AWAITING_APPROVAL = "awaiting_approval"
    RESPONDING = "responding"
    COMPLETE = "complete"
    ERROR = "error"


class AgentConfig(BaseModel):
    """Agent configuration."""

    max_steps: int = Field(default=10, ge=1, le=50)
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: int = Field(default=300, ge=1, le=600)
    model: str = Field(default="gpt-4")
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    require_approval_for_restricted: bool = True
    extended_thinking: bool = Field(default=False)
    thinking_budget_tokens: int = Field(default=10000, ge=1024, le=128000)
    use_llm_planning: bool = Field(default=False)
    self_critique: bool = Field(default=False)


class AgentContext(BaseModel):
    """Agent execution context — passed through the state machine."""

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID = Field(default_factory=uuid4)
    config: AgentConfig = Field(default_factory=AgentConfig)
    messages: list[ChatMessage] = Field(default_factory=list)
    current_plan_id: UUID | None = None
    current_step_index: int = 0
    step_count: int = 0
    skill_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResult(BaseModel):
    """Final agent state snapshot returned after execution."""

    id: UUID = Field(default_factory=uuid4)
    state: AgentStateEnum = AgentStateEnum.IDLE
    context: AgentContext
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_terminal(self) -> bool:
        """Check if agent is in a terminal state."""
        return self.state in (AgentStateEnum.COMPLETE, AgentStateEnum.ERROR)

    @property
    def response(self) -> str | None:
        """Get the agent's response text, if any."""
        return self.context.metadata.get("response")
