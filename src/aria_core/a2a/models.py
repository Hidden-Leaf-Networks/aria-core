"""A2A protocol data models — Agent Cards, Tasks, Messages."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member


class TaskState(StrEnum):
    """A2A task lifecycle states."""

    CREATED = "created"
    QUEUED = "queued"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"


TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}


class A2APart(BaseModel):
    """Content part of an A2A message."""

    text: str | None = None
    file_uri: str | None = None
    mime_type: str | None = None
    structured_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2AMessage(BaseModel):
    """A2A protocol message."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = "user"  # "user" or "agent"
    parts: list[A2APart] = Field(default_factory=list)
    context_id: str | None = None
    task_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text(cls, content: str, role: str = "user") -> A2AMessage:
        """Create a simple text message."""
        return cls(role=role, parts=[A2APart(text=content)])


class A2AArtifact(BaseModel):
    """Output artifact from task execution."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    parts: list[A2APart] = Field(default_factory=list)
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATask(BaseModel):
    """A2A task — a unit of work delegated between agents."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    context_id: str | None = None
    status: TaskState = TaskState.CREATED
    messages: list[A2AMessage] = Field(default_factory=list)
    artifacts: list[A2AArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES


class AgentSkill(BaseModel):
    """A capability exposed by an A2A agent."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCapabilities(BaseModel):
    """A2A agent capability flags."""

    streaming: bool = False
    push_notifications: bool = False
    extended_agent_card: bool = False


class AgentProvider(BaseModel):
    """Organization providing the agent."""

    organization: str
    url: str | None = None


class AgentCard(BaseModel):
    """A2A Agent Card — discovery document for agent capabilities.

    Served at: GET /.well-known/a2a/agent-card
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    provider: AgentProvider = Field(default_factory=lambda: AgentProvider(organization="Aria Core"))
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    security_schemes: list[dict[str, Any]] = Field(default_factory=list)
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def for_aria_core(
        cls,
        name: str = "Aria Core Agent",
        base_url: str = "http://localhost:8000",
        skills: list[AgentSkill] | None = None,
    ) -> AgentCard:
        """Create an AgentCard for an aria-core instance."""
        import aria_core

        default_skills = [
            AgentSkill(
                id="plan-execution",
                name="Plan Execution",
                description="Create and execute multi-step plans with dependency tracking and risk scoring",
                tags=["planning", "orchestration", "execution"],
                examples=["Create a CI/CD deployment plan", "Execute a data pipeline"],
            ),
            AgentSkill(
                id="risk-assessment",
                name="Risk Assessment",
                description="Score risk for proposed actions (0-100) with factor analysis",
                tags=["security", "risk", "compliance"],
                examples=["Assess risk of deploying to production", "Score a database migration"],
            ),
            AgentSkill(
                id="research-analysis",
                name="Research & Analysis",
                description="Multi-source research with synthesis and structured reports",
                tags=["research", "analysis", "reports"],
                examples=["Research competitor landscape", "Analyze market trends"],
            ),
            AgentSkill(
                id="code-assistance",
                name="Code Assistance",
                description="Code generation, review, testing, and refactoring",
                tags=["code", "engineering", "development"],
                examples=["Generate a REST API", "Review this pull request"],
            ),
        ]

        return cls(
            name=name,
            description=f"Aria Core v{aria_core.__version__} — Deterministic AI agent with multi-model consensus, risk scoring, and approval gates.",
            provider=AgentProvider(
                organization="Hidden Leaf Networks",
                url="https://hiddenleafnetworks.com",
            ),
            version=aria_core.__version__,
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
                extended_agent_card=False,
            ),
            skills=skills or default_skills,
            url=base_url,
        )
