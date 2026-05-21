"""Archetype data models."""

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


class ArchetypeCategory(StrEnum):
    RESEARCH = "research"
    ENGINEERING = "engineering"
    CONTENT = "content"
    DATA = "data"
    SUPPORT = "support"
    SECURITY = "security"
    OPERATIONS = "operations"
    CUSTOM = "custom"


class Archetype(BaseModel):
    """Reusable agent configuration template."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID | None = None  # None = global/built-in
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=1, max_length=100)
    description: str = ""
    category: ArchetypeCategory = ArchetypeCategory.CUSTOM
    icon: str = "⬢"

    # Agent config
    model: str = "gpt-4"
    system_prompt: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_steps: int = Field(default=10, ge=1, le=50)
    max_tokens: int = Field(default=4096, ge=1)
    allowed_skills: list[str] = Field(default_factory=list)

    # Risk config
    require_approval: bool = False
    risk_threshold: int | None = None

    # Metadata
    is_builtin: bool = False
    is_active: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
