"""Knowledge system data models."""

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


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class Document(BaseModel):
    """A document to be ingested into a knowledge base."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID | None = None
    content: str
    title: str = ""
    source: str = ""
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chunk(BaseModel):
    """A chunk of a document with embedding vector."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    tenant_id: UUID | None = None
    content: str
    index: int = 0
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchResult(BaseModel):
    """A search result from the knowledge base."""

    chunk: Chunk
    score: float = 0.0
    document_title: str = ""
    document_source: str = ""
