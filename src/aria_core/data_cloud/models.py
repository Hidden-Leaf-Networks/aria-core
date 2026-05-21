"""Data cloud models."""

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


class DataSourceType(StrEnum):
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    DATABRICKS = "databricks"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REST_API = "rest_api"
    CSV = "csv"


class DataSource(BaseModel):
    """Connection configuration for an external data store."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID | None = None
    name: str = Field(min_length=1, max_length=100)
    type: DataSourceType
    description: str = ""

    # Connection
    connection_string: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    schema_name: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)

    # Access control
    allowed_tables: list[str] = Field(default_factory=list)
    denied_tables: list[str] = Field(default_factory=list)
    max_rows: int = Field(default=10000, ge=1)
    read_only: bool = True
    timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Metadata
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TableSchema(BaseModel):
    """Schema information for a table."""

    name: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    row_count: int | None = None
    description: str = ""


class ColumnInfo(BaseModel):
    """Column metadata."""

    name: str
    type: str
    nullable: bool = True
    description: str = ""


class QueryResult(BaseModel):
    """Result of a data query."""

    source_name: str
    query: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: int = 0
    truncated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_records(self) -> list[dict[str, Any]]:
        """Convert to list of dicts."""
        return [dict(zip(self.columns, row)) for row in self.rows]
