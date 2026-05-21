"""Query engine — tenant-scoped data access with federation and access control.

In-memory connector for testing. Production connectors plug in via protocol.
"""

from __future__ import annotations

import csv
import io
import time
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from aria_core.data_cloud.models import (
    ColumnInfo,
    DataSource,
    DataSourceType,
    QueryResult,
    TableSchema,
)


@runtime_checkable
class DataConnector(Protocol):
    """Protocol for data source connectors."""

    async def execute(self, source: DataSource, query: str) -> QueryResult: ...
    async def discover_schema(self, source: DataSource) -> list[TableSchema]: ...


class InMemoryConnector:
    """In-memory data connector for testing and CSV sources.

    Stores tables as dicts of lists. Supports basic SQL-like queries
    via simple parsing (SELECT * FROM table [LIMIT n]).
    """

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, Any]] = {}

    def load_table(
        self, name: str, columns: list[str], rows: list[list[Any]]
    ) -> None:
        """Load a table into memory."""
        self._tables[name] = {"columns": columns, "rows": rows}

    def load_csv(self, name: str, csv_content: str) -> None:
        """Load a CSV string as a table."""
        reader = csv.reader(io.StringIO(csv_content))
        rows_raw = list(reader)
        if not rows_raw:
            return
        columns = rows_raw[0]
        rows = rows_raw[1:]
        self._tables[name] = {"columns": columns, "rows": rows}

    async def execute(self, source: DataSource, query: str) -> QueryResult:
        """Execute a simple query."""
        start = time.monotonic()

        # Parse simple SELECT
        q = query.strip().lower()
        table_name = None
        limit = source.max_rows

        if q.startswith("select"):
            parts = q.split()
            try:
                from_idx = parts.index("from")
                table_name = parts[from_idx + 1].strip(";")
            except (ValueError, IndexError):
                return QueryResult(
                    source_name=source.name,
                    query=query,
                    error="Could not parse table name from query",
                )

            if "limit" in parts:
                try:
                    limit_idx = parts.index("limit")
                    limit = min(int(parts[limit_idx + 1].strip(";")), source.max_rows)
                except (ValueError, IndexError):
                    pass

        if not table_name or table_name not in self._tables:
            return QueryResult(
                source_name=source.name,
                query=query,
                error=f"Table '{table_name}' not found",
            )

        # Access control
        if source.allowed_tables and table_name not in source.allowed_tables:
            return QueryResult(
                source_name=source.name,
                query=query,
                error=f"Access denied: table '{table_name}' not in allowed list",
            )
        if table_name in source.denied_tables:
            return QueryResult(
                source_name=source.name,
                query=query,
                error=f"Access denied: table '{table_name}' is blocked",
            )

        table = self._tables[table_name]
        rows = table["rows"][:limit]
        truncated = len(table["rows"]) > limit

        elapsed = int((time.monotonic() - start) * 1000)

        return QueryResult(
            source_name=source.name,
            query=query,
            columns=table["columns"],
            rows=rows,
            row_count=len(rows),
            execution_time_ms=elapsed,
            truncated=truncated,
        )

    async def discover_schema(self, source: DataSource) -> list[TableSchema]:
        schemas = []
        for name, table in self._tables.items():
            columns = [
                ColumnInfo(name=col, type="text") for col in table["columns"]
            ]
            schemas.append(TableSchema(
                name=name,
                columns=columns,
                row_count=len(table["rows"]),
            ))
        return schemas


class QueryEngine:
    """Tenant-scoped query engine with data source federation."""

    def __init__(self, tenant_id: UUID) -> None:
        self.tenant_id = tenant_id
        self._sources: dict[str, DataSource] = {}
        self._connectors: dict[str, DataConnector] = {}
        self._default_connector = InMemoryConnector()

    def register(
        self,
        source: DataSource,
        connector: DataConnector | None = None,
    ) -> None:
        """Register a data source."""
        source = source.model_copy(update={"tenant_id": self.tenant_id})
        self._sources[source.name] = source
        if connector:
            self._connectors[source.name] = connector

    def get_connector(self, source_name: str) -> DataConnector:
        return self._connectors.get(source_name, self._default_connector)

    async def query(
        self, source_name: str, query: str
    ) -> QueryResult:
        """Execute a query against a registered data source."""
        source = self._sources.get(source_name)
        if not source:
            return QueryResult(
                source_name=source_name,
                query=query,
                error=f"Data source '{source_name}' not registered",
            )
        if not source.is_active:
            return QueryResult(
                source_name=source_name,
                query=query,
                error=f"Data source '{source_name}' is inactive",
            )

        connector = self.get_connector(source_name)
        return await connector.execute(source, query)

    async def discover(self, source_name: str) -> list[TableSchema]:
        """Discover schema for a data source."""
        source = self._sources.get(source_name)
        if not source:
            return []
        connector = self.get_connector(source_name)
        return await connector.discover_schema(source)

    def list_sources(self) -> list[DataSource]:
        return list(self._sources.values())

    def remove_source(self, name: str) -> bool:
        if name in self._sources:
            del self._sources[name]
            self._connectors.pop(name, None)
            return True
        return False
