"""Tests for Agentic Data Cloud."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.data_cloud.models import DataSource, DataSourceType, QueryResult
from aria_core.data_cloud.engine import QueryEngine, InMemoryConnector


@pytest.fixture
def connector() -> InMemoryConnector:
    c = InMemoryConnector()
    c.load_table("users", ["id", "name", "email"], [
        ["1", "Alice", "alice@example.com"],
        ["2", "Bob", "bob@example.com"],
        ["3", "Charlie", "charlie@example.com"],
    ])
    c.load_table("orders", ["id", "user_id", "amount"], [
        ["101", "1", "99.99"],
        ["102", "2", "149.50"],
    ])
    return c


@pytest.fixture
def source() -> DataSource:
    return DataSource(name="test-db", type=DataSourceType.SQLITE)


class TestInMemoryConnector:
    async def test_select_all(self, connector: InMemoryConnector, source: DataSource) -> None:
        result = await connector.execute(source, "SELECT * FROM users")
        assert result.success
        assert result.row_count == 3
        assert result.columns == ["id", "name", "email"]

    async def test_select_with_limit(self, connector: InMemoryConnector, source: DataSource) -> None:
        result = await connector.execute(source, "SELECT * FROM users LIMIT 2")
        assert result.row_count == 2

    async def test_table_not_found(self, connector: InMemoryConnector, source: DataSource) -> None:
        result = await connector.execute(source, "SELECT * FROM nonexistent")
        assert not result.success
        assert "not found" in (result.error or "")

    async def test_access_control_allowed(self, connector: InMemoryConnector) -> None:
        source = DataSource(name="restricted", type=DataSourceType.SQLITE, allowed_tables=["users"])
        result = await connector.execute(source, "SELECT * FROM users")
        assert result.success

    async def test_access_control_denied_not_in_allowed(self, connector: InMemoryConnector) -> None:
        source = DataSource(name="restricted", type=DataSourceType.SQLITE, allowed_tables=["users"])
        result = await connector.execute(source, "SELECT * FROM orders")
        assert not result.success
        assert "Access denied" in (result.error or "")

    async def test_access_control_denied_blocked(self, connector: InMemoryConnector) -> None:
        source = DataSource(name="blocked", type=DataSourceType.SQLITE, denied_tables=["orders"])
        result = await connector.execute(source, "SELECT * FROM orders")
        assert not result.success
        assert "blocked" in (result.error or "")

    async def test_max_rows_enforced(self, connector: InMemoryConnector) -> None:
        source = DataSource(name="limited", type=DataSourceType.SQLITE, max_rows=1)
        result = await connector.execute(source, "SELECT * FROM users")
        assert result.row_count == 1
        assert result.truncated is True

    async def test_discover_schema(self, connector: InMemoryConnector, source: DataSource) -> None:
        schemas = await connector.discover_schema(source)
        assert len(schemas) == 2
        names = {s.name for s in schemas}
        assert "users" in names
        assert "orders" in names

    async def test_load_csv(self) -> None:
        c = InMemoryConnector()
        c.load_csv("data", "name,value\nfoo,1\nbar,2")
        source = DataSource(name="csv", type=DataSourceType.CSV)
        result = await c.execute(source, "SELECT * FROM data")
        assert result.row_count == 2

    async def test_to_records(self, connector: InMemoryConnector, source: DataSource) -> None:
        result = await connector.execute(source, "SELECT * FROM users LIMIT 1")
        records = result.to_records()
        assert len(records) == 1
        assert records[0]["name"] == "Alice"


class TestQueryEngine:
    async def test_register_and_query(self, connector: InMemoryConnector) -> None:
        engine = QueryEngine(tenant_id=uuid4())
        source = DataSource(name="warehouse", type=DataSourceType.SNOWFLAKE)
        engine.register(source, connector)

        result = await engine.query("warehouse", "SELECT * FROM users")
        assert result.success
        assert result.row_count == 3

    async def test_unregistered_source(self) -> None:
        engine = QueryEngine(tenant_id=uuid4())
        result = await engine.query("missing", "SELECT 1")
        assert not result.success
        assert "not registered" in (result.error or "")

    async def test_inactive_source(self, connector: InMemoryConnector) -> None:
        engine = QueryEngine(tenant_id=uuid4())
        source = DataSource(name="down", type=DataSourceType.POSTGRES, is_active=False)
        engine.register(source, connector)

        result = await engine.query("down", "SELECT 1")
        assert not result.success
        assert "inactive" in (result.error or "")

    async def test_discover(self, connector: InMemoryConnector) -> None:
        engine = QueryEngine(tenant_id=uuid4())
        engine.register(DataSource(name="db", type=DataSourceType.SQLITE), connector)
        schemas = await engine.discover("db")
        assert len(schemas) == 2

    async def test_list_sources(self, connector: InMemoryConnector) -> None:
        engine = QueryEngine(tenant_id=uuid4())
        engine.register(DataSource(name="s1", type=DataSourceType.SNOWFLAKE))
        engine.register(DataSource(name="s2", type=DataSourceType.BIGQUERY))
        assert len(engine.list_sources()) == 2

    async def test_remove_source(self, connector: InMemoryConnector) -> None:
        engine = QueryEngine(tenant_id=uuid4())
        engine.register(DataSource(name="temp", type=DataSourceType.SQLITE))
        assert engine.remove_source("temp") is True
        assert engine.remove_source("temp") is False

    async def test_tenant_scoped(self, connector: InMemoryConnector) -> None:
        tid = uuid4()
        engine = QueryEngine(tenant_id=tid)
        source = DataSource(name="scoped", type=DataSourceType.POSTGRES)
        engine.register(source)
        assert engine.list_sources()[0].tenant_id == tid
