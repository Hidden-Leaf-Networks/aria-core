"""Agentic Data Cloud — enterprise data access for AI agents.

Query federation across Snowflake, BigQuery, Databricks, and PostgreSQL
without moving data. Agents query where the data lives.

Provides:
- DataSource: connection config for external data stores
- DataConnector protocol: pluggable backend adapters
- QueryEngine: tenant-scoped query execution with access control
- Schema discovery for agent context

Usage:
    from aria_core.data_cloud import QueryEngine, DataSource

    engine = QueryEngine(tenant_id=tid)
    engine.register(DataSource(name="warehouse", type="snowflake", ...))
    results = await engine.query("warehouse", "SELECT * FROM users LIMIT 10")
"""

from aria_core.data_cloud.models import DataSource, QueryResult, DataSourceType
from aria_core.data_cloud.engine import QueryEngine

__all__ = ["DataSource", "DataSourceType", "QueryEngine", "QueryResult"]
