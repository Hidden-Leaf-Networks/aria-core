"""Persistence layer — tenant-aware storage for plans, approvals, events, and context.

Provides a protocol-based persistence interface with pluggable backends:
- InMemoryProvider: Default, for local dev and single-tenant mode
- PostgresProvider: Production, multi-tenant with asyncpg

Usage:
    from aria_core.persistence import InMemoryProvider, PersistenceProvider

    # For production with PostgreSQL:
    from aria_core.persistence.postgres import PostgresProvider, create_engine, create_session_factory
"""

from aria_core.persistence.protocol import PersistenceProvider
from aria_core.persistence.memory import InMemoryProvider
from aria_core.persistence.event_store import EventStore

__all__ = [
    "EventStore",
    "InMemoryProvider",
    "PersistenceProvider",
]
