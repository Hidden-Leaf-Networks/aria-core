"""PostgreSQL persistence backend with SQLAlchemy 2.0 async + asyncpg.

Provides:
- SQLAlchemy ORM models mapped to tenant-scoped tables
- PostgresProvider implementing PersistenceProvider protocol
- Alembic migration support
- Connection pooling via asyncpg
"""

from aria_core.persistence.postgres.models import (
    Base,
    TenantRow,
    PlanRow,
    PlanActionRow,
    ApprovalRow,
    ApprovalDecisionRow,
    EventRow,
    AgentContextRow,
    RiskPolicyRow,
    ApprovalGateRow,
)
from aria_core.persistence.postgres.provider import PostgresProvider
from aria_core.persistence.postgres.engine import create_engine, create_session_factory

__all__ = [
    "Base",
    "TenantRow",
    "PlanRow",
    "PlanActionRow",
    "ApprovalRow",
    "ApprovalDecisionRow",
    "EventRow",
    "AgentContextRow",
    "RiskPolicyRow",
    "ApprovalGateRow",
    "PostgresProvider",
    "create_engine",
    "create_session_factory",
]
