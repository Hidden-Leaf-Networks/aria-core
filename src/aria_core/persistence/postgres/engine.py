"""Database engine and session factory for PostgreSQL 16 + asyncpg."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(
    database_url: str,
    echo: bool = False,
    pool_size: int = 10,
    max_overflow: int = 20,
    pool_pre_ping: bool = True,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine with asyncpg.

    Args:
        database_url: PostgreSQL connection URL.
            Format: postgresql+asyncpg://user:pass@host:port/dbname
        echo: Log SQL statements (dev only).
        pool_size: Number of persistent connections.
        max_overflow: Extra connections allowed beyond pool_size.
        pool_pre_ping: Test connections before use (handles dropped connections).
    """
    return create_async_engine(
        database_url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the engine."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
