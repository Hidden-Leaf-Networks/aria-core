"""API configuration — environment-driven settings."""

from __future__ import annotations

import os
from typing import Any


class APIConfig:
    """Configuration loaded from environment variables."""

    def __init__(self) -> None:
        # Server
        self.host: str = os.getenv("ARIA_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("ARIA_PORT", "8000"))
        self.debug: bool = os.getenv("ARIA_DEBUG", "false").lower() == "true"

        # Database
        self.database_url: str = os.getenv(
            "ARIA_DATABASE_URL",
            "postgresql+asyncpg://aria:aria@localhost:5432/aria_core",
        )

        # Auth
        self.jwt_secret: str = os.getenv("ARIA_JWT_SECRET", "")
        self.jwt_algorithm: str = os.getenv("ARIA_JWT_ALGORITHM", "HS256")
        self.jwt_issuer: str = os.getenv("ARIA_JWT_ISSUER", "aria-core")
        self.jwt_audience: str = os.getenv("ARIA_JWT_AUDIENCE", "aria-core-api")

        # CORS
        self.cors_origins: list[str] = os.getenv(
            "ARIA_CORS_ORIGINS", "*"
        ).split(",")

        # Provider mode
        self.persistence_mode: str = os.getenv("ARIA_PERSISTENCE", "memory")

    @property
    def is_production(self) -> bool:
        return not self.debug

    @property
    def use_postgres(self) -> bool:
        return self.persistence_mode == "postgres"
