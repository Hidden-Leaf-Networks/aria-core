"""API middleware — request ID injection and auth extraction."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from aria_core.api.auth import AuthError, AuthUser, decode_token, extract_user
from aria_core.api.config import APIConfig


class RequestContext:
    """Per-request context attached to request state."""

    def __init__(
        self,
        request_id: str,
        user: AuthUser | None = None,
    ) -> None:
        self.request_id = request_id
        self.user = user


def create_auth_dependency(config: APIConfig) -> Callable:
    """Create a FastAPI dependency that extracts AuthUser from the Authorization header.

    Returns a dependency function compatible with FastAPI's Depends().
    """

    async def get_current_user(authorization: str | None = None) -> AuthUser:
        """Extract and validate the JWT from the Authorization header."""
        if not authorization:
            raise AuthError("Missing Authorization header")

        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthError("Authorization header must be: Bearer <token>")

        token = parts[1]

        if not config.jwt_secret:
            raise AuthError("JWT secret not configured", status_code=500)

        claims = decode_token(
            token,
            secret=config.jwt_secret,
            algorithm=config.jwt_algorithm,
            issuer=config.jwt_issuer if config.jwt_issuer else None,
            audience=config.jwt_audience if config.jwt_audience else None,
        )

        return extract_user(claims)

    return get_current_user
