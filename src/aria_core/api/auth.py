"""JWT authentication and RBAC for the Aria Core API.

Supports:
- HS256 (shared secret) and RS256 (public key) JWT validation
- Tenant extraction from token claims (tenant_id, tenant_slug)
- Role-based access: admin, operator, viewer
- FastAPI dependency injection

Token claims expected:
    {
        "sub": "user-id",
        "tenant_id": "uuid",
        "tenant_slug": "acme-co",
        "role": "operator",
        "iss": "aria-core",
        "aud": "aria-core-api",
        "exp": 1234567890
    }
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member


class Role(StrEnum):
    """RBAC roles — ordered by privilege level."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


# Privilege hierarchy: admin > operator > viewer
ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.OPERATOR: 1,
    Role.ADMIN: 2,
}


class AuthUser(BaseModel):
    """Authenticated user extracted from JWT."""

    user_id: str
    tenant_id: UUID
    tenant_slug: str | None = None
    role: Role = Role.VIEWER
    claims: dict[str, Any] = Field(default_factory=dict)

    def has_role(self, required: Role) -> bool:
        """Check if user has at least the required role level."""
        return ROLE_HIERARCHY.get(self.role, 0) >= ROLE_HIERARCHY.get(required, 0)

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    @property
    def is_operator(self) -> bool:
        return self.has_role(Role.OPERATOR)


class AuthError(Exception):
    """Authentication/authorization error."""

    def __init__(self, detail: str, status_code: int = 401):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def decode_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
    issuer: str | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    """Decode and validate a JWT token.

    Raises AuthError on invalid/expired tokens.
    """
    try:
        from jose import jwt, JWTError, ExpiredSignatureError
    except ImportError:
        raise AuthError(
            "python-jose not installed. Install with: pip install 'aria-core[api]'",
            status_code=500,
        )

    try:
        options: dict[str, Any] = {}
        if not audience:
            options["verify_aud"] = False

        payload = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            issuer=issuer,
            audience=audience,
            options=options,
        )
        return payload

    except ExpiredSignatureError:
        raise AuthError("Token has expired")
    except JWTError as e:
        raise AuthError(f"Invalid token: {e}")


def extract_user(claims: dict[str, Any]) -> AuthUser:
    """Extract AuthUser from JWT claims."""
    user_id = claims.get("sub")
    if not user_id:
        raise AuthError("Token missing 'sub' claim")

    tenant_id_str = claims.get("tenant_id")
    if not tenant_id_str:
        raise AuthError("Token missing 'tenant_id' claim")

    try:
        tenant_id = UUID(tenant_id_str)
    except (ValueError, TypeError):
        raise AuthError(f"Invalid tenant_id: {tenant_id_str}")

    role_str = claims.get("role", "viewer")
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.VIEWER

    return AuthUser(
        user_id=str(user_id),
        tenant_id=tenant_id,
        tenant_slug=claims.get("tenant_slug"),
        role=role,
        claims=claims,
    )


def create_token(
    user_id: str,
    tenant_id: UUID,
    role: Role = Role.OPERATOR,
    secret: str = "",
    algorithm: str = "HS256",
    issuer: str = "aria-core",
    audience: str = "aria-core-api",
    tenant_slug: str | None = None,
    expires_in_seconds: int = 3600,
) -> str:
    """Create a JWT token (for testing and development)."""
    try:
        from jose import jwt
    except ImportError:
        raise AuthError(
            "python-jose not installed",
            status_code=500,
        )

    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tenant_id": str(tenant_id),
        "role": role.value,
        "iss": issuer,
        "aud": audience,
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + expires_in_seconds,
    }
    if tenant_slug:
        payload["tenant_slug"] = tenant_slug

    return jwt.encode(payload, secret, algorithm=algorithm)


def require_role(user: AuthUser, role: Role) -> None:
    """Raise AuthError if user doesn't have the required role."""
    if not user.has_role(role):
        raise AuthError(
            f"Requires '{role.value}' role, you have '{user.role.value}'",
            status_code=403,
        )
