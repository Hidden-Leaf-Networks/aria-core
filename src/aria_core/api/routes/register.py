"""Public registration endpoint for free-tier signup.

ARIA-326: Allows new users to self-register, creating a tenant
and returning a JWT with viewer role (starter tier).
"""

from __future__ import annotations

import re
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from aria_core.api.auth import Role, create_token
from aria_core.api.deps import get_guard
from aria_core.tenant.models import Tenant, TenantConfig


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-IP)
# ---------------------------------------------------------------------------

_REGISTER_RATE: dict[str, list[float]] = {}
_MAX_REGISTRATIONS_PER_HOUR = 10


def _check_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if rate-limited."""
    now = time.time()
    window = 3600  # 1 hour

    if ip not in _REGISTER_RATE:
        _REGISTER_RATE[ip] = []

    # Prune expired entries
    _REGISTER_RATE[ip] = [t for t in _REGISTER_RATE[ip] if now - t < window]

    if len(_REGISTER_RATE[ip]) >= _MAX_REGISTRATIONS_PER_HOUR:
        return False

    _REGISTER_RATE[ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Email validation
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


def _email_to_slug(email: str) -> str:
    """Derive a tenant slug from an email address.

    e.g. "user@example.com" -> "user-at-example-com"
    """
    slug = email.lower().replace("@", "-at-").replace(".", "-")
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    # Ensure minimum length for Tenant slug pattern (needs >= 2 chars)
    if len(slug) < 2:
        slug = slug + "-tenant"
    return slug


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """Public registration payload."""

    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)
    company: str | None = Field(default=None, max_length=200)


class RegisterResponse(BaseModel):
    """Registration success response."""

    tenant_id: str
    user_id: str
    token: str
    tier: str = "starter"
    message: str


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


async def register_user(
    data: dict[str, Any],
    client_ip: str,
    jwt_secret: str,
    jwt_algorithm: str = "HS256",
) -> dict[str, Any]:
    """Register a new free-tier user.

    Creates a tenant, generates a JWT with viewer role, and returns
    the registration response.
    """
    # Rate limit
    if not _check_rate_limit(client_ip):
        return {
            "error": "Too many registrations. Try again later.",
            "_status": 429,
        }

    email: str = data["email"]
    password: str = data["password"]
    name: str = data["name"]
    company: str | None = data.get("company")

    # Validate email
    if not _is_valid_email(email):
        return {
            "error": "Invalid email format.",
            "_status": 422,
        }

    # Create tenant
    slug = _email_to_slug(email)
    tenant_name = company or name
    tenant_id = uuid4()

    tenant = Tenant(
        id=tenant_id,
        slug=slug,
        name=tenant_name,
        config=TenantConfig(
            metadata={
                "tier": "starter",
                "registered_email": email,
                "registered_name": name,
            }
        ),
    )

    guard = get_guard()
    await guard.save_tenant(tenant)

    # Create JWT (viewer role for starter tier)
    user_id = str(uuid4())
    token = create_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role=Role.VIEWER,
        secret=jwt_secret,
        algorithm=jwt_algorithm,
        tenant_slug=slug,
        expires_in_seconds=86400,  # 24 hours
    )

    response = RegisterResponse(
        tenant_id=str(tenant_id),
        user_id=user_id,
        token=token,
        tier="starter",
        message=f"Welcome to Aria Core, {name}!",
    )

    return response.model_dump()
