"""Capability-based fine-grained authorization.

Provides per-skill, per-resource permission grants that layer on top of
the coarse-grained RBAC roles in ``aria_core.api.auth``.

ARIA-305: Fine-grained Auth — capability tokens, per-skill permissions.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResourceType(StrEnum):
    """Resource types that capabilities can target."""

    SKILL = "skill"
    PLAN = "plan"
    AGENT = "agent"
    TENANT = "tenant"
    EVENT = "event"
    APPROVAL = "approval"


class Action(StrEnum):
    """Actions that can be granted on a resource."""

    EXECUTE = "execute"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    APPROVE = "approve"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Capability(BaseModel):
    """A single permission grant for a specific resource and action set."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    description: str = ""
    resource_type: ResourceType
    resource_id: str = "*"
    actions: List[str] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    granted_to: str
    granted_by: str
    expires_at: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CapabilityToken(BaseModel):
    """A bundle of capabilities issued to a user with an expiry."""

    token_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: uuid.UUID
    user_id: str
    capabilities: List[Capability] = Field(default_factory=list)
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------

_BUSINESS_HOUR_START = 9
_BUSINESS_HOUR_END = 17


def _evaluate_condition(
    key: str,
    value: Any,
    context: Dict[str, Any],
) -> Tuple[bool, str]:
    """Evaluate a single condition against the provided context.

    Returns ``(allowed, reason)``.
    """

    if key == "max_risk_score":
        risk = context.get("risk_score", 0)
        if risk > value:
            return False, f"Risk score {risk} exceeds max allowed {value}"
        return True, ""

    if key == "time_window":
        if value == "always":
            return True, ""
        if value == "business_hours":
            now = context.get("current_time", datetime.now(timezone.utc))
            if not (_BUSINESS_HOUR_START <= now.hour < _BUSINESS_HOUR_END):
                return False, (
                    f"Action restricted to business hours "
                    f"({_BUSINESS_HOUR_START}:00-{_BUSINESS_HOUR_END}:00), "
                    f"current hour is {now.hour}"
                )
            return True, ""
        return False, f"Unknown time_window value: {value}"

    if key == "max_daily_executions":
        count = context.get("daily_execution_count", 0)
        if count >= value:
            return False, (
                f"Daily execution limit reached ({count}/{value})"
            )
        return True, ""

    if key == "required_approval":
        if value and not context.get("has_approval", False):
            return False, "Action requires prior approval"
        return True, ""

    # Unknown conditions pass by default (open-world policy).
    return True, ""


# ---------------------------------------------------------------------------
# CapabilityChecker
# ---------------------------------------------------------------------------

class CapabilityChecker:
    """Manages capability grants and performs authorization checks.

    All state is held in-memory; plug a persistence backend for production.
    """

    def __init__(self, tenant_id: uuid.UUID) -> None:
        self.tenant_id = tenant_id
        self._grants: Dict[uuid.UUID, Capability] = {}

    # -- Grant management ----------------------------------------------------

    def grant(self, capability: Capability) -> Capability:
        """Store a capability grant and return it."""
        self._grants[capability.id] = capability
        return capability

    def revoke(self, capability_id: uuid.UUID) -> bool:
        """Deactivate a capability by ID. Returns ``True`` if found."""
        cap = self._grants.get(capability_id)
        if cap is None:
            return False
        cap.is_active = False
        return True

    def list_grants(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
    ) -> List[Capability]:
        """List grants, optionally filtered by user and/or resource type."""
        results: List[Capability] = []
        for cap in self._grants.values():
            if user_id and cap.granted_to != user_id:
                continue
            if resource_type and cap.resource_type != resource_type:
                continue
            results.append(cap)
        return results

    # -- Authorization checks ------------------------------------------------

    def _is_expired(self, cap: Capability) -> bool:
        if cap.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= cap.expires_at

    def _matching_grants(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: str,
    ) -> List[Capability]:
        """Return all active, non-expired grants that match the request."""
        matches: List[Capability] = []
        for cap in self._grants.values():
            if not cap.is_active:
                continue
            if self._is_expired(cap):
                continue
            if cap.granted_to != user_id:
                continue
            if cap.resource_type != resource_type:
                continue
            if cap.resource_id != "*" and cap.resource_id != resource_id:
                continue
            if action not in cap.actions:
                continue
            matches.append(cap)
        return matches

    def check(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: str,
    ) -> bool:
        """Return ``True`` if the user has at least one matching grant."""
        return len(self._matching_grants(user_id, resource_type, resource_id, action)) > 0

    def check_with_conditions(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: str,
        context: Dict[str, Any] | None = None,
    ) -> Tuple[bool, str]:
        """Check permission **and** evaluate conditions.

        Returns ``(allowed, reason)`` where *reason* explains denial.
        """
        context = context or {}
        grants = self._matching_grants(user_id, resource_type, resource_id, action)
        if not grants:
            return False, "No matching capability grant"

        # A grant passes if ALL its conditions are satisfied.
        for cap in grants:
            all_ok = True
            denial_reason = ""
            for cond_key, cond_val in cap.conditions.items():
                ok, reason = _evaluate_condition(cond_key, cond_val, context)
                if not ok:
                    all_ok = False
                    denial_reason = reason
                    break
            if all_ok:
                return True, ""

        # All matching grants failed their conditions.
        return False, denial_reason

    # -- Token management ----------------------------------------------------

    def create_token(
        self,
        user_id: str,
        capability_ids: List[uuid.UUID],
        expires_in_seconds: int = 3600,
    ) -> CapabilityToken:
        """Bundle selected capabilities into a ``CapabilityToken``."""
        caps: List[Capability] = []
        for cid in capability_ids:
            cap = self._grants.get(cid)
            if cap and cap.is_active and not self._is_expired(cap):
                caps.append(cap)

        now = datetime.now(timezone.utc)
        from datetime import timedelta

        return CapabilityToken(
            tenant_id=self.tenant_id,
            user_id=user_id,
            capabilities=caps,
            issued_at=now,
            expires_at=now + timedelta(seconds=expires_in_seconds),
        )

    def validate_token(self, token: CapabilityToken) -> bool:
        """Validate a token: check expiry and that all capabilities are active."""
        now = datetime.now(timezone.utc)
        if now >= token.expires_at:
            return False
        for cap in token.capabilities:
            if not cap.is_active:
                return False
            stored = self._grants.get(cap.id)
            if stored is None or not stored.is_active:
                return False
        return True

    # -- Effective capabilities ----------------------------------------------

    def get_effective_capabilities(self, user_id: str) -> List[Capability]:
        """Return all active, non-expired grants for a user."""
        results: List[Capability] = []
        for cap in self._grants.values():
            if cap.granted_to != user_id and cap.is_active and not self._is_expired(cap):
                continue
            if cap.granted_to == user_id and cap.is_active and not self._is_expired(cap):
                results.append(cap)
        return results


# ---------------------------------------------------------------------------
# SkillPermissionGuard
# ---------------------------------------------------------------------------

class SkillPermissionGuard:
    """Convenience wrapper around :class:`CapabilityChecker` for skill execution."""

    def __init__(self, checker: CapabilityChecker) -> None:
        self._checker = checker

    def can_execute(
        self,
        user_id: str,
        skill_name: str,
        risk_score: int = 0,
    ) -> Tuple[bool, str]:
        """Check whether *user_id* may execute *skill_name*.

        Returns ``(allowed, reason)``.
        """
        context: Dict[str, Any] = {"risk_score": risk_score}
        return self._checker.check_with_conditions(
            user_id=user_id,
            resource_type=ResourceType.SKILL,
            resource_id=skill_name,
            action="execute",
            context=context,
        )

    def can_read(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
    ) -> bool:
        """Check read access on any resource type."""
        return self._checker.check(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action="read",
        )

    def enforce(
        self,
        user_id: str,
        skill_name: str,
        risk_score: int = 0,
    ) -> None:
        """Raise ``PermissionError`` if execution is denied."""
        allowed, reason = self.can_execute(user_id, skill_name, risk_score)
        if not allowed:
            raise PermissionError(
                f"Permission denied for skill '{skill_name}': {reason}"
            )
