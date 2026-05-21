"""Tests for fine-grained capability auth — ARIA-305."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from aria_core.auth.capabilities import (
    Action,
    Capability,
    CapabilityChecker,
    CapabilityToken,
    ResourceType,
    SkillPermissionGuard,
)


TENANT_ID = uuid4()


def _make_cap(
    user: str = "user-1",
    resource_type: ResourceType = ResourceType.SKILL,
    resource_id: str = "*",
    actions: list[str] | None = None,
    conditions: dict | None = None,
    granted_by: str = "admin-1",
    expires_at: datetime | None = None,
    is_active: bool = True,
    name: str = "test-cap",
) -> Capability:
    return Capability(
        name=name,
        resource_type=resource_type,
        resource_id=resource_id,
        actions=actions or ["execute"],
        conditions=conditions or {},
        granted_to=user,
        granted_by=granted_by,
        expires_at=expires_at,
        is_active=is_active,
    )


def _checker() -> CapabilityChecker:
    return CapabilityChecker(tenant_id=TENANT_ID)


# ---- Grant & revoke -------------------------------------------------------

class TestGrantAndRevoke:
    def test_grant_returns_capability(self) -> None:
        cc = _checker()
        cap = _make_cap()
        result = cc.grant(cap)
        assert result.id == cap.id

    def test_grant_is_stored(self) -> None:
        cc = _checker()
        cap = _make_cap()
        cc.grant(cap)
        grants = cc.list_grants()
        assert len(grants) == 1
        assert grants[0].id == cap.id

    def test_revoke_deactivates(self) -> None:
        cc = _checker()
        cap = _make_cap()
        cc.grant(cap)
        assert cc.revoke(cap.id) is True
        assert cc.check("user-1", ResourceType.SKILL, "any", "execute") is False

    def test_revoke_nonexistent_returns_false(self) -> None:
        cc = _checker()
        assert cc.revoke(uuid4()) is False

    def test_list_grants_filter_user(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="alice"))
        cc.grant(_make_cap(user="bob"))
        assert len(cc.list_grants(user_id="alice")) == 1

    def test_list_grants_filter_resource_type(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(resource_type=ResourceType.SKILL))
        cc.grant(_make_cap(resource_type=ResourceType.PLAN))
        assert len(cc.list_grants(resource_type=ResourceType.PLAN)) == 1


# ---- Basic checks ----------------------------------------------------------

class TestCheck:
    def test_check_allowed(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", actions=["execute"]))
        assert cc.check("u1", ResourceType.SKILL, "web-search", "execute") is True

    def test_check_denied_no_grant(self) -> None:
        cc = _checker()
        assert cc.check("u1", ResourceType.SKILL, "web-search", "execute") is False

    def test_check_denied_wrong_action(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", actions=["read"]))
        assert cc.check("u1", ResourceType.SKILL, "web-search", "execute") is False

    def test_check_denied_wrong_user(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", actions=["execute"]))
        assert cc.check("u2", ResourceType.SKILL, "web-search", "execute") is False

    def test_check_specific_resource_id(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", resource_id="web-search", actions=["execute"]))
        assert cc.check("u1", ResourceType.SKILL, "web-search", "execute") is True
        assert cc.check("u1", ResourceType.SKILL, "code-gen", "execute") is False

    def test_wildcard_resource_matches_all(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", resource_id="*", actions=["execute"]))
        assert cc.check("u1", ResourceType.SKILL, "anything", "execute") is True

    def test_expired_grant_denied(self) -> None:
        cc = _checker()
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        cc.grant(_make_cap(user="u1", expires_at=past))
        assert cc.check("u1", ResourceType.SKILL, "x", "execute") is False

    def test_inactive_grant_denied(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", is_active=False))
        assert cc.check("u1", ResourceType.SKILL, "x", "execute") is False


# ---- Condition evaluation --------------------------------------------------

class TestConditions:
    def test_max_risk_score_pass(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"max_risk_score": 50}))
        ok, reason = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute", {"risk_score": 30}
        )
        assert ok is True

    def test_max_risk_score_fail(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"max_risk_score": 50}))
        ok, reason = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute", {"risk_score": 80}
        )
        assert ok is False
        assert "exceeds" in reason

    def test_business_hours_pass(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"time_window": "business_hours"}))
        noon = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
        ok, _ = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute", {"current_time": noon}
        )
        assert ok is True

    def test_business_hours_fail(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"time_window": "business_hours"}))
        midnight = datetime(2026, 5, 20, 2, 0, tzinfo=timezone.utc)
        ok, reason = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute", {"current_time": midnight}
        )
        assert ok is False
        assert "business hours" in reason

    def test_time_window_always(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"time_window": "always"}))
        ok, _ = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute", {}
        )
        assert ok is True

    def test_max_daily_executions_pass(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"max_daily_executions": 10}))
        ok, _ = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute",
            {"daily_execution_count": 5},
        )
        assert ok is True

    def test_max_daily_executions_fail(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"max_daily_executions": 10}))
        ok, reason = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute",
            {"daily_execution_count": 10},
        )
        assert ok is False
        assert "limit" in reason

    def test_required_approval_pass(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"required_approval": True}))
        ok, _ = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute",
            {"has_approval": True},
        )
        assert ok is True

    def test_required_approval_fail(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(conditions={"required_approval": True}))
        ok, reason = cc.check_with_conditions(
            "user-1", ResourceType.SKILL, "*", "execute",
            {"has_approval": False},
        )
        assert ok is False
        assert "approval" in reason

    def test_no_grant_returns_reason(self) -> None:
        cc = _checker()
        ok, reason = cc.check_with_conditions(
            "nobody", ResourceType.SKILL, "*", "execute", {}
        )
        assert ok is False
        assert "No matching" in reason


# ---- Token management ------------------------------------------------------

class TestTokens:
    def test_create_token(self) -> None:
        cc = _checker()
        cap = _make_cap()
        cc.grant(cap)
        token = cc.create_token("user-1", [cap.id], expires_in_seconds=600)
        assert token.user_id == "user-1"
        assert len(token.capabilities) == 1
        assert token.tenant_id == TENANT_ID

    def test_validate_token_valid(self) -> None:
        cc = _checker()
        cap = _make_cap()
        cc.grant(cap)
        token = cc.create_token("user-1", [cap.id])
        assert cc.validate_token(token) is True

    def test_validate_token_expired(self) -> None:
        cc = _checker()
        cap = _make_cap()
        cc.grant(cap)
        token = cc.create_token("user-1", [cap.id], expires_in_seconds=0)
        assert cc.validate_token(token) is False

    def test_validate_token_revoked_cap(self) -> None:
        cc = _checker()
        cap = _make_cap()
        cc.grant(cap)
        token = cc.create_token("user-1", [cap.id])
        cc.revoke(cap.id)
        assert cc.validate_token(token) is False

    def test_create_token_skips_inactive(self) -> None:
        cc = _checker()
        cap = _make_cap(is_active=False)
        cc.grant(cap)
        token = cc.create_token("user-1", [cap.id])
        assert len(token.capabilities) == 0


# ---- Effective capabilities ------------------------------------------------

class TestEffectiveCapabilities:
    def test_returns_active_only(self) -> None:
        cc = _checker()
        cap_a = _make_cap(name="a")
        cap_b = _make_cap(name="b", is_active=False)
        cc.grant(cap_a)
        cc.grant(cap_b)
        effective = cc.get_effective_capabilities("user-1")
        assert len(effective) == 1
        assert effective[0].name == "a"

    def test_excludes_expired(self) -> None:
        cc = _checker()
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        cc.grant(_make_cap(expires_at=past))
        assert len(cc.get_effective_capabilities("user-1")) == 0


# ---- SkillPermissionGuard --------------------------------------------------

class TestSkillPermissionGuard:
    def test_can_execute_allowed(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", resource_id="web-search"))
        guard = SkillPermissionGuard(cc)
        ok, _ = guard.can_execute("u1", "web-search")
        assert ok is True

    def test_can_execute_denied(self) -> None:
        cc = _checker()
        guard = SkillPermissionGuard(cc)
        ok, reason = guard.can_execute("u1", "web-search")
        assert ok is False

    def test_can_execute_risk_exceeded(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", conditions={"max_risk_score": 30}))
        guard = SkillPermissionGuard(cc)
        ok, reason = guard.can_execute("u1", "dangerous-skill", risk_score=80)
        assert ok is False
        assert "exceeds" in reason

    def test_can_read(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", resource_type=ResourceType.PLAN, actions=["read"]))
        guard = SkillPermissionGuard(cc)
        assert guard.can_read("u1", ResourceType.PLAN, "plan-123") is True

    def test_enforce_raises(self) -> None:
        cc = _checker()
        guard = SkillPermissionGuard(cc)
        with pytest.raises(PermissionError, match="Permission denied"):
            guard.enforce("u1", "restricted-skill")

    def test_enforce_passes(self) -> None:
        cc = _checker()
        cc.grant(_make_cap(user="u1", resource_id="allowed-skill"))
        guard = SkillPermissionGuard(cc)
        guard.enforce("u1", "allowed-skill")  # should not raise


# ---- Model construction ---------------------------------------------------

class TestModels:
    def test_capability_defaults(self) -> None:
        cap = Capability(
            name="test",
            resource_type=ResourceType.SKILL,
            granted_to="u1",
            granted_by="admin",
        )
        assert cap.resource_id == "*"
        assert cap.is_active is True
        assert cap.id is not None
        assert cap.created_at is not None

    def test_capability_token_defaults(self) -> None:
        token = CapabilityToken(
            tenant_id=uuid4(),
            user_id="u1",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert token.token_id is not None
        assert len(token.capabilities) == 0

    def test_resource_type_values(self) -> None:
        assert ResourceType.SKILL.value == "skill"
        assert ResourceType.APPROVAL.value == "approval"

    def test_action_values(self) -> None:
        assert Action.EXECUTE.value == "execute"
        assert Action.APPROVE.value == "approve"
