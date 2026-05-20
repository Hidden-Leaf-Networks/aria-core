"""Tests for JWT auth and RBAC."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.api.auth import (
    AuthError,
    AuthUser,
    Role,
    ROLE_HIERARCHY,
    create_token,
    decode_token,
    extract_user,
    require_role,
)


SECRET = "test-secret-key-for-unit-tests"


class TestRBAC:
    def test_role_hierarchy(self) -> None:
        assert ROLE_HIERARCHY[Role.ADMIN] > ROLE_HIERARCHY[Role.OPERATOR]
        assert ROLE_HIERARCHY[Role.OPERATOR] > ROLE_HIERARCHY[Role.VIEWER]

    def test_has_role(self) -> None:
        admin = AuthUser(user_id="u1", tenant_id=uuid4(), role=Role.ADMIN)
        assert admin.has_role(Role.ADMIN) is True
        assert admin.has_role(Role.OPERATOR) is True
        assert admin.has_role(Role.VIEWER) is True

        viewer = AuthUser(user_id="u2", tenant_id=uuid4(), role=Role.VIEWER)
        assert viewer.has_role(Role.VIEWER) is True
        assert viewer.has_role(Role.OPERATOR) is False
        assert viewer.has_role(Role.ADMIN) is False

    def test_is_admin(self) -> None:
        admin = AuthUser(user_id="u1", tenant_id=uuid4(), role=Role.ADMIN)
        assert admin.is_admin is True
        assert admin.is_operator is True

        operator = AuthUser(user_id="u2", tenant_id=uuid4(), role=Role.OPERATOR)
        assert operator.is_admin is False
        assert operator.is_operator is True

    def test_require_role_passes(self) -> None:
        admin = AuthUser(user_id="u1", tenant_id=uuid4(), role=Role.ADMIN)
        require_role(admin, Role.ADMIN)  # should not raise

    def test_require_role_fails(self) -> None:
        viewer = AuthUser(user_id="u1", tenant_id=uuid4(), role=Role.VIEWER)
        with pytest.raises(AuthError, match="Requires 'admin'"):
            require_role(viewer, Role.ADMIN)


class TestJWT:
    def test_create_and_decode_token(self) -> None:
        tid = uuid4()
        token = create_token(
            user_id="user-1",
            tenant_id=tid,
            role=Role.OPERATOR,
            secret=SECRET,
            tenant_slug="test-co",
        )
        claims = decode_token(token, SECRET)
        assert claims["sub"] == "user-1"
        assert claims["tenant_id"] == str(tid)
        assert claims["role"] == "operator"
        assert claims["tenant_slug"] == "test-co"

    def test_expired_token_raises(self) -> None:
        token = create_token(
            user_id="user-1",
            tenant_id=uuid4(),
            secret=SECRET,
            expires_in_seconds=-1,
        )
        with pytest.raises(AuthError, match="expired"):
            decode_token(token, SECRET)

    def test_wrong_secret_raises(self) -> None:
        token = create_token(
            user_id="user-1",
            tenant_id=uuid4(),
            secret=SECRET,
        )
        with pytest.raises(AuthError, match="Invalid token"):
            decode_token(token, "wrong-secret")

    def test_extract_user(self) -> None:
        tid = uuid4()
        claims = {
            "sub": "user-42",
            "tenant_id": str(tid),
            "tenant_slug": "acme",
            "role": "admin",
        }
        user = extract_user(claims)
        assert user.user_id == "user-42"
        assert user.tenant_id == tid
        assert user.tenant_slug == "acme"
        assert user.role == Role.ADMIN

    def test_extract_user_missing_sub_raises(self) -> None:
        with pytest.raises(AuthError, match="sub"):
            extract_user({"tenant_id": str(uuid4())})

    def test_extract_user_missing_tenant_raises(self) -> None:
        with pytest.raises(AuthError, match="tenant_id"):
            extract_user({"sub": "user-1"})

    def test_extract_user_invalid_role_defaults_to_viewer(self) -> None:
        user = extract_user({
            "sub": "user-1",
            "tenant_id": str(uuid4()),
            "role": "superadmin",
        })
        assert user.role == Role.VIEWER
