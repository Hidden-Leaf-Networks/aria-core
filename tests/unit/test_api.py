"""Integration tests for the FastAPI REST API.

Uses httpx AsyncClient with the app directly (no network).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from aria_core.api.app import create_app
from aria_core.api.auth import Role, create_token
from aria_core.api.config import APIConfig


SECRET = "test-api-secret-key"


def _make_config() -> APIConfig:
    config = APIConfig()
    config.jwt_secret = SECRET
    config.jwt_algorithm = "HS256"
    config.persistence_mode = "memory"
    return config


def _make_token(
    role: Role = Role.OPERATOR,
    tenant_id: str | None = None,
    tenant_slug: str = "test-co",
) -> str:
    tid = tenant_id or str(uuid4())
    return create_token(
        user_id="test-user",
        tenant_id=uuid4() if tenant_id is None else __import__("uuid").UUID(tenant_id),
        role=role,
        secret=SECRET,
        tenant_slug=tenant_slug,
    )


@pytest.fixture
def app() -> Any:
    return create_app(_make_config())


@pytest.fixture
async def client(app: Any) -> AsyncClient:
    transport = ASGITransport(app=app)
    # Use app as context manager to trigger lifespan events
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Manually trigger lifespan since ASGITransport doesn't
        from aria_core.api.deps import set_provider
        from aria_core.persistence.memory import InMemoryProvider
        from aria_core.tenant.models import DEFAULT_TENANT
        provider = InMemoryProvider()
        await provider.save_tenant(DEFAULT_TENANT)
        set_provider(provider)
        yield ac


@pytest.fixture
def admin_token() -> str:
    return _make_token(role=Role.ADMIN)


@pytest.fixture
def operator_token() -> str:
    return _make_token(role=Role.OPERATOR)


@pytest.fixture
def viewer_token() -> str:
    return _make_token(role=Role.VIEWER)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Health endpoints (no auth)
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_health(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    async def test_ready(self, client: AsyncClient) -> None:
        resp = await client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["provider"] == "ok"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    async def test_no_auth_header_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/plans")
        assert resp.status_code == 401

    async def test_invalid_token_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/plans", headers={"Authorization": "Bearer garbage"})
        assert resp.status_code == 401

    async def test_valid_token_works(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get("/api/v1/plans", headers=_auth(operator_token))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tenant CRUD (admin only)
# ---------------------------------------------------------------------------


class TestTenantAPI:
    async def test_create_tenant_admin(self, client: AsyncClient, admin_token: str) -> None:
        resp = await client.post(
            "/api/v1/tenants",
            json={"slug": "new-co", "name": "New Company"},
            headers=_auth(admin_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "new-co"
        assert data["name"] == "New Company"
        assert data["is_active"] is True

    async def test_create_tenant_operator_forbidden(
        self, client: AsyncClient, operator_token: str
    ) -> None:
        resp = await client.post(
            "/api/v1/tenants",
            json={"slug": "forbidden-co", "name": "Nope"},
            headers=_auth(operator_token),
        )
        assert resp.status_code == 403

    async def test_list_tenants_admin(self, client: AsyncClient, admin_token: str) -> None:
        # Create a tenant first
        await client.post(
            "/api/v1/tenants",
            json={"slug": "list-co", "name": "List Co"},
            headers=_auth(admin_token),
        )
        resp = await client.get("/api/v1/tenants", headers=_auth(admin_token))
        assert resp.status_code == 200
        data = resp.json()
        # At least 1 (plus the default tenant)
        assert len(data) >= 1


# ---------------------------------------------------------------------------
# Plan CRUD
# ---------------------------------------------------------------------------


class TestPlanAPI:
    async def test_create_plan(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.post(
            "/api/v1/plans",
            json={
                "name": "Deploy Pipeline",
                "description": "Build, test, deploy",
                "actions": [
                    {"name": "Build", "skill_name": "build_project"},
                    {"name": "Test", "skill_name": "run_tests", "dependencies": [0]},
                ],
            },
            headers=_auth(operator_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Deploy Pipeline"
        assert data["state"] == "draft"
        assert len(data["actions"]) == 2

    async def test_list_plans_empty(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get("/api/v1/plans", headers=_auth(operator_token))
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_viewer_cannot_create_plan(
        self, client: AsyncClient, viewer_token: str
    ) -> None:
        resp = await client.post(
            "/api/v1/plans",
            json={"name": "Nope", "actions": []},
            headers=_auth(viewer_token),
        )
        assert resp.status_code == 403

    async def test_get_plan_not_found(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get(
            f"/api/v1/plans/{uuid4()}", headers=_auth(operator_token)
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Event routes
# ---------------------------------------------------------------------------


class TestEventAPI:
    async def test_list_events_empty(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get("/api/v1/events", headers=_auth(operator_token))
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_count_events(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get("/api/v1/events/count", headers=_auth(operator_token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    async def test_replay_events_empty(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get("/api/v1/events/replay", headers=_auth(operator_token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["events"] == []


# ---------------------------------------------------------------------------
# Context routes
# ---------------------------------------------------------------------------


class TestContextAPI:
    async def test_list_contexts_empty(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get("/api/v1/contexts", headers=_auth(operator_token))
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_context_not_found(self, client: AsyncClient, operator_token: str) -> None:
        resp = await client.get(
            f"/api/v1/contexts/{uuid4()}", headers=_auth(operator_token)
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tenant isolation through API
# ---------------------------------------------------------------------------


class TestAPIIsolation:
    async def test_tenants_cannot_see_each_others_plans(
        self, client: AsyncClient
    ) -> None:
        """Two operators with different tenant_ids cannot see each other's plans."""
        tid_a = uuid4()
        tid_b = uuid4()

        token_a = create_token(
            user_id="user-a", tenant_id=tid_a, role=Role.OPERATOR,
            secret=SECRET, tenant_slug="tenant-a",
        )
        token_b = create_token(
            user_id="user-b", tenant_id=tid_b, role=Role.OPERATOR,
            secret=SECRET, tenant_slug="tenant-b",
        )

        # Tenant A creates a plan
        resp = await client.post(
            "/api/v1/plans",
            json={"name": "A's Secret Plan", "actions": [{"name": "Step"}]},
            headers=_auth(token_a),
        )
        assert resp.status_code == 200
        plan_id = resp.json()["id"]

        # Tenant A can see it
        resp = await client.get("/api/v1/plans", headers=_auth(token_a))
        assert len(resp.json()) == 1

        # Tenant B cannot see it
        resp = await client.get("/api/v1/plans", headers=_auth(token_b))
        assert len(resp.json()) == 0

        # Tenant B cannot get it by ID
        resp = await client.get(f"/api/v1/plans/{plan_id}", headers=_auth(token_b))
        assert resp.status_code == 404
