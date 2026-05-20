"""Tests for security middleware — rate limiting, headers."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.api.security import RateLimiter, SECURITY_HEADERS


class TestRateLimiter:
    def test_allows_under_limit(self) -> None:
        limiter = RateLimiter(requests_per_minute=5)
        tid = uuid4()

        for _ in range(5):
            allowed, headers = limiter.check(tid)
            assert allowed is True

    def test_blocks_over_limit(self) -> None:
        limiter = RateLimiter(requests_per_minute=3)
        tid = uuid4()

        for _ in range(3):
            limiter.check(tid)

        allowed, headers = limiter.check(tid)
        assert allowed is False
        assert headers["X-RateLimit-Remaining"] == "0"

    def test_headers_present(self) -> None:
        limiter = RateLimiter(requests_per_minute=10)
        _, headers = limiter.check(uuid4())
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers

    def test_tenant_isolation(self) -> None:
        limiter = RateLimiter(requests_per_minute=2)
        tid_a = uuid4()
        tid_b = uuid4()

        # Exhaust tenant A
        limiter.check(tid_a)
        limiter.check(tid_a)
        allowed_a, _ = limiter.check(tid_a)
        assert allowed_a is False

        # Tenant B still has quota
        allowed_b, _ = limiter.check(tid_b)
        assert allowed_b is True

    def test_reset_single_tenant(self) -> None:
        limiter = RateLimiter(requests_per_minute=1)
        tid = uuid4()

        limiter.check(tid)
        allowed, _ = limiter.check(tid)
        assert allowed is False

        limiter.reset(tid)
        allowed, _ = limiter.check(tid)
        assert allowed is True

    def test_reset_all(self) -> None:
        limiter = RateLimiter(requests_per_minute=1)
        t1, t2 = uuid4(), uuid4()

        limiter.check(t1)
        limiter.check(t2)
        limiter.reset()

        allowed1, _ = limiter.check(t1)
        allowed2, _ = limiter.check(t2)
        assert allowed1 is True
        assert allowed2 is True


class TestSecurityHeaders:
    def test_all_headers_defined(self) -> None:
        assert "X-Content-Type-Options" in SECURITY_HEADERS
        assert "X-Frame-Options" in SECURITY_HEADERS
        assert "Cache-Control" in SECURITY_HEADERS
        assert "Permissions-Policy" in SECURITY_HEADERS

    async def test_headers_in_response(self) -> None:
        """Security headers appear in API responses."""
        from httpx import ASGITransport, AsyncClient
        from aria_core.api.app import create_app
        from aria_core.api.config import APIConfig
        from aria_core.api.deps import set_provider
        from aria_core.persistence.memory import InMemoryProvider
        from aria_core.tenant.models import DEFAULT_TENANT

        config = APIConfig()
        config.jwt_secret = "test"
        app = create_app(config)

        provider = InMemoryProvider()
        await provider.save_tenant(DEFAULT_TENANT)
        set_provider(provider)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert resp.headers["X-Frame-Options"] == "DENY"
            assert resp.headers["Cache-Control"] == "no-store"
