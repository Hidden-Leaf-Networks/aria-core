"""Security middleware — rate limiting, request validation, security headers.

Provides:
- Per-tenant rate limiting (in-memory token bucket)
- Request body size limits
- Security response headers
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any
from uuid import UUID


class RateLimiter:
    """Per-tenant token bucket rate limiter.

    Tracks request counts per tenant within sliding windows.
    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._window_seconds = 60.0
        # {tenant_id: list of timestamps}
        self._requests: dict[UUID, list[float]] = defaultdict(list)

    def check(self, tenant_id: UUID) -> tuple[bool, dict[str, Any]]:
        """Check if a request is allowed.

        Returns (allowed, headers) where headers contains rate limit info.
        """
        now = time.monotonic()
        window_start = now - self._window_seconds

        # Clean old entries
        entries = self._requests[tenant_id]
        self._requests[tenant_id] = [t for t in entries if t > window_start]
        entries = self._requests[tenant_id]

        remaining = max(0, self.requests_per_minute - len(entries))
        headers = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(window_start + self._window_seconds)),
        }

        if len(entries) >= self.requests_per_minute:
            return False, headers

        # Allow and record
        self._requests[tenant_id].append(now)
        headers["X-RateLimit-Remaining"] = str(remaining - 1)
        return True, headers

    def reset(self, tenant_id: UUID | None = None) -> None:
        """Reset rate limit counters."""
        if tenant_id:
            self._requests.pop(tenant_id, None)
        else:
            self._requests.clear()


# Security headers applied to all responses
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


# Max request body size (1MB default)
MAX_REQUEST_BODY_BYTES = 1_048_576
