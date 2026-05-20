"""Health and readiness endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aria_core


async def health() -> dict[str, Any]:
    """Liveness probe — always returns OK if the process is running."""
    return {
        "status": "ok",
        "version": aria_core.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def ready() -> dict[str, Any]:
    """Readiness probe — checks that dependencies are available."""
    from aria_core.api.deps import get_provider

    checks: dict[str, str] = {}

    try:
        provider = get_provider()
        checks["provider"] = "ok"
    except RuntimeError:
        checks["provider"] = "not_initialized"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "version": aria_core.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
