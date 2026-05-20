"""Aria Core REST API — FastAPI application for white-label deployments.

Provides:
- JWT/OAuth2 authentication with tenant extraction
- RBAC (admin, operator, viewer)
- Tenant-scoped CRUD for plans, approvals, events, contexts
- WebSocket for real-time agent execution streaming
- OpenAPI docs auto-generated

Usage:
    from aria_core.api import create_app

    app = create_app()
    # uvicorn aria_core.api:create_app --factory
"""

from aria_core.api.app import create_app

__all__ = ["create_app"]
