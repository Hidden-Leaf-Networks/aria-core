"""FastAPI application factory for Aria Core API.

Usage:
    # Development (in-memory):
    ARIA_JWT_SECRET=dev-secret uvicorn aria_core.api:create_app --factory

    # Production (PostgreSQL):
    ARIA_PERSISTENCE=postgres \
    ARIA_DATABASE_URL=postgresql+asyncpg://... \
    ARIA_JWT_SECRET=... \
    uvicorn aria_core.api:create_app --factory --host 0.0.0.0

NOTE: Do NOT use `from __future__ import annotations` here.
FastAPI needs runtime type introspection for dependency injection.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import aria_core
from aria_core.api.auth import AuthError, AuthUser, Role, require_role
from aria_core.api.config import APIConfig
from aria_core.api.deps import get_guard, get_resolver, set_provider
from aria_core.api.middleware import create_auth_dependency
from aria_core.api.schemas import CreatePlanRequest, CreateTenantRequest, UpdateTenantConfigRequest
from aria_core.api.security import SECURITY_HEADERS, RateLimiter
from aria_core.billing.meter import UsageMeter
from aria_core.api.ws import WebSocketManager
from aria_core.persistence.memory import InMemoryProvider
from aria_core.tenant.models import DEFAULT_TENANT


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or APIConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup/shutdown lifecycle."""
        if config.use_postgres:
            from aria_core.persistence.postgres import (
                PostgresProvider,
                create_engine,
                create_session_factory,
            )

            pg_engine = create_engine(config.database_url, echo=config.debug)
            session_factory = create_session_factory(pg_engine)
            provider = PostgresProvider(session_factory)
            set_provider(provider)
            yield
            await pg_engine.dispose()
        else:
            provider = InMemoryProvider()
            await provider.save_tenant(DEFAULT_TENANT)
            set_provider(provider)
            yield

    app = FastAPI(
        title="Aria Core API",
        description="Multi-tenant AI agent framework API",
        version=aria_core.__version__,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security: rate limiter
    rate_limiter = RateLimiter(
        requests_per_minute=int(config.cors_origins[0] == "*" and 120 or 60),
    )
    app.state.rate_limiter = rate_limiter

    # Security: response headers middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response as StarletteResponse

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next: Any) -> StarletteResponse:
            response = await call_next(request)
            for key, value in SECURITY_HEADERS.items():
                response.headers[key] = value
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # Billing: usage meter
    usage_meter = UsageMeter()
    app.state.usage_meter = usage_meter

    # Auth dependency
    _get_current_user = create_auth_dependency(config)

    async def get_current_user(
        authorization: Optional[str] = Header(None),
    ) -> AuthUser:
        return await _get_current_user(authorization)

    # Error handler
    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Any, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    # -----------------------------------------------------------------------
    # Health (no auth)
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health_endpoint() -> dict:
        from aria_core.api.routes.health import health
        return await health()

    @app.get("/ready")
    async def ready_endpoint() -> dict:
        from aria_core.api.routes.health import ready
        return await ready()

    # -----------------------------------------------------------------------
    # Tenants (admin only)
    # -----------------------------------------------------------------------

    @app.post("/api/v1/tenants")
    async def create_tenant_endpoint(
        body: CreateTenantRequest,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.tenants import create_tenant
        return await create_tenant(body.model_dump(), user)

    @app.get("/api/v1/tenants")
    async def list_tenants_endpoint(
        user: AuthUser = Depends(get_current_user),
    ) -> list:
        from aria_core.api.routes.tenants import list_tenants
        return await list_tenants(user)

    @app.get("/api/v1/tenants/{tenant_id}")
    async def get_tenant_endpoint(
        tenant_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.tenants import get_tenant
        result = await get_tenant(tenant_id, user)
        if result is None:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return result

    @app.put("/api/v1/tenants/{tenant_id}/config")
    async def update_config_endpoint(
        tenant_id: UUID,
        body: UpdateTenantConfigRequest,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.tenants import update_config
        return await update_config(tenant_id, body.model_dump(exclude_unset=True), user)

    # -----------------------------------------------------------------------
    # Plans
    # -----------------------------------------------------------------------

    @app.post("/api/v1/plans")
    async def create_plan_endpoint(
        body: CreatePlanRequest,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.plans import create_plan
        return await create_plan(body.model_dump(), user)

    @app.get("/api/v1/plans")
    async def list_plans_endpoint(
        user: AuthUser = Depends(get_current_user),
        state: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list:
        from aria_core.api.routes.plans import list_plans
        return await list_plans(user, state=state, limit=limit, offset=offset)

    @app.get("/api/v1/plans/{plan_id}")
    async def get_plan_endpoint(
        plan_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.plans import get_plan
        result = await get_plan(plan_id, user)
        if result is None:
            raise HTTPException(status_code=404, detail="Plan not found")
        return result

    @app.delete("/api/v1/plans/{plan_id}")
    async def delete_plan_endpoint(
        plan_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.plans import delete_plan
        return await delete_plan(plan_id, user)

    # -----------------------------------------------------------------------
    # Approvals
    # -----------------------------------------------------------------------

    @app.get("/api/v1/approvals")
    async def list_approvals_endpoint(
        user: AuthUser = Depends(get_current_user),
        state: Optional[str] = Query(None),
        plan_id: Optional[UUID] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list:
        from aria_core.api.routes.approvals import list_approvals
        return await list_approvals(user, state=state, plan_id=plan_id, limit=limit, offset=offset)

    @app.get("/api/v1/approvals/{approval_id}")
    async def get_approval_endpoint(
        approval_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.approvals import get_approval
        result = await get_approval(approval_id, user)
        if result is None:
            raise HTTPException(status_code=404, detail="Approval not found")
        return result

    @app.post("/api/v1/approvals/{approval_id}/approve")
    async def approve_endpoint(
        approval_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.approvals import approve_approval
        return await approve_approval(approval_id, user)

    @app.post("/api/v1/approvals/{approval_id}/reject")
    async def reject_endpoint(
        approval_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.approvals import reject_approval
        return await reject_approval(approval_id, user)

    # -----------------------------------------------------------------------
    # Agents
    # -----------------------------------------------------------------------

    @app.get("/api/v1/agents")
    async def list_agents_endpoint(
        user: AuthUser = Depends(get_current_user),
    ) -> list:
        from aria_core.api.routes.agents import list_agents
        return await list_agents(user)

    @app.post("/api/v1/agents")
    async def register_agent_endpoint(
        body: dict,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.agents import register_agent
        return await register_agent(body, user)

    @app.get("/api/v1/agents/{agent_id}")
    async def get_agent_endpoint(
        agent_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.agents import get_agent
        result = await get_agent(agent_id, user)
        if result is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return result

    @app.delete("/api/v1/agents/{agent_id}")
    async def delete_agent_endpoint(
        agent_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.agents import delete_agent
        return await delete_agent(agent_id, user)

    # -----------------------------------------------------------------------
    # Archetypes
    # -----------------------------------------------------------------------

    @app.get("/api/v1/archetypes")
    async def list_archetypes_endpoint(
        user: AuthUser = Depends(get_current_user),
        category: Optional[str] = Query(None),
    ) -> list:
        from aria_core.api.routes.archetypes import list_archetypes
        return await list_archetypes(user, category=category)

    @app.get("/api/v1/archetypes/{archetype_id}")
    async def get_archetype_endpoint(
        archetype_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.archetypes import get_archetype
        result = await get_archetype(archetype_id, user)
        if result is None:
            raise HTTPException(status_code=404, detail="Archetype not found")
        return result

    @app.post("/api/v1/archetypes")
    async def create_archetype_endpoint(
        body: dict,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.archetypes import create_archetype
        return await create_archetype(body, user)

    @app.delete("/api/v1/archetypes/{archetype_id}")
    async def delete_archetype_endpoint(
        archetype_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.archetypes import delete_archetype
        return await delete_archetype(archetype_id, user)

    @app.post("/api/v1/archetypes/{archetype_id}/deploy")
    async def deploy_archetype_endpoint(
        archetype_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.archetypes import deploy_archetype
        return await deploy_archetype(archetype_id, user)

    @app.post("/api/v1/archetypes/seed")
    async def seed_archetypes_endpoint(
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.archetypes import seed_defaults
        return await seed_defaults(user)

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    @app.get("/api/v1/events")
    async def list_events_endpoint(
        user: AuthUser = Depends(get_current_user),
        event_type: Optional[str] = Query(None),
        agent_id: Optional[UUID] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> list:
        from aria_core.api.routes.events import list_events
        return await list_events(user, event_type=event_type, agent_id=agent_id, limit=limit, offset=offset)

    @app.get("/api/v1/events/replay")
    async def replay_events_endpoint(
        user: AuthUser = Depends(get_current_user),
        event_type: Optional[str] = Query(None),
        agent_id: Optional[UUID] = Query(None),
        limit: int = Query(10000, ge=1, le=100000),
    ) -> dict:
        from aria_core.api.routes.events import replay_events
        return await replay_events(user, event_type=event_type, agent_id=agent_id, limit=limit)

    @app.get("/api/v1/events/count")
    async def count_events_endpoint(
        user: AuthUser = Depends(get_current_user),
        event_type: Optional[str] = Query(None),
    ) -> dict:
        from aria_core.api.routes.events import count_events
        return await count_events(user, event_type=event_type)

    # -----------------------------------------------------------------------
    # Contexts
    # -----------------------------------------------------------------------

    @app.get("/api/v1/contexts")
    async def list_contexts_endpoint(
        user: AuthUser = Depends(get_current_user),
        conversation_id: Optional[UUID] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list:
        from aria_core.api.routes.events import list_contexts
        return await list_contexts(user, conversation_id=conversation_id, limit=limit, offset=offset)

    @app.get("/api/v1/contexts/{context_id}")
    async def get_context_endpoint(
        context_id: UUID,
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        from aria_core.api.routes.events import get_context
        result = await get_context(context_id, user)
        if result is None:
            raise HTTPException(status_code=404, detail="Context not found")
        return result

    # -----------------------------------------------------------------------
    # WebSocket
    # -----------------------------------------------------------------------

    ws_manager = WebSocketManager()
    app.state.ws_manager = ws_manager

    @app.websocket("/ws/events")
    async def ws_events_endpoint(websocket: WebSocket) -> None:
        """WebSocket for real-time event streaming.

        Client sends initial auth message:
            {"token": "Bearer ..."}

        Server streams events for the authenticated tenant.
        """
        await websocket.accept()

        try:
            # Wait for auth message
            auth_msg = await websocket.receive_json()
            token_str = auth_msg.get("token", "")
            if token_str.startswith("Bearer "):
                token_str = token_str[7:]

            if not config.jwt_secret:
                await websocket.close(code=4001, reason="JWT not configured")
                return

            from aria_core.api.auth import decode_token, extract_user
            try:
                claims = decode_token(
                    token_str,
                    secret=config.jwt_secret,
                    algorithm=config.jwt_algorithm,
                )
                user = extract_user(claims)
            except AuthError as e:
                await websocket.close(code=4001, reason=str(e.detail))
                return

            tenant_id = user.tenant_id
            await ws_manager.connect(websocket, tenant_id)

            # Send confirmation
            await websocket.send_json({
                "event_type": "connected",
                "payload": {
                    "tenant_id": str(tenant_id),
                    "user_id": user.user_id,
                    "connections": ws_manager.connection_count(tenant_id),
                },
            })

            # Keep alive — listen for client messages (ping/pong)
            while True:
                await websocket.receive_text()

        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            # Best-effort disconnect
            try:
                await ws_manager.disconnect(websocket, tenant_id)
            except Exception:
                pass

    @app.get("/api/v1/ws/status")
    async def ws_status_endpoint(
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        """Get WebSocket connection status for the tenant."""
        return {
            "tenant_connections": ws_manager.connection_count(user.tenant_id),
            "total_connections": ws_manager.connection_count(),
        }

    # -----------------------------------------------------------------------
    # JWKS endpoint (public — no auth)
    # -----------------------------------------------------------------------

    @app.get("/.well-known/jwks.json")
    async def jwks_endpoint() -> dict:
        """Public JWKS endpoint for RS256 token verification."""
        try:
            from aria_core.api.jwks import KeyManager
            km = getattr(app.state, "key_manager", None)
            if km:
                return km.get_jwks()
        except ImportError:
            pass
        return {"keys": []}

    # -----------------------------------------------------------------------
    # Billing
    # -----------------------------------------------------------------------

    @app.get("/api/v1/billing/usage")
    async def usage_endpoint(
        user: AuthUser = Depends(get_current_user),
    ) -> dict:
        """Get usage report for the authenticated tenant."""
        from aria_core.api.routes.billing import get_usage
        return await get_usage(user, usage_meter)

    @app.get("/api/v1/billing/usage/all")
    async def all_usage_endpoint(
        user: AuthUser = Depends(get_current_user),
    ) -> list:
        """Get usage reports for all tenants. Admin only."""
        from aria_core.api.routes.billing import get_all_usage
        return await get_all_usage(user, usage_meter)

    # -----------------------------------------------------------------------
    # Pricing (public — no auth for tiers, calculator)
    # -----------------------------------------------------------------------

    @app.get("/api/v1/pricing/tiers")
    async def pricing_tiers_endpoint() -> list:
        from aria_core.api.routes.billing import get_pricing_tiers
        return await get_pricing_tiers()

    @app.get("/api/v1/pricing/calculate")
    async def pricing_calculate_endpoint(
        api_calls: int = Query(0, ge=0),
        events: int = Query(0, ge=0),
        agent_runs: int = Query(0, ge=0),
        agents: int = Query(0, ge=0),
        tenants: int = Query(1, ge=1),
        storage_gb: float = Query(0, ge=0),
    ) -> dict:
        from aria_core.api.routes.billing import calculate_pricing
        return await calculate_pricing(
            api_calls=api_calls, events=events, agent_runs=agent_runs,
            agents=agents, tenants=tenants, storage_gb=storage_gb,
        )

    return app
