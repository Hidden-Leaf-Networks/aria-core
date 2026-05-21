"""Agent registry API routes — register and manage agents per tenant."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from aria_core.api.auth import AuthUser, Role, require_role
from aria_core.api.deps import get_guard


# In-memory agent registry (per-tenant)
# In production this would go through the persistence provider
_agent_registry: dict[UUID, dict[UUID, dict[str, Any]]] = {}


def _get_tenant_agents(tenant_id: UUID) -> dict[UUID, dict[str, Any]]:
    if tenant_id not in _agent_registry:
        _agent_registry[tenant_id] = {}
    return _agent_registry[tenant_id]


async def list_agents(user: AuthUser) -> list[dict[str, Any]]:
    """List all registered agents for the tenant."""
    agents = _get_tenant_agents(user.tenant_id)
    return list(agents.values())


async def register_agent(
    data: dict[str, Any],
    user: AuthUser,
) -> dict[str, Any]:
    """Register a new agent."""
    require_role(user, Role.OPERATOR)
    agents = _get_tenant_agents(user.tenant_id)

    agent_id = uuid4()
    agent = {
        "id": str(agent_id),
        "tenant_id": str(user.tenant_id),
        "name": data.get("name", "Unnamed Agent"),
        "slug": data.get("slug", f"agent-{agent_id.hex[:8]}"),
        "description": data.get("description", ""),
        "model": data.get("model", "gpt-4"),
        "system_prompt": data.get("system_prompt"),
        "allowed_skills": data.get("allowed_skills", []),
        "max_steps": data.get("max_steps", 10),
        "temperature": data.get("temperature", 0.7),
        "status": "active",
        "executions": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": user.user_id,
    }
    agents[agent_id] = agent
    return agent


async def get_agent(
    agent_id: UUID,
    user: AuthUser,
) -> dict[str, Any] | None:
    """Get an agent by ID."""
    agents = _get_tenant_agents(user.tenant_id)
    return agents.get(agent_id)


async def delete_agent(
    agent_id: UUID,
    user: AuthUser,
) -> dict[str, Any]:
    """Delete an agent."""
    require_role(user, Role.OPERATOR)
    agents = _get_tenant_agents(user.tenant_id)
    if agent_id in agents:
        del agents[agent_id]
        return {"deleted": True, "agent_id": str(agent_id)}
    return {"deleted": False, "agent_id": str(agent_id)}
