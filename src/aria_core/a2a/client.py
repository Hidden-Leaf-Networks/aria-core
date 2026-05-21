"""A2A Client — delegate tasks to external A2A agents.

Discovers agents via their Agent Cards, sends tasks, and tracks results.

Usage:
    client = A2AClient()
    card = await client.discover("http://external-agent.com")
    task = await client.send_task(card, "Analyze this data")
    result = await client.get_task(card, task["id"])
"""

from __future__ import annotations

from typing import Any

from aria_core.a2a.models import AgentCard, A2AMessage, A2APart


class A2AClientError(Exception):
    pass


class A2AClient:
    """Client for delegating tasks to external A2A agents."""

    def __init__(self) -> None:
        self._discovered: dict[str, AgentCard] = {}

    async def discover(self, base_url: str) -> AgentCard:
        """Discover an agent by fetching its Agent Card.

        Args:
            base_url: Agent's base URL (e.g. "http://agent.example.com")
        """
        try:
            import httpx
        except ImportError:
            raise A2AClientError("httpx required for A2A client: pip install httpx")

        url = f"{base_url.rstrip('/')}/.well-known/a2a/agent-card"
        async with httpx.AsyncClient() as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()

        card = AgentCard(**data)
        card = card.model_copy(update={"url": base_url})
        self._discovered[base_url] = card
        return card

    async def send_task(
        self,
        card: AgentCard,
        text: str,
        task_id: str | None = None,
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a task to an A2A agent via SendMessage.

        Args:
            card: Agent Card (from discover())
            text: Task description
            task_id: Existing task ID (for continuation)
            context_id: Conversation context
            metadata: Additional metadata
        """
        try:
            import httpx
        except ImportError:
            raise A2AClientError("httpx required")

        message = {
            "role": "user",
            "parts": [{"text": text}],
        }
        if task_id:
            message["task_id"] = task_id
        if context_id:
            message["context_id"] = context_id

        rpc_request = {
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {"message": message, "metadata": metadata or {}},
            "id": "1",
        }

        url = f"{card.url.rstrip('/')}/a2a"
        async with httpx.AsyncClient() as http:
            resp = await http.post(url, json=rpc_request)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise A2AClientError(f"A2A error: {data['error']}")

        return data.get("result", {})

    async def get_task(
        self,
        card: AgentCard,
        task_id: str,
        history_length: int | None = None,
    ) -> dict[str, Any]:
        """Get task status from an A2A agent."""
        try:
            import httpx
        except ImportError:
            raise A2AClientError("httpx required")

        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length

        rpc_request = {
            "jsonrpc": "2.0",
            "method": "GetTask",
            "params": params,
            "id": "1",
        }

        url = f"{card.url.rstrip('/')}/a2a"
        async with httpx.AsyncClient() as http:
            resp = await http.post(url, json=rpc_request)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise A2AClientError(f"A2A error: {data['error']}")

        return data.get("result", {})

    async def cancel_task(
        self, card: AgentCard, task_id: str
    ) -> dict[str, Any]:
        """Cancel a task on an A2A agent."""
        try:
            import httpx
        except ImportError:
            raise A2AClientError("httpx required")

        rpc_request = {
            "jsonrpc": "2.0",
            "method": "CancelTask",
            "params": {"id": task_id},
            "id": "1",
        }

        url = f"{card.url.rstrip('/')}/a2a"
        async with httpx.AsyncClient() as http:
            resp = await http.post(url, json=rpc_request)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise A2AClientError(f"A2A error: {data['error']}")

        return data.get("result", {})

    async def list_tasks(
        self,
        card: AgentCard,
        status: str | None = None,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """List tasks on an A2A agent."""
        try:
            import httpx
        except ImportError:
            raise A2AClientError("httpx required")

        params: dict[str, Any] = {"pageSize": page_size}
        if status:
            params["status"] = status

        rpc_request = {
            "jsonrpc": "2.0",
            "method": "ListTasks",
            "params": params,
            "id": "1",
        }

        url = f"{card.url.rstrip('/')}/a2a"
        async with httpx.AsyncClient() as http:
            resp = await http.post(url, json=rpc_request)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise A2AClientError(f"A2A error: {data['error']}")

        return data.get("result", {})

    @property
    def discovered_agents(self) -> list[AgentCard]:
        return list(self._discovered.values())
