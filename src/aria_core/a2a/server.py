"""A2A Server — handles incoming agent-to-agent task requests.

Implements the A2A protocol server-side:
- Agent Card discovery (GET /.well-known/a2a/agent-card)
- SendMessage (task creation/continuation)
- GetTask, ListTasks, CancelTask
- JSON-RPC 2.0 over HTTP

Integrates with aria-core's plan engine to execute delegated tasks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from aria_core.a2a.models import (
    A2AArtifact,
    A2AMessage,
    A2APart,
    A2ATask,
    AgentCard,
    TaskState,
)


# Handler type: receives a task, returns updated task
TaskHandler = Callable[[A2ATask], Awaitable[A2ATask]]


class A2AServer:
    """A2A protocol server for aria-core.

    Manages task lifecycle and delegates execution to registered handlers.
    """

    def __init__(
        self,
        agent_card: AgentCard | None = None,
        handler: TaskHandler | None = None,
    ) -> None:
        self.agent_card = agent_card or AgentCard.for_aria_core()
        self._handler = handler
        self._tasks: dict[str, A2ATask] = {}

    def set_handler(self, handler: TaskHandler) -> None:
        """Register the task execution handler."""
        self._handler = handler

    # -------------------------------------------------------------------
    # Agent Card Discovery
    # -------------------------------------------------------------------

    def get_agent_card(self) -> dict[str, Any]:
        """Return the agent card for discovery."""
        return self.agent_card.model_dump(mode="json")

    # -------------------------------------------------------------------
    # JSON-RPC dispatch
    # -------------------------------------------------------------------

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming JSON-RPC 2.0 request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "SendMessage":
                result = await self.send_message(params)
            elif method == "GetTask":
                result = await self.get_task(params.get("id", ""), params.get("historyLength"))
            elif method == "ListTasks":
                result = await self.list_tasks(params)
            elif method == "CancelTask":
                result = await self.cancel_task(params.get("id", ""))
            else:
                return self._error_response(req_id, -32601, f"Method not found: {method}")

            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": req_id,
            }
        except Exception as e:
            return self._error_response(req_id, -32000, str(e))

    # -------------------------------------------------------------------
    # Protocol methods
    # -------------------------------------------------------------------

    async def send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle SendMessage — create or continue a task."""
        message_data = params.get("message", {})
        task_id = message_data.get("task_id")

        # Build message
        parts = [A2APart(**p) for p in message_data.get("parts", [])]
        if not parts and "text" in message_data:
            parts = [A2APart(text=message_data["text"])]

        message = A2AMessage(
            role=message_data.get("role", "user"),
            parts=parts,
            context_id=message_data.get("context_id"),
            task_id=task_id,
            metadata=message_data.get("metadata", {}),
        )

        if task_id and task_id in self._tasks:
            # Continue existing task
            task = self._tasks[task_id]
            task = task.model_copy(update={
                "messages": list(task.messages) + [message],
                "status": TaskState.WORKING,
                "updated_at": datetime.now(timezone.utc),
            })
        else:
            # Create new task
            task = A2ATask(
                context_id=message.context_id,
                status=TaskState.CREATED,
                messages=[message],
            )

        self._tasks[task.id] = task

        # Execute via handler
        if self._handler:
            task = task.model_copy(update={"status": TaskState.WORKING})
            self._tasks[task.id] = task

            try:
                task = await self._handler(task)
            except Exception as e:
                task = task.model_copy(update={
                    "status": TaskState.FAILED,
                    "metadata": {**task.metadata, "error": str(e)},
                    "updated_at": datetime.now(timezone.utc),
                })

            self._tasks[task.id] = task

        return task.model_dump(mode="json")

    async def get_task(
        self, task_id: str, history_length: int | None = None
    ) -> dict[str, Any]:
        """Get task by ID."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if history_length is not None:
            task = task.model_copy(update={
                "messages": task.messages[-history_length:],
            })

        return task.model_dump(mode="json")

    async def list_tasks(self, params: dict[str, Any]) -> dict[str, Any]:
        """List tasks with optional filters."""
        tasks = list(self._tasks.values())

        status = params.get("status")
        if status:
            tasks = [t for t in tasks if t.status.value == status]

        context_id = params.get("contextId")
        if context_id:
            tasks = [t for t in tasks if t.context_id == context_id]

        page_size = params.get("pageSize", 50)
        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)[:page_size]

        return {
            "tasks": [t.model_dump(mode="json") for t in tasks],
            "total": len(self._tasks),
        }

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if task.is_terminal:
            raise ValueError(f"Task {task_id} is already in terminal state: {task.status}")

        task = task.model_copy(update={
            "status": TaskState.CANCELED,
            "updated_at": datetime.now(timezone.utc),
        })
        self._tasks[task.id] = task
        return task.model_dump(mode="json")

    # -------------------------------------------------------------------
    # FastAPI integration
    # -------------------------------------------------------------------

    def mount(self, app: Any) -> None:
        """Mount A2A endpoints on a FastAPI app."""
        try:
            from fastapi import Request
            from fastapi.responses import JSONResponse
        except ImportError:
            raise RuntimeError("FastAPI required for A2A server mounting")

        server = self

        @app.get("/.well-known/a2a/agent-card")
        async def agent_card_endpoint() -> dict:
            return server.get_agent_card()

        @app.post("/a2a")
        async def a2a_jsonrpc_endpoint(request: Request) -> JSONResponse:
            body = await request.json()
            response = await server.handle_jsonrpc(body)
            return JSONResponse(content=response)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _error_response(
        req_id: Any, code: int, message: str
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }

    @property
    def task_count(self) -> int:
        return len(self._tasks)
