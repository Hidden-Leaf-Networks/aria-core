"""Tests for A2A protocol integration."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.a2a.models import (
    AgentCard,
    AgentSkill,
    A2AMessage,
    A2APart,
    A2ATask,
    TaskState,
    TERMINAL_STATES,
)
from aria_core.a2a.server import A2AServer
from aria_core.a2a.client import A2AClient


class TestAgentCard:
    def test_create_default_card(self) -> None:
        card = AgentCard.for_aria_core()
        assert card.name == "Aria Core Agent"
        assert card.provider.organization == "Hidden Leaf Networks"
        assert len(card.skills) > 0
        assert card.capabilities.streaming is True

    def test_card_serialization(self) -> None:
        card = AgentCard.for_aria_core()
        d = card.model_dump(mode="json")
        assert "name" in d
        assert "skills" in d
        assert "capabilities" in d

    def test_custom_card(self) -> None:
        card = AgentCard(
            name="Custom Agent",
            description="Does things",
            skills=[AgentSkill(id="s1", name="Skill 1")],
        )
        assert card.name == "Custom Agent"
        assert len(card.skills) == 1


class TestA2AModels:
    def test_message_text_factory(self) -> None:
        msg = A2AMessage.text("Hello agent", role="user")
        assert msg.role == "user"
        assert len(msg.parts) == 1
        assert msg.parts[0].text == "Hello agent"

    def test_task_lifecycle(self) -> None:
        task = A2ATask()
        assert task.status == TaskState.CREATED
        assert not task.is_terminal

        task = task.model_copy(update={"status": TaskState.COMPLETED})
        assert task.is_terminal

    def test_terminal_states(self) -> None:
        assert TaskState.COMPLETED in TERMINAL_STATES
        assert TaskState.FAILED in TERMINAL_STATES
        assert TaskState.CANCELED in TERMINAL_STATES
        assert TaskState.WORKING not in TERMINAL_STATES


class TestA2AServer:
    async def test_get_agent_card(self) -> None:
        server = A2AServer()
        card = server.get_agent_card()
        assert card["name"] == "Aria Core Agent"

    async def test_send_message_creates_task(self) -> None:
        server = A2AServer()
        result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"text": "Analyze this data"}],
                }
            },
            "id": "1",
        })
        assert "result" in result
        assert result["result"]["status"] == "created"
        assert len(result["result"]["messages"]) == 1

    async def test_send_message_with_handler(self) -> None:
        async def handler(task: A2ATask) -> A2ATask:
            return task.model_copy(update={
                "status": TaskState.COMPLETED,
                "artifacts": [],
            })

        server = A2AServer(handler=handler)
        result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {"message": {"text": "Do work"}},
            "id": "1",
        })
        assert result["result"]["status"] == "completed"

    async def test_get_task(self) -> None:
        server = A2AServer()
        # Create a task
        create_result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {"message": {"text": "Hello"}},
            "id": "1",
        })
        task_id = create_result["result"]["id"]

        # Get it
        get_result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "GetTask",
            "params": {"id": task_id},
            "id": "2",
        })
        assert get_result["result"]["id"] == task_id

    async def test_list_tasks(self) -> None:
        server = A2AServer()
        await server.send_message({"message": {"text": "Task 1"}})
        await server.send_message({"message": {"text": "Task 2"}})

        result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "ListTasks",
            "params": {},
            "id": "1",
        })
        assert result["result"]["total"] == 2

    async def test_cancel_task(self) -> None:
        server = A2AServer()
        create = await server.send_message({"message": {"text": "Cancel me"}})
        task_id = create["id"]

        result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "CancelTask",
            "params": {"id": task_id},
            "id": "1",
        })
        assert result["result"]["status"] == "canceled"

    async def test_unknown_method(self) -> None:
        server = A2AServer()
        result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "Unknown",
            "params": {},
            "id": "1",
        })
        assert "error" in result

    async def test_handler_error_fails_task(self) -> None:
        async def bad_handler(task: A2ATask) -> A2ATask:
            raise RuntimeError("Handler crashed")

        server = A2AServer(handler=bad_handler)
        result = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": {"message": {"text": "Crash"}},
            "id": "1",
        })
        assert result["result"]["status"] == "failed"


class TestA2AClient:
    def test_client_init(self) -> None:
        client = A2AClient()
        assert len(client.discovered_agents) == 0
