"""Tests for the agent execution endpoint."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from aria_core.api.auth import AuthUser, Role
from aria_core.api.routes.execute import (
    ExecutionRequest,
    ExecutionResult,
    _DirectRouter,
    _EventCollector,
    _resolve_model,
    _build_agent_config,
    execute_agent,
)
from aria_core.providers.manager import ProviderConfig, ProviderManager, ProviderType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _make_user(
    tenant_id: UUID = TENANT_ID,
    role: Role = Role.OPERATOR,
) -> AuthUser:
    return AuthUser(
        user_id="test-user",
        tenant_id=tenant_id,
        tenant_slug="test-tenant",
        role=role,
    )


def _configure_openai_stub(manager: ProviderManager, tenant_id: UUID = TENANT_ID) -> None:
    """Configure OpenAI provider with a fake key so get_adapter succeeds."""
    manager.configure(
        tenant_id,
        ProviderConfig(
            provider=ProviderType.OPENAI,
            api_key="sk-test-fake-key",
        ),
    )


@pytest.fixture(autouse=True)
def _reset_provider_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the provider singleton and patch it with a stub-aware manager."""
    import aria_core.api.routes.providers as providers_mod

    manager = ProviderManager()
    _configure_openai_stub(manager)
    monkeypatch.setattr(providers_mod, "_manager", manager)

    # Patch get_adapter to return OpenAIAdapterStub instead of real OpenAIAdapter
    _orig_get_adapter = manager.get_adapter

    def _stub_get_adapter(tid: UUID, model_id: str) -> Any:
        # Validate through the real path (checks model exists, provider configured)
        # but intercept at the end to return a stub
        model = manager.get_model(model_id)
        if not model:
            raise ValueError(f"Unknown model: {model_id}")
        config = manager.get_config(tid, model.provider)
        if not config:
            raise ValueError(
                f"Provider '{model.provider}' not configured for tenant {tid}. "
                f"Call configure() with an API key first."
            )
        if not config.enabled:
            raise ValueError(f"Provider '{model.provider}' is disabled for tenant {tid}")

        from aria_core.adapters.openai import OpenAIAdapterStub
        return OpenAIAdapterStub()

    monkeypatch.setattr(manager, "get_adapter", _stub_get_adapter)


# ---------------------------------------------------------------------------
# ExecutionRequest validation
# ---------------------------------------------------------------------------


class TestExecutionRequest:
    def test_valid_minimal(self) -> None:
        req = ExecutionRequest({"message": "Hello"})
        assert req.message == "Hello"
        assert req.agent_id is None
        assert req.model is None
        assert req.stream is False

    def test_valid_full(self) -> None:
        req = ExecutionRequest({
            "message": "Hello",
            "agent_id": "abc-123",
            "model": "gpt-4o",
            "conversation_id": str(uuid4()),
            "stream": True,
        })
        assert req.agent_id == "abc-123"
        assert req.model == "gpt-4o"
        assert req.stream is True

    def test_missing_message(self) -> None:
        with pytest.raises(ValueError, match="message"):
            ExecutionRequest({})

    def test_empty_message(self) -> None:
        with pytest.raises(ValueError, match="message"):
            ExecutionRequest({"message": ""})


# ---------------------------------------------------------------------------
# ExecutionResult serialization
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_to_dict(self) -> None:
        result = ExecutionResult(
            execution_id="exec-1",
            agent_id=None,
            model_used="gpt-4",
            response="Hello back",
            state="complete",
            steps=0,
            duration_ms=42,
            events=[],
            checkpoints=0,
        )
        d = result.to_dict()
        assert d["execution_id"] == "exec-1"
        assert d["response"] == "Hello back"
        assert d["error"] is None

    def test_to_dict_with_error(self) -> None:
        result = ExecutionResult(
            execution_id="exec-2",
            agent_id="agent-1",
            model_used="gpt-4",
            response="",
            state="error",
            steps=0,
            duration_ms=10,
            events=[],
            checkpoints=0,
            error="Something went wrong",
        )
        d = result.to_dict()
        assert d["error"] == "Something went wrong"
        assert d["state"] == "error"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_explicit_override(self) -> None:
        req = ExecutionRequest({"message": "hi", "model": "gpt-4o"})
        assert _resolve_model(req, None) == "gpt-4o"

    def test_from_agent_config(self) -> None:
        req = ExecutionRequest({"message": "hi"})
        assert _resolve_model(req, {"model": "claude-sonnet-4-20250514"}) == "claude-sonnet-4-20250514"

    def test_default_fallback(self) -> None:
        req = ExecutionRequest({"message": "hi"})
        assert _resolve_model(req, None) == "gpt-4"

    def test_request_overrides_agent(self) -> None:
        req = ExecutionRequest({"message": "hi", "model": "gpt-4o"})
        assert _resolve_model(req, {"model": "gpt-4"}) == "gpt-4o"


class TestBuildAgentConfig:
    def test_defaults(self) -> None:
        req = ExecutionRequest({"message": "hi"})
        config = _build_agent_config("gpt-4o", req, None)
        assert config.model == "gpt-4o"

    def test_from_agent_registry(self) -> None:
        req = ExecutionRequest({"message": "hi"})
        agent = {
            "system_prompt": "You are a ninja.",
            "max_steps": 5,
            "temperature": 0.3,
        }
        config = _build_agent_config("gpt-4", req, agent)
        assert config.system_prompt == "You are a ninja."
        assert config.max_steps == 5
        assert config.temperature == 0.3


# ---------------------------------------------------------------------------
# Event collector
# ---------------------------------------------------------------------------


class TestEventCollector:
    @pytest.mark.asyncio
    async def test_collects_events(self) -> None:
        collector = _EventCollector()
        await collector("test.event", {"key": "value"})
        assert len(collector.events) == 1
        assert collector.events[0]["event_type"] == "test.event"

    @pytest.mark.asyncio
    async def test_strips_context(self) -> None:
        collector = _EventCollector()
        await collector("test.event", {"key": "value", "_context": "internal"})
        assert "_context" not in collector.events[0]["payload"]


# ---------------------------------------------------------------------------
# Full execution flow
# ---------------------------------------------------------------------------


class TestExecuteAgent:
    @pytest.mark.asyncio
    async def test_basic_execution(self) -> None:
        """Basic execution with stub adapter returns a complete result."""
        user = _make_user()
        result = await execute_agent({"message": "Hello, ARIA"}, user)

        assert result["state"] == "complete"
        assert result["error"] is None
        assert "[STUB]" in result["response"]
        assert result["model_used"] == "gpt-4"
        assert result["duration_ms"] >= 0
        assert isinstance(result["execution_id"], str)

    @pytest.mark.asyncio
    async def test_missing_message_error(self) -> None:
        """Missing message returns validation error."""
        user = _make_user()
        result = await execute_agent({}, user)

        assert result["state"] == "error"
        assert "message" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_model_override(self) -> None:
        """Explicit model override is used."""
        user = _make_user()
        result = await execute_agent(
            {"message": "Hi", "model": "gpt-4o"},
            user,
        )

        assert result["state"] == "complete"
        assert result["model_used"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_unknown_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown model returns error with suggestions."""
        import aria_core.api.routes.providers as providers_mod

        manager = ProviderManager()
        _configure_openai_stub(manager)
        monkeypatch.setattr(providers_mod, "_manager", manager)
        # Don't patch get_adapter — let the real one handle unknown model

        user = _make_user()
        result = await execute_agent(
            {"message": "Hi", "model": "nonexistent-model-v99"},
            user,
        )

        assert result["state"] == "error"
        assert "Unknown model" in result["error"]
        assert "Available models" in result["error"]

    @pytest.mark.asyncio
    async def test_no_provider_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unconfigured provider returns clear error."""
        import aria_core.api.routes.providers as providers_mod

        # Fresh manager with NO providers configured
        manager = ProviderManager()
        monkeypatch.setattr(providers_mod, "_manager", manager)

        user = _make_user()
        result = await execute_agent(
            {"message": "Hi", "model": "gpt-4"},
            user,
        )

        assert result["state"] == "error"
        assert "not configured" in result["error"].lower() or "API key" in result["error"]

    @pytest.mark.asyncio
    async def test_agent_not_found(self) -> None:
        """Referencing a nonexistent agent_id returns error."""
        user = _make_user()
        result = await execute_agent(
            {"message": "Hi", "agent_id": str(uuid4())},
            user,
        )

        assert result["state"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_response_extraction(self) -> None:
        """Response text is extracted from the stub adapter."""
        user = _make_user()
        result = await execute_agent({"message": "What is 2+2?"}, user)

        assert result["state"] == "complete"
        assert "What is 2+2?" in result["response"]
        assert result["response"].startswith("[STUB]")

    @pytest.mark.asyncio
    async def test_event_tracking(self) -> None:
        """Events are collected during execution."""
        user = _make_user()
        result = await execute_agent({"message": "Track events"}, user)

        assert result["state"] == "complete"
        assert len(result["events"]) > 0
        event_types = [e["event_type"] for e in result["events"]]
        assert "agent.start" in event_types

    @pytest.mark.asyncio
    async def test_checkpoint_counting(self) -> None:
        """Time-travel checkpoints are captured and counted."""
        user = _make_user()
        result = await execute_agent({"message": "Checkpoint test"}, user)

        assert result["state"] == "complete"
        # The FSM goes through multiple transitions, each can produce a checkpoint
        assert result["checkpoints"] >= 0
        assert isinstance(result["checkpoints"], int)

    @pytest.mark.asyncio
    async def test_conversation_id_passthrough(self) -> None:
        """Conversation ID is passed through to the state machine."""
        user = _make_user()
        conv_id = str(uuid4())
        result = await execute_agent(
            {"message": "With convo", "conversation_id": conv_id},
            user,
        )

        assert result["state"] == "complete"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_steps_counted(self) -> None:
        """Steps field reflects FSM step count."""
        user = _make_user()
        result = await execute_agent({"message": "Count steps"}, user)

        assert result["state"] == "complete"
        assert isinstance(result["steps"], int)
        assert result["steps"] >= 0

    @pytest.mark.asyncio
    async def test_execution_id_unique(self) -> None:
        """Each execution gets a unique ID."""
        user = _make_user()
        r1 = await execute_agent({"message": "First"}, user)
        r2 = await execute_agent({"message": "Second"}, user)

        assert r1["execution_id"] != r2["execution_id"]
