"""Agent execution endpoint — run agents via API with provider routing.

Connects the agent registry, provider manager, state machine, time-travel
debugger, and usage meter into a single execution pipeline.

Usage:
    POST /api/v1/execute
    {
        "message": "Hello, world!",
        "agent_id": "optional-uuid",
        "model": "gpt-4o",
        "conversation_id": "optional-uuid",
        "stream": false
    }
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID, uuid4

from aria_core.api.auth import AuthUser
from aria_core.runtime.models import AgentConfig, AgentStateEnum


class ExecutionRequest:
    """Validated execution request."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.agent_id: str | None = data.get("agent_id")
        self.message: str = data.get("message", "")
        self.model: str | None = data.get("model")
        self.conversation_id: str | None = data.get("conversation_id")
        self.stream: bool = data.get("stream", False)

        if not self.message:
            raise ValueError("'message' is required and must be non-empty")


class ExecutionResult:
    """Structured execution result."""

    def __init__(
        self,
        execution_id: str,
        agent_id: str | None,
        model_used: str,
        response: str,
        state: str,
        steps: int,
        duration_ms: int,
        events: list[dict[str, Any]],
        checkpoints: int,
        error: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.agent_id = agent_id
        self.model_used = model_used
        self.response = response
        self.state = state
        self.steps = steps
        self.duration_ms = duration_ms
        self.events = events
        self.checkpoints = checkpoints
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "agent_id": self.agent_id,
            "model_used": self.model_used,
            "response": self.response,
            "state": self.state,
            "steps": self.steps,
            "duration_ms": self.duration_ms,
            "events": self.events,
            "checkpoints": self.checkpoints,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Stub collaborators for direct-response execution
# ---------------------------------------------------------------------------


class _DirectRouter:
    """Routes all messages to the 'direct' strategy (skip planning)."""

    async def route(self, context: Any) -> dict[str, Any]:
        return {"strategy": "direct"}


class _NoOpPlanner:
    """No-op planner — never called for direct strategy."""

    async def create_plan(self, context: Any) -> Any:
        return None  # pragma: no cover


class _NoOpExecutor:
    """No-op executor — never called for direct strategy."""

    async def execute_step(self, context: Any) -> dict[str, Any]:
        return {"success": True}  # pragma: no cover


# ---------------------------------------------------------------------------
# Event collector callback
# ---------------------------------------------------------------------------


class _EventCollector:
    """Collects events emitted by the state machine during execution."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        # Strip internal _context refs before storing
        clean_payload = {k: v for k, v in payload.items() if k != "_context"}
        self.events.append({"event_type": event_type, "payload": clean_payload})


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------


async def execute_agent(data: dict[str, Any], user: AuthUser) -> dict[str, Any]:
    """Execute an agent with provider routing.

    Orchestrates:
    1. Request validation
    2. Agent lookup (optional)
    3. Provider/adapter resolution
    4. State machine execution with time-travel checkpointing
    5. Usage metering
    6. Result packaging

    Returns an ExecutionResult dict.
    """
    start_time = time.monotonic()
    execution_id = str(uuid4())

    # -- 1. Validate request --------------------------------------------------
    try:
        request = ExecutionRequest(data)
    except ValueError as e:
        return ExecutionResult(
            execution_id=execution_id,
            agent_id=None,
            model_used="none",
            response="",
            state="error",
            steps=0,
            duration_ms=0,
            events=[],
            checkpoints=0,
            error=str(e),
        ).to_dict()

    # -- 2. Resolve agent config (optional) ------------------------------------
    agent_config: dict[str, Any] | None = None
    if request.agent_id:
        from aria_core.api.routes.agents import get_agent

        try:
            agent_config = await get_agent(UUID(request.agent_id), user)
        except (ValueError, TypeError):
            pass

        if agent_config is None:
            return ExecutionResult(
                execution_id=execution_id,
                agent_id=request.agent_id,
                model_used="none",
                response="",
                state="error",
                steps=0,
                duration_ms=_elapsed_ms(start_time),
                events=[],
                checkpoints=0,
                error=f"Agent '{request.agent_id}' not found",
            ).to_dict()

    # -- 3. Determine model ----------------------------------------------------
    model_id = _resolve_model(request, agent_config)

    # -- 4. Get adapter via ProviderManager ------------------------------------
    from aria_core.api.routes.providers import get_manager

    provider_manager = get_manager()

    try:
        adapter = provider_manager.get_adapter(user.tenant_id, model_id)
    except ValueError as e:
        error_msg = str(e)
        # If model not found, suggest available models
        if "Unknown model" in error_msg:
            available = provider_manager.list_models()
            model_ids = [m.id for m in available]
            error_msg += f". Available models: {', '.join(model_ids[:10])}"
        # If provider not configured, list what's needed
        elif "not configured" in error_msg:
            model_info = provider_manager.get_model(model_id)
            if model_info:
                error_msg = (
                    f"No API key configured for provider '{model_info.provider.value}'. "
                    f"Configure it via POST /api/v1/providers with your API key."
                )

        return ExecutionResult(
            execution_id=execution_id,
            agent_id=request.agent_id,
            model_used=model_id,
            response="",
            state="error",
            steps=0,
            duration_ms=_elapsed_ms(start_time),
            events=[],
            checkpoints=0,
            error=error_msg,
        ).to_dict()

    # -- 5. Build state machine with time-travel --------------------------------
    from aria_core.runtime.state_machine import AgentStateMachine
    from aria_core.runtime.time_travel import TimeTravel

    time_travel = TimeTravel()
    event_collector = _EventCollector()

    # Composite callback: feed both the event collector and time-travel
    async def _composite_callback(event_type: str, payload: dict[str, Any]) -> None:
        await event_collector(event_type, payload)
        tt_callback = time_travel.as_event_callback()
        await tt_callback(event_type, payload)

    # Build agent config from registry or defaults
    fsm_config = _build_agent_config(model_id, request, agent_config)

    machine = AgentStateMachine(
        router=_DirectRouter(),
        planner=_NoOpPlanner(),
        executor=_NoOpExecutor(),
        adapter=adapter,
        config=fsm_config,
        event_callback=_composite_callback,
    )

    # -- 6. Execute with timeout ------------------------------------------------
    conversation_id = (
        UUID(request.conversation_id) if request.conversation_id else None
    )

    result_error: str | None = None
    response_text = ""
    final_state = "error"
    step_count = 0

    try:
        result = await asyncio.wait_for(
            machine.process_message(
                message=request.message,
                conversation_id=conversation_id,
                tenant_id=user.tenant_id,
                config=fsm_config,
            ),
            timeout=fsm_config.timeout_seconds,
        )

        response_text = result.response or ""
        final_state = result.state.value
        step_count = result.context.step_count
        result_error = result.error

    except asyncio.TimeoutError:
        final_state = "error"
        result_error = (
            f"Execution timed out after {fsm_config.timeout_seconds}s. "
            f"Partial events captured: {len(event_collector.events)}"
        )
    except Exception as e:
        final_state = "error"
        result_error = f"Execution failed: {str(e)}"

    # -- 7. Meter usage ---------------------------------------------------------
    try:
        from aria_core.billing.meter import UsageMeter

        # Best-effort metering — don't fail the request if metering fails
        meter = UsageMeter()
        meter.record(user.tenant_id, "agent_run")
        meter.record(user.tenant_id, "api_call")
    except Exception:
        pass

    # -- 8. Package result ------------------------------------------------------
    checkpoint_count = len(time_travel.list_checkpoints())

    return ExecutionResult(
        execution_id=execution_id,
        agent_id=request.agent_id,
        model_used=model_id,
        response=response_text,
        state=final_state,
        steps=step_count,
        duration_ms=_elapsed_ms(start_time),
        events=event_collector.events,
        checkpoints=checkpoint_count,
        error=result_error,
    ).to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> int:
    """Calculate elapsed milliseconds from a monotonic start time."""
    return int((time.monotonic() - start) * 1000)


def _resolve_model(
    request: ExecutionRequest,
    agent_config: dict[str, Any] | None,
) -> str:
    """Determine which model to use, in priority order:

    1. request.model (explicit override)
    2. agent_config["model"] (from registry)
    3. default "gpt-4"
    """
    if request.model:
        return request.model
    if agent_config and agent_config.get("model"):
        return agent_config["model"]
    return "gpt-4"


def _build_agent_config(
    model_id: str,
    request: ExecutionRequest,
    agent_config: dict[str, Any] | None,
) -> AgentConfig:
    """Build an AgentConfig from the resolved model and optional agent registry data."""
    kwargs: dict[str, Any] = {"model": model_id}

    if agent_config:
        if agent_config.get("system_prompt"):
            kwargs["system_prompt"] = agent_config["system_prompt"]
        if agent_config.get("max_steps"):
            kwargs["max_steps"] = agent_config["max_steps"]
        if agent_config.get("temperature") is not None:
            kwargs["temperature"] = agent_config["temperature"]

    return AgentConfig(**kwargs)
