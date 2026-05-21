"""Tests for OpenTelemetry tracing."""

from __future__ import annotations

import pytest

from aria_core.telemetry.tracer import AriaTracer, setup_tracing, _HAS_OTEL


class TestAriaTracer:
    def test_tracer_init(self) -> None:
        tracer = AriaTracer()
        # Should not crash even without OTEL
        assert isinstance(tracer.enabled, bool)

    def test_start_and_end_span_no_crash(self) -> None:
        """Tracer is safe to use even without OpenTelemetry installed."""
        tracer = AriaTracer()
        span = tracer.start_span("test.span", {"key": "value"})
        tracer.end_span("test.span")
        # No error = success

    def test_end_nonexistent_span(self) -> None:
        tracer = AriaTracer()
        tracer.end_span("nonexistent")  # Should not crash

    def test_record_event_no_crash(self) -> None:
        tracer = AriaTracer()
        tracer.start_span("test")
        tracer.record_event("test", "event.happened", {"data": "value"})
        tracer.end_span("test")

    async def test_event_callback_agent_lifecycle(self) -> None:
        """Full agent lifecycle through the event callback."""
        tracer = AriaTracer()
        cb = tracer.as_event_callback()

        # Simulate agent lifecycle
        await cb("agent.start", {"context_id": "ctx-1"})
        await cb("transition.complete", {"from": "idle", "to": "routing"})
        await cb("routing.complete", {"strategy": "direct", "intent": "simple_query", "confidence": 0.9})
        await cb("transition.complete", {"from": "routing", "to": "planning"})
        await cb("planning.complete", {"plan_id": "plan-1", "action_count": 3})
        await cb("transition.complete", {"from": "planning", "to": "executing_step"})
        await cb("step.complete", {"step_index": 0, "skill_name": "build", "success": True})
        await cb("step.complete", {"step_index": 1, "skill_name": "test", "success": True})
        await cb("approval.required", {"gate_name": "high-risk", "risk_score": 75})
        await cb("approval.granted", {})
        await cb("transition.complete", {"from": "executing_step", "to": "responding"})
        await cb("responding.complete", {"model": "gpt-4", "tokens": 150})
        await cb("agent.complete", {})

        # No errors = full lifecycle traced successfully

    async def test_event_callback_error_path(self) -> None:
        tracer = AriaTracer()
        cb = tracer.as_event_callback()

        await cb("agent.start", {"context_id": "ctx-err"})
        await cb("agent.error", {"error": "Something went wrong"})
        # Error span properly closed

    def test_setup_tracing_no_crash(self) -> None:
        """setup_tracing is safe even without OTEL."""
        result = setup_tracing(service_name="test")
        # Returns None if OTEL not installed, TracerProvider if it is


@pytest.mark.skipif(not _HAS_OTEL, reason="opentelemetry not installed")
class TestWithOTEL:
    def test_tracer_is_enabled(self) -> None:
        setup_tracing(service_name="test-aria", console_export=False)
        tracer = AriaTracer()
        assert tracer.enabled is True

    def test_span_lifecycle(self) -> None:
        setup_tracing(service_name="test-aria")
        tracer = AriaTracer()
        span = tracer.start_span("test.real", {"key": "value"})
        assert span is not None
        tracer.end_span("test.real")

    def test_span_with_error(self) -> None:
        setup_tracing(service_name="test-aria")
        tracer = AriaTracer()
        tracer.start_span("test.error")
        tracer.end_span("test.error", error="something broke")
