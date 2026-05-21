"""OpenTelemetry tracer for Aria Core agent execution.

Creates spans for every significant operation in the agent lifecycle.
Exports to any OTLP-compatible backend (Jaeger, Grafana Tempo, Datadog, etc).

Span hierarchy:
  agent.run
  ├── fsm.transition (idle → routing)
  ├── router.route
  ├── fsm.transition (routing → planning)
  ├── planner.create_plan
  ├── fsm.transition (planning → executing)
  ├── executor.step
  │   ├── skill.execute
  │   └── risk.calculate
  ├── approval.evaluate
  ├── adapter.generate (LLM call)
  ├── deep_bridge.validate
  └── fsm.transition (responding → complete)
"""

from __future__ import annotations

from typing import Any


# Graceful fallback when opentelemetry is not installed
_HAS_OTEL = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode, Span

    _HAS_OTEL = True
except ImportError:
    pass


def setup_tracing(
    service_name: str = "aria-core",
    endpoint: str | None = None,
    console_export: bool = False,
) -> Any:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces
        endpoint: OTLP endpoint (e.g. "http://jaeger:4317"). None = no export.
        console_export: Also print spans to console (dev mode)

    Returns:
        TracerProvider instance
    """
    if not _HAS_OTEL:
        return None

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass

    if console_export:
        try:
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        except ImportError:
            pass

    trace.set_tracer_provider(provider)
    return provider


class AriaTracer:
    """Instruments aria-core operations with OpenTelemetry spans.

    Can be used as:
    1. Event callback for AgentStateMachine
    2. Standalone tracer for manual instrumentation
    """

    def __init__(self, tracer_name: str = "aria_core") -> None:
        self._tracer_name = tracer_name
        self._tracer = None
        if _HAS_OTEL:
            self._tracer = trace.get_tracer(tracer_name)
        # Active span tracking
        self._spans: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent_key: str | None = None,
    ) -> Any:
        """Start a new span. Returns the span (or None if not enabled)."""
        if not self._tracer:
            return None

        kwargs: dict[str, Any] = {}
        if attributes:
            kwargs["attributes"] = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in attributes.items()
            }

        span = self._tracer.start_span(name, **kwargs)
        self._spans[name] = span
        return span

    def end_span(self, name: str, error: str | None = None) -> None:
        """End a span by name."""
        span = self._spans.pop(name, None)
        if span and _HAS_OTEL:
            if error:
                span.set_status(StatusCode.ERROR, error)
                span.set_attribute("error.message", error)
            else:
                span.set_status(StatusCode.OK)
            span.end()

    def record_event(
        self,
        span_name: str,
        event_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record an event on an existing span."""
        span = self._spans.get(span_name)
        if span and _HAS_OTEL:
            span.add_event(
                event_name,
                attributes={
                    k: str(v) for k, v in (attributes or {}).items()
                },
            )

    async def event_callback(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        """EventCallback-compatible handler for AgentStateMachine.

        Automatically creates and manages spans based on event types.
        Wire this into: AgentStateMachine(event_callback=tracer.event_callback)
        """
        if not self._tracer:
            return

        if event_type == "agent.start":
            self.start_span("agent.run", {
                "agent.context_id": payload.get("context_id", ""),
            })

        elif event_type == "transition.complete":
            from_state = payload.get("from", "")
            to_state = payload.get("to", "")
            span = self.start_span(f"fsm.{from_state}_to_{to_state}", {
                "fsm.from_state": from_state,
                "fsm.to_state": to_state,
            })
            if span:
                span.end()

        elif event_type == "routing.complete":
            span = self.start_span("router.route", {
                "router.strategy": payload.get("strategy", ""),
                "router.intent": payload.get("intent", ""),
                "router.confidence": payload.get("confidence", 0),
            })
            if span:
                span.end()

        elif event_type == "planning.complete":
            span = self.start_span("planner.create_plan", {
                "plan.id": payload.get("plan_id", ""),
                "plan.actions": payload.get("action_count", 0),
            })
            if span:
                span.end()

        elif event_type == "step.complete":
            span = self.start_span("executor.step", {
                "step.index": payload.get("step_index", 0),
                "step.skill": payload.get("skill_name", ""),
                "step.success": payload.get("success", False),
            })
            if span:
                if not payload.get("success", True):
                    span.set_status(StatusCode.ERROR, payload.get("error", ""))
                span.end()

        elif event_type == "approval.required":
            self.start_span("approval.evaluate", {
                "approval.gate": payload.get("gate_name", ""),
                "approval.risk_score": payload.get("risk_score", 0),
            })

        elif event_type == "approval.granted":
            self.end_span("approval.evaluate")

        elif event_type == "responding.complete":
            span = self.start_span("adapter.generate", {
                "llm.model": payload.get("model", ""),
                "llm.tokens": payload.get("tokens", 0),
            })
            if span:
                span.end()

        elif event_type == "agent.complete":
            self.end_span("agent.run")

        elif event_type == "agent.error":
            self.end_span("agent.run", error=payload.get("error", "Unknown error"))

    def as_event_callback(self) -> Any:
        """Return the event callback for wiring into AgentStateMachine."""
        return self.event_callback
