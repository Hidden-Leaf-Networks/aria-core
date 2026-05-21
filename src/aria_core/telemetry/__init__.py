"""OpenTelemetry integration for Aria Core.

Provides automatic tracing for:
- FSM state transitions
- LLM adapter calls
- Plan lifecycle events
- Skill executions
- Approval decisions
- Deep Bridge consensus

Usage:
    from aria_core.telemetry import setup_tracing, AriaTracer

    setup_tracing(service_name="aria-core", endpoint="http://jaeger:4317")
    tracer = AriaTracer()
    # Pass tracer as event_callback to machines
"""

from aria_core.telemetry.tracer import AriaTracer, setup_tracing

__all__ = ["AriaTracer", "setup_tracing"]
