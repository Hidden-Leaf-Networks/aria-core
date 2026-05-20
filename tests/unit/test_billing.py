"""Tests for billing — usage metering and Stripe adapter."""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from aria_core.billing.meter import UsageMeter, METER_TYPES
from aria_core.billing.stripe_adapter import StripeAdapter


class TestUsageMeter:
    def test_record_and_report(self) -> None:
        meter = UsageMeter()
        tid = uuid4()

        meter.record(tid, "api_call")
        meter.record(tid, "api_call")
        meter.record(tid, "event")

        report = meter.get_report(tid, since=0)
        assert report.totals["api_call"] == 2
        assert report.totals["event"] == 1
        assert report.record_count == 3

    def test_tenant_isolation(self) -> None:
        meter = UsageMeter()
        t1, t2 = uuid4(), uuid4()

        meter.record(t1, "api_call", quantity=10)
        meter.record(t2, "api_call", quantity=5)

        r1 = meter.get_report(t1, since=0)
        r2 = meter.get_report(t2, since=0)
        assert r1.totals["api_call"] == 10
        assert r2.totals["api_call"] == 5

    def test_invalid_meter_type_raises(self) -> None:
        meter = UsageMeter()
        with pytest.raises(ValueError, match="Unknown meter type"):
            meter.record(uuid4(), "invalid_type")

    def test_negative_quantity_raises(self) -> None:
        meter = UsageMeter()
        with pytest.raises(ValueError, match="non-negative"):
            meter.record(uuid4(), "api_call", quantity=-1)

    def test_report_windowed(self) -> None:
        meter = UsageMeter()
        tid = uuid4()

        meter.record(tid, "api_call")
        # Report with since=future should show 0
        future = time.time() + 3600
        report = meter.get_report(tid, since=future)
        assert report.record_count == 0

    def test_get_all_reports(self) -> None:
        meter = UsageMeter()
        t1, t2 = uuid4(), uuid4()

        meter.record(t1, "api_call")
        meter.record(t2, "event")

        reports = meter.get_all_reports(since=0)
        assert len(reports) == 2

    def test_flush(self) -> None:
        meter = UsageMeter()
        tid = uuid4()

        meter.record(tid, "api_call")
        meter.record(tid, "api_call")
        assert meter.get_report(tid, since=0).record_count == 2

        flushed = meter.flush(tid)
        assert flushed == 2
        assert meter.get_report(tid, since=0).record_count == 0

    def test_flush_before_timestamp(self) -> None:
        meter = UsageMeter()
        tid = uuid4()

        meter.record(tid, "api_call")
        time.sleep(0.01)
        cutoff = time.time()
        time.sleep(0.01)
        meter.record(tid, "api_call")

        flushed = meter.flush(tid, before=cutoff)
        assert flushed == 1
        assert meter.get_report(tid, since=0).record_count == 1

    def test_report_to_dict(self) -> None:
        meter = UsageMeter()
        tid = uuid4()
        meter.record(tid, "agent_run")

        report = meter.get_report(tid, since=0)
        d = report.to_dict()
        assert d["tenant_id"] == str(tid)
        assert "period_start" in d
        assert d["totals"]["agent_run"] == 1

    def test_as_event_handler(self) -> None:
        """Meter can act as EventStore subscriber."""
        import asyncio

        meter = UsageMeter()
        tid = uuid4()
        handler = meter.as_event_handler(tid)

        asyncio.get_event_loop().run_until_complete(handler("agent.start", {}))
        asyncio.get_event_loop().run_until_complete(handler("plan.started", {}))
        asyncio.get_event_loop().run_until_complete(handler("step.complete", {}))

        report = meter.get_report(tid, since=0)
        # agent.start = 1 event + 1 agent_run
        # plan.started = 1 event + 1 plan_execution
        # step.complete = 1 event
        assert report.totals["event"] == 3
        assert report.totals["agent_run"] == 1
        assert report.totals["plan_execution"] == 1


class TestStripeAdapter:
    def test_register_subscription(self) -> None:
        adapter = StripeAdapter()
        tid = uuid4()
        adapter.register_subscription(tid, "api_call", "si_abc123")

        mapping = adapter.get_subscription_mapping(tid)
        assert mapping["api_call"] == "si_abc123"

    async def test_report_usage(self) -> None:
        meter = UsageMeter()
        adapter = StripeAdapter(meter=meter)
        tid = uuid4()

        adapter.register_subscription(tid, "api_call", "si_api")
        adapter.register_subscription(tid, "event", "si_evt")

        meter.record(tid, "api_call", quantity=100)
        meter.record(tid, "event", quantity=500)
        meter.record(tid, "agent_run", quantity=5)  # No subscription for this

        payloads = await adapter.report_usage(tid, since=0)
        assert len(payloads) == 2  # Only api_call and event (agent_run has no sub)

        api_payload = next(p for p in payloads if p["subscription_item"] == "si_api")
        assert api_payload["quantity"] == 100
        assert api_payload["action"] == "set"

    async def test_report_all_tenants(self) -> None:
        meter = UsageMeter()
        adapter = StripeAdapter(meter=meter)
        t1, t2 = uuid4(), uuid4()

        adapter.register_subscription(t1, "api_call", "si_t1_api")
        adapter.register_subscription(t2, "api_call", "si_t2_api")

        meter.record(t1, "api_call", quantity=50)
        meter.record(t2, "api_call", quantity=75)

        results = await adapter.report_all_tenants(since=0)
        assert len(results) == 2
        assert results[t1][0]["quantity"] == 50
        assert results[t2][0]["quantity"] == 75

    async def test_no_api_key_raises_on_send(self) -> None:
        from aria_core.billing.stripe_adapter import StripeAdapterError

        adapter = StripeAdapter()
        with pytest.raises(StripeAdapterError, match="No Stripe API key"):
            await adapter.send_to_stripe([{"subscription_item": "si_x", "quantity": 1}])


class TestWebhookHandler:
    async def test_handle_known_event(self) -> None:
        from aria_core.billing.stripe_adapter import WebhookHandler

        adapter = StripeAdapter()
        handler = WebhookHandler(adapter)

        received = []
        handler.on("customer.subscription.created", lambda e: received.append(e))

        result = await handler.handle({
            "type": "customer.subscription.created",
            "data": {"object": {"id": "sub_123"}},
        })

        assert result["handled"] is True
        assert result["handler_count"] == 1
        assert len(received) == 1

    async def test_handle_unknown_event(self) -> None:
        from aria_core.billing.stripe_adapter import WebhookHandler

        adapter = StripeAdapter()
        handler = WebhookHandler(adapter)

        result = await handler.handle({"type": "unknown.event"})
        assert result["handled"] is False
