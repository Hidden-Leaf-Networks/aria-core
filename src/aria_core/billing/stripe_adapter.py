"""Stripe adapter — reports tenant usage to Stripe for metered billing.

Requires STRIPE_API_KEY environment variable.
Uses Stripe's Usage Records API for metered subscriptions.

Usage:
    adapter = StripeAdapter(api_key="sk_...", meter=usage_meter)
    await adapter.report_usage(tenant_id, subscription_item_id)

Stripe Setup:
1. Create a Product with metered pricing
2. Create Prices for each meter type (api_calls, events, agent_runs, etc.)
3. Subscribe tenants and store subscription_item_id per meter
4. Call report_usage periodically (e.g., hourly cron)
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from aria_core.billing.meter import UsageMeter


class StripeAdapterError(Exception):
    pass


class StripeAdapter:
    """Reports usage to Stripe's metered billing API.

    This adapter is Stripe-ready but doesn't import stripe directly —
    it produces the API call payload that can be sent via stripe SDK
    or raw HTTP. This keeps the dependency optional.
    """

    def __init__(
        self,
        api_key: str | None = None,
        meter: UsageMeter | None = None,
    ) -> None:
        self._api_key = api_key
        self._meter = meter or UsageMeter()
        # {tenant_id: {meter_type: subscription_item_id}}
        self._subscriptions: dict[UUID, dict[str, str]] = {}

    def register_subscription(
        self,
        tenant_id: UUID,
        meter_type: str,
        subscription_item_id: str,
    ) -> None:
        """Map a tenant's meter type to a Stripe subscription item."""
        if tenant_id not in self._subscriptions:
            self._subscriptions[tenant_id] = {}
        self._subscriptions[tenant_id][meter_type] = subscription_item_id

    def get_subscription_mapping(self, tenant_id: UUID) -> dict[str, str]:
        """Get all subscription mappings for a tenant."""
        return dict(self._subscriptions.get(tenant_id, {}))

    async def report_usage(
        self,
        tenant_id: UUID,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Generate Stripe usage record payloads for a tenant.

        Returns list of payloads ready for Stripe's create_usage_record API.
        In production, send these via stripe.SubscriptionItem.create_usage_record().
        """
        report = self._meter.get_report(tenant_id, since=since)
        subscriptions = self._subscriptions.get(tenant_id, {})

        payloads: list[dict[str, Any]] = []
        timestamp = int(time.time())

        for meter_type, quantity in report.totals.items():
            if quantity <= 0:
                continue

            sub_item_id = subscriptions.get(meter_type)
            if not sub_item_id:
                continue

            payloads.append({
                "subscription_item": sub_item_id,
                "quantity": quantity,
                "timestamp": timestamp,
                "action": "set",
            })

        return payloads

    async def report_all_tenants(
        self, since: float | None = None
    ) -> dict[UUID, list[dict[str, Any]]]:
        """Generate usage reports for all subscribed tenants."""
        results: dict[UUID, list[dict[str, Any]]] = {}
        for tenant_id in self._subscriptions:
            payloads = await self.report_usage(tenant_id, since=since)
            if payloads:
                results[tenant_id] = payloads
        return results

    async def send_to_stripe(
        self, payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Send usage records to Stripe.

        Requires stripe SDK: pip install stripe
        Returns list of Stripe API responses.
        """
        if not self._api_key:
            raise StripeAdapterError("No Stripe API key configured")

        try:
            import stripe

            stripe.api_key = self._api_key
        except ImportError:
            raise StripeAdapterError("stripe SDK not installed: pip install stripe")

        responses: list[dict[str, Any]] = []
        for payload in payloads:
            record = stripe.SubscriptionItem.create_usage_record(
                payload["subscription_item"],
                quantity=payload["quantity"],
                timestamp=payload["timestamp"],
                action=payload["action"],
            )
            responses.append({"id": record.id, "quantity": record.quantity})

        return responses


class WebhookHandler:
    """Handles Stripe webhook events for subscription lifecycle.

    Events handled:
    - customer.subscription.created: New tenant subscription
    - customer.subscription.updated: Plan changes
    - customer.subscription.deleted: Cancellation
    - invoice.payment_failed: Payment issues
    """

    def __init__(self, adapter: StripeAdapter) -> None:
        self._adapter = adapter
        self._handlers: dict[str, list[Any]] = {}

    def on(self, event_type: str, handler: Any) -> None:
        """Register a handler for a Stripe event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        """Process a Stripe webhook event.

        Args:
            event: Parsed Stripe event payload (after signature verification).

        Returns:
            {"handled": bool, "event_type": str}
        """
        event_type = event.get("type", "")
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            result = handler(event)
            if hasattr(result, "__await__"):
                await result

        return {
            "handled": len(handlers) > 0,
            "event_type": event_type,
            "handler_count": len(handlers),
        }
