"""Billing module — per-tenant usage metering and Stripe integration.

Provides:
- UsageMeter: tracks API calls, events, agent runs per tenant
- StripeAdapter: reports usage to Stripe for metered billing
- Billing webhook handler for subscription lifecycle

Usage:
    from aria_core.billing import UsageMeter, StripeAdapter

    meter = UsageMeter()
    meter.record(tenant_id, "api_call")
    report = meter.get_report(tenant_id)
"""

from aria_core.billing.meter import UsageMeter, UsageReport
from aria_core.billing.stripe_adapter import StripeAdapter

__all__ = [
    "StripeAdapter",
    "UsageMeter",
    "UsageReport",
]
