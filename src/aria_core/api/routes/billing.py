"""Billing API routes — usage reports and subscription management."""

from __future__ import annotations

from typing import Any

from aria_core.api.auth import AuthUser, Role, require_role


async def get_usage(user: AuthUser, meter: Any) -> dict[str, Any]:
    """Get usage report for the authenticated tenant."""
    report = meter.get_report(user.tenant_id)
    return report.to_dict()


async def get_all_usage(user: AuthUser, meter: Any) -> list[dict[str, Any]]:
    """Get usage reports for all tenants. Admin only."""
    require_role(user, Role.ADMIN)
    reports = meter.get_all_reports()
    return [r.to_dict() for r in reports]


async def get_pricing_tiers() -> list[dict[str, Any]]:
    """Get all pricing tiers (public endpoint)."""
    from aria_core.billing.pricing import PricingCalculator
    return PricingCalculator().get_tiers()


async def calculate_pricing(
    api_calls: int = 0,
    events: int = 0,
    agent_runs: int = 0,
    agents: int = 0,
    tenants: int = 1,
    storage_gb: float = 0,
) -> dict[str, Any]:
    """Calculate pricing based on projected usage (public endpoint)."""
    from aria_core.billing.pricing import PricingCalculator
    return PricingCalculator().recommend_tier(
        api_calls=api_calls,
        events=events,
        agent_runs=agent_runs,
        agents=agents,
        tenants=tenants,
        storage_gb=storage_gb,
    )
