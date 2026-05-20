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
