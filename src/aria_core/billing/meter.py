"""Usage metering — tracks per-tenant consumption for billing.

Meters:
- api_calls: REST API requests
- events: events emitted to the event store
- agent_runs: agent state machine executions
- plan_executions: plans executed
- ws_messages: WebSocket messages sent
- storage_bytes: approximate data stored

Usage:
    meter = UsageMeter()
    meter.record(tenant_id, "api_call")
    meter.record(tenant_id, "agent_run", quantity=1)
    report = meter.get_report(tenant_id)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID


METER_TYPES = {
    "api_call",
    "event",
    "agent_run",
    "plan_execution",
    "ws_message",
    "storage_bytes",
}


@dataclass
class UsageRecord:
    """Single usage record."""

    meter_type: str
    quantity: int
    timestamp: float


@dataclass
class UsageReport:
    """Aggregated usage report for a tenant."""

    tenant_id: UUID
    period_start: datetime
    period_end: datetime
    totals: dict[str, int] = field(default_factory=dict)
    record_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "totals": self.totals,
            "record_count": self.record_count,
        }


class UsageMeter:
    """In-memory usage meter with windowed aggregation.

    For production, records should be flushed to a persistent store
    (PostgreSQL, Redis, or Stripe directly) on a schedule.
    """

    def __init__(self) -> None:
        # {tenant_id: list[UsageRecord]}
        self._records: dict[UUID, list[UsageRecord]] = defaultdict(list)

    def record(
        self,
        tenant_id: UUID,
        meter_type: str,
        quantity: int = 1,
    ) -> None:
        """Record a usage event."""
        if meter_type not in METER_TYPES:
            raise ValueError(f"Unknown meter type: {meter_type}. Valid: {METER_TYPES}")
        if quantity < 0:
            raise ValueError("Quantity must be non-negative")

        self._records[tenant_id].append(
            UsageRecord(
                meter_type=meter_type,
                quantity=quantity,
                timestamp=time.time(),
            )
        )

    def get_report(
        self,
        tenant_id: UUID,
        since: float | None = None,
    ) -> UsageReport:
        """Get aggregated usage report for a tenant.

        Args:
            tenant_id: Tenant to report on.
            since: Unix timestamp. Only records after this time are included.
                   Defaults to beginning of current UTC day.
        """
        if since is None:
            now = datetime.now(timezone.utc)
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            since = start_of_day.timestamp()

        records = self._records.get(tenant_id, [])
        filtered = [r for r in records if r.timestamp >= since]

        totals: dict[str, int] = {}
        for r in filtered:
            totals[r.meter_type] = totals.get(r.meter_type, 0) + r.quantity

        return UsageReport(
            tenant_id=tenant_id,
            period_start=datetime.fromtimestamp(since, tz=timezone.utc),
            period_end=datetime.now(timezone.utc),
            totals=totals,
            record_count=len(filtered),
        )

    def get_all_reports(self, since: float | None = None) -> list[UsageReport]:
        """Get usage reports for all tenants."""
        return [self.get_report(tid, since) for tid in self._records]

    def flush(self, tenant_id: UUID | None = None, before: float | None = None) -> int:
        """Remove old records. Returns count of records flushed.

        Args:
            tenant_id: Flush only this tenant. None = all tenants.
            before: Flush records older than this timestamp.
        """
        count = 0
        targets = [tenant_id] if tenant_id else list(self._records.keys())

        for tid in targets:
            if before:
                old_len = len(self._records[tid])
                self._records[tid] = [r for r in self._records[tid] if r.timestamp >= before]
                count += old_len - len(self._records[tid])
            else:
                count += len(self._records[tid])
                self._records[tid] = []

        return count

    def as_event_handler(self, tenant_id: UUID) -> Any:
        """Return an EventStore-compatible handler that meters events."""
        meter = self

        async def handler(event_type: str, payload: dict[str, Any]) -> None:
            meter.record(tenant_id, "event")
            if event_type == "agent.start":
                meter.record(tenant_id, "agent_run")
            elif event_type.startswith("plan.") and event_type == "plan.started":
                meter.record(tenant_id, "plan_execution")

        return handler
