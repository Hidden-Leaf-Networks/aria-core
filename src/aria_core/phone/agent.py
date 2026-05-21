"""Phone agent — telephony abstraction for ARIA agents.

Supports Plivo, Twilio, and Vonage as telephony providers with a unified
interface for initiating outbound calls, handling inbound calls, and
managing call lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CallConfig(BaseModel):
    """Telephony provider configuration."""

    provider: Literal["plivo", "twilio", "vonage"]
    api_key: str
    api_secret: Optional[str] = None
    from_number: str
    webhook_url: Optional[str] = None
    max_duration_seconds: int = 600
    recording_enabled: bool = False


class CallRecord(BaseModel):
    """Immutable record of a phone call."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    direction: Literal["inbound", "outbound"]
    from_number: str
    to_number: str
    status: Literal[
        "initiated", "ringing", "answered", "completed", "failed", "busy", "no_answer"
    ] = "initiated"
    duration_seconds: int = 0
    recording_url: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class PhoneAgent:
    """Manages phone calls for a single tenant.

    All calls are simulated — no real telephony provider is contacted.
    This allows integration testing and white-label demos without
    requiring live credentials.
    """

    def __init__(self, config: CallConfig, tenant_id: UUID) -> None:
        self.config = config
        self.tenant_id = tenant_id
        self._calls: dict[UUID, CallRecord] = {}

    # ------------------------------------------------------------------
    # Call lifecycle
    # ------------------------------------------------------------------

    async def initiate_call(self, to_number: str, purpose: str) -> CallRecord:
        """Start a simulated outbound call."""
        record = CallRecord(
            tenant_id=self.tenant_id,
            direction="outbound",
            from_number=self.config.from_number,
            to_number=to_number,
            status="ringing",
            metadata={"purpose": purpose},
        )
        self._calls[record.id] = record
        return record

    async def handle_inbound(
        self, from_number: str, metadata: dict | None = None
    ) -> CallRecord:
        """Handle a simulated inbound call."""
        record = CallRecord(
            tenant_id=self.tenant_id,
            direction="inbound",
            from_number=from_number,
            to_number=self.config.from_number,
            status="ringing",
            metadata=metadata or {},
        )
        self._calls[record.id] = record
        return record

    async def end_call(self, call_id: UUID) -> CallRecord:
        """End an active call and mark it completed."""
        record = self._calls.get(call_id)
        if record is None:
            raise ValueError(f"Call {call_id} not found")

        now = datetime.now(timezone.utc)
        elapsed = int((now - record.started_at).total_seconds())
        updated = record.model_copy(
            update={
                "status": "completed",
                "ended_at": now,
                "duration_seconds": elapsed,
            }
        )
        self._calls[call_id] = updated
        return updated

    def get_call(self, call_id: UUID) -> CallRecord | None:
        """Retrieve a call record by ID."""
        return self._calls.get(call_id)

    def list_calls(
        self,
        direction: Literal["inbound", "outbound"] | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[CallRecord]:
        """List call records with optional filters."""
        results = list(self._calls.values())
        if direction is not None:
            results = [c for c in results if c.direction == direction]
        if status is not None:
            results = [c for c in results if c.status == status]
        return results[:limit]

    def get_stats(self) -> dict:
        """Return aggregate call statistics."""
        calls = list(self._calls.values())
        total = len(calls)
        if total == 0:
            return {
                "total_calls": 0,
                "avg_duration_seconds": 0.0,
                "success_rate": 0.0,
            }

        completed = [c for c in calls if c.status == "completed"]
        durations = [c.duration_seconds for c in completed]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        success_rate = len(completed) / total

        return {
            "total_calls": total,
            "avg_duration_seconds": avg_duration,
            "success_rate": success_rate,
        }
