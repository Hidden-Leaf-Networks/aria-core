"""Tests for phone agent (ARIA-312)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.phone.agent import CallConfig, CallRecord, PhoneAgent


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_config(**overrides) -> CallConfig:
    defaults = {
        "provider": "twilio",
        "api_key": "test-key",
        "api_secret": "test-secret",
        "from_number": "+15551234567",
    }
    return CallConfig(**{**defaults, **overrides})


def _make_agent(**overrides) -> PhoneAgent:
    return PhoneAgent(config=_make_config(**overrides), tenant_id=uuid4())


# ── CallConfig ────────────────────────────────────────────────────────

class TestCallConfig:
    def test_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.max_duration_seconds == 600
        assert cfg.recording_enabled is False
        assert cfg.webhook_url is None

    def test_all_providers(self) -> None:
        for provider in ("plivo", "twilio", "vonage"):
            cfg = _make_config(provider=provider)
            assert cfg.provider == provider

    def test_invalid_provider(self) -> None:
        with pytest.raises(Exception):
            _make_config(provider="invalid")

    def test_custom_values(self) -> None:
        cfg = _make_config(
            max_duration_seconds=120,
            recording_enabled=True,
            webhook_url="https://example.com/hook",
        )
        assert cfg.max_duration_seconds == 120
        assert cfg.recording_enabled is True
        assert cfg.webhook_url == "https://example.com/hook"


# ── CallRecord ────────────────────────────────────────────────────────

class TestCallRecord:
    def test_defaults(self) -> None:
        rec = CallRecord(
            tenant_id=uuid4(),
            direction="outbound",
            from_number="+1555",
            to_number="+1666",
        )
        assert rec.status == "initiated"
        assert rec.duration_seconds == 0
        assert rec.metadata == {}
        assert rec.id is not None

    def test_all_statuses(self) -> None:
        for status in (
            "initiated", "ringing", "answered", "completed",
            "failed", "busy", "no_answer",
        ):
            rec = CallRecord(
                tenant_id=uuid4(),
                direction="inbound",
                from_number="+1",
                to_number="+2",
                status=status,
            )
            assert rec.status == status


# ── PhoneAgent ────────────────────────────────────────────────────────

class TestPhoneAgent:
    async def test_initiate_call(self) -> None:
        agent = _make_agent()
        rec = await agent.initiate_call("+19995550000", "demo")
        assert rec.direction == "outbound"
        assert rec.to_number == "+19995550000"
        assert rec.status == "ringing"
        assert rec.metadata["purpose"] == "demo"

    async def test_handle_inbound(self) -> None:
        agent = _make_agent()
        rec = await agent.handle_inbound("+18885550000", {"source": "website"})
        assert rec.direction == "inbound"
        assert rec.from_number == "+18885550000"
        assert rec.status == "ringing"
        assert rec.metadata == {"source": "website"}

    async def test_handle_inbound_no_metadata(self) -> None:
        agent = _make_agent()
        rec = await agent.handle_inbound("+18885550000")
        assert rec.metadata == {}

    async def test_end_call(self) -> None:
        agent = _make_agent()
        rec = await agent.initiate_call("+19995550000", "end-test")
        ended = await agent.end_call(rec.id)
        assert ended.status == "completed"
        assert ended.ended_at is not None
        assert ended.duration_seconds >= 0

    async def test_end_call_not_found(self) -> None:
        agent = _make_agent()
        with pytest.raises(ValueError, match="not found"):
            await agent.end_call(uuid4())

    async def test_get_call(self) -> None:
        agent = _make_agent()
        rec = await agent.initiate_call("+1", "get-test")
        assert agent.get_call(rec.id) is not None
        assert agent.get_call(uuid4()) is None

    async def test_list_calls_unfiltered(self) -> None:
        agent = _make_agent()
        await agent.initiate_call("+1", "a")
        await agent.handle_inbound("+2")
        assert len(agent.list_calls()) == 2

    async def test_list_calls_by_direction(self) -> None:
        agent = _make_agent()
        await agent.initiate_call("+1", "a")
        await agent.handle_inbound("+2")
        assert len(agent.list_calls(direction="outbound")) == 1
        assert len(agent.list_calls(direction="inbound")) == 1

    async def test_list_calls_by_status(self) -> None:
        agent = _make_agent()
        rec = await agent.initiate_call("+1", "a")
        await agent.end_call(rec.id)
        assert len(agent.list_calls(status="completed")) == 1
        assert len(agent.list_calls(status="ringing")) == 0

    async def test_list_calls_limit(self) -> None:
        agent = _make_agent()
        for i in range(5):
            await agent.initiate_call(f"+{i}", f"call-{i}")
        assert len(agent.list_calls(limit=3)) == 3

    async def test_stats_empty(self) -> None:
        agent = _make_agent()
        stats = agent.get_stats()
        assert stats["total_calls"] == 0
        assert stats["avg_duration_seconds"] == 0.0
        assert stats["success_rate"] == 0.0

    async def test_stats_with_calls(self) -> None:
        agent = _make_agent()
        rec1 = await agent.initiate_call("+1", "a")
        await agent.initiate_call("+2", "b")
        await agent.end_call(rec1.id)
        stats = agent.get_stats()
        assert stats["total_calls"] == 2
        assert stats["success_rate"] == 0.5
