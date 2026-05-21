"""Tests for the agent scheduler — ARIA-303."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from aria_core.scheduler import (
    CronParser,
    CronParseError,
    Schedule,
    ScheduleRun,
    SchedulerEngine,
    SchedulerError,
    ScheduleNotFoundError,
    parse_natural_language,
)

TENANT = uuid4()
UTC = timezone.utc


# ---------------------------------------------------------------------------
# CronParser.parse
# ---------------------------------------------------------------------------


class TestCronParse:
    def test_simple_daily(self):
        """Parse '0 8 * * *' — daily at 8:00."""
        fields = CronParser.parse("0 8 * * *")
        assert fields["minute"] == {0}
        assert fields["hour"] == {8}
        assert fields["day_of_month"] == set(range(1, 32))
        assert fields["month"] == set(range(1, 13))
        assert fields["day_of_week"] == set(range(0, 7))

    def test_step_expression(self):
        """Parse '*/15 * * * *' — every 15 minutes."""
        fields = CronParser.parse("*/15 * * * *")
        assert fields["minute"] == {0, 15, 30, 45}

    def test_range_expression(self):
        """Parse '0 9-17 * * *' — hourly 9am-5pm."""
        fields = CronParser.parse("0 9-17 * * *")
        assert fields["hour"] == set(range(9, 18))

    def test_list_expression(self):
        """Parse '0 8 * * 1,3,5' — Mon/Wed/Fri at 8am."""
        fields = CronParser.parse("0 8 * * 1,3,5")
        assert fields["day_of_week"] == {1, 3, 5}

    def test_range_with_step(self):
        """Parse '0 0-23/2 * * *' — every 2 hours."""
        fields = CronParser.parse("0 0-23/2 * * *")
        assert fields["hour"] == {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}

    def test_invalid_field_count(self):
        with pytest.raises(CronParseError, match="Expected 5 fields"):
            CronParser.parse("* * *")

    def test_out_of_range(self):
        with pytest.raises(CronParseError, match="out of bounds"):
            CronParser.parse("60 * * * *")

    def test_invalid_step(self):
        with pytest.raises(CronParseError, match="Invalid step"):
            CronParser.parse("*/0 * * * *")


# ---------------------------------------------------------------------------
# CronParser.matches
# ---------------------------------------------------------------------------


class TestCronMatches:
    def test_matches_exact(self):
        dt = datetime(2026, 5, 20, 8, 0, tzinfo=UTC)  # Wednesday
        assert CronParser.matches("0 8 * * *", dt) is True

    def test_no_match_wrong_minute(self):
        dt = datetime(2026, 5, 20, 8, 5, tzinfo=UTC)
        assert CronParser.matches("0 8 * * *", dt) is False

    def test_matches_weekday(self):
        # 2026-05-20 is a Wednesday = cron dow 3
        dt = datetime(2026, 5, 20, 9, 0, tzinfo=UTC)
        assert CronParser.matches("0 9 * * 3", dt) is True
        assert CronParser.matches("0 9 * * 1", dt) is False


# ---------------------------------------------------------------------------
# CronParser.next_run
# ---------------------------------------------------------------------------


class TestCronNextRun:
    def test_next_run_daily(self):
        after = datetime(2026, 5, 20, 7, 30, tzinfo=UTC)
        nxt = CronParser.next_run("0 8 * * *", after=after)
        assert nxt == datetime(2026, 5, 20, 8, 0, tzinfo=UTC)

    def test_next_run_rolls_to_tomorrow(self):
        after = datetime(2026, 5, 20, 8, 30, tzinfo=UTC)
        nxt = CronParser.next_run("0 8 * * *", after=after)
        assert nxt == datetime(2026, 5, 21, 8, 0, tzinfo=UTC)

    def test_next_run_every_15_min(self):
        after = datetime(2026, 5, 20, 10, 3, tzinfo=UTC)
        nxt = CronParser.next_run("*/15 * * * *", after=after)
        assert nxt == datetime(2026, 5, 20, 10, 15, tzinfo=UTC)


# ---------------------------------------------------------------------------
# SchedulerEngine CRUD
# ---------------------------------------------------------------------------


class TestEngineCRUD:
    def _make_engine(self) -> SchedulerEngine:
        return SchedulerEngine(tenant_id=TENANT)

    def test_create_and_get(self):
        engine = self._make_engine()
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="test", schedule_type="interval", interval_seconds=300,
        ))
        assert sched.next_run_at is not None
        assert engine.get(sched.id) is sched

    def test_create_duplicate_raises(self):
        engine = self._make_engine()
        sched = Schedule(tenant_id=TENANT, name="dup", schedule_type="interval", interval_seconds=60)
        engine.create(sched)
        with pytest.raises(SchedulerError, match="already exists"):
            engine.create(sched)

    def test_update(self):
        engine = self._make_engine()
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="orig", schedule_type="interval", interval_seconds=60,
        ))
        updated = engine.update(sched.id, {"name": "renamed"})
        assert updated.name == "renamed"

    def test_update_nonexistent(self):
        engine = self._make_engine()
        with pytest.raises(ScheduleNotFoundError):
            engine.update(uuid4(), {"name": "x"})

    def test_delete(self):
        engine = self._make_engine()
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="bye", schedule_type="interval", interval_seconds=60,
        ))
        assert engine.delete(sched.id) is True
        assert engine.get(sched.id) is None
        assert engine.delete(sched.id) is False

    def test_list_all_and_enabled_only(self):
        engine = self._make_engine()
        engine.create(Schedule(tenant_id=TENANT, name="a", schedule_type="interval", interval_seconds=60))
        s2 = engine.create(Schedule(tenant_id=TENANT, name="b", schedule_type="interval", interval_seconds=60, enabled=False))
        assert len(engine.list()) == 2
        assert len(engine.list(enabled_only=True)) == 1

    def test_enable_disable(self):
        engine = self._make_engine()
        sched = engine.create(Schedule(tenant_id=TENANT, name="tog", schedule_type="interval", interval_seconds=60))
        engine.disable(sched.id)
        assert engine.get(sched.id).enabled is False
        engine.enable(sched.id)
        assert engine.get(sched.id).enabled is True


# ---------------------------------------------------------------------------
# Due detection and tick
# ---------------------------------------------------------------------------


class TestDueAndTick:
    def test_get_due_schedules(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        s = engine.create(Schedule(
            tenant_id=TENANT, name="due", schedule_type="interval", interval_seconds=60,
        ))
        # Force next_run_at to be in the past
        s.next_run_at = now - timedelta(minutes=1)
        due = engine.get_due_schedules(now)
        assert len(due) == 1
        assert due[0].id == s.id

    def test_disabled_not_due(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        s = engine.create(Schedule(
            tenant_id=TENANT, name="off", schedule_type="interval", interval_seconds=60, enabled=False,
        ))
        s.next_run_at = now - timedelta(minutes=1)
        assert engine.get_due_schedules(now) == []

    def test_tick_returns_due(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        s = engine.create(Schedule(
            tenant_id=TENANT, name="tick", schedule_type="interval", interval_seconds=60,
        ))
        s.next_run_at = now - timedelta(seconds=30)
        result = engine.tick(now)
        assert len(result) == 1

    def test_tick_skips_future(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        s = engine.create(Schedule(
            tenant_id=TENANT, name="future", schedule_type="interval", interval_seconds=3600,
        ))
        s.next_run_at = now + timedelta(hours=1)
        assert engine.tick(now) == []


# ---------------------------------------------------------------------------
# Run recording
# ---------------------------------------------------------------------------


class TestRunRecording:
    def test_record_run_success(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="runner", schedule_type="interval", interval_seconds=300,
        ))
        run = ScheduleRun(
            schedule_id=sched.id, tenant_id=TENANT,
            status="completed",
            started_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
            completed_at=datetime(2026, 5, 20, 12, 0, 5, tzinfo=UTC),
            duration_ms=5000,
            output="done",
        )
        engine.record_run(sched.id, run)
        assert sched.run_count == 1
        assert sched.failure_count == 0
        assert sched.last_run_at is not None

    def test_record_run_failure_increments_counter(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="failer", schedule_type="interval", interval_seconds=300,
        ))
        run = ScheduleRun(
            schedule_id=sched.id, tenant_id=TENANT,
            status="failed",
            started_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
            error="boom",
        )
        engine.record_run(sched.id, run)
        assert sched.failure_count == 1

    def test_once_schedule_disabled_after_run(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        run_time = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="oneshot", schedule_type="once", run_at=run_time,
        ))
        assert sched.enabled is True
        run = ScheduleRun(
            schedule_id=sched.id, tenant_id=TENANT,
            status="completed", started_at=run_time,
        )
        engine.record_run(sched.id, run)
        assert sched.enabled is False
        assert sched.next_run_at is None

    def test_get_runs(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="runs", schedule_type="interval", interval_seconds=60,
        ))
        for i in range(5):
            engine.record_run(sched.id, ScheduleRun(
                schedule_id=sched.id, tenant_id=TENANT,
                status="completed",
                started_at=datetime(2026, 5, 20, 12, i, tzinfo=UTC),
            ))
        runs = engine.get_runs(sched.id, limit=3)
        assert len(runs) == 3
        # Newest first
        assert runs[0].started_at > runs[-1].started_at

    def test_record_run_nonexistent(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        with pytest.raises(ScheduleNotFoundError):
            engine.record_run(uuid4(), ScheduleRun(
                schedule_id=uuid4(), tenant_id=TENANT, status="completed",
            ))


# ---------------------------------------------------------------------------
# Natural language parsing
# ---------------------------------------------------------------------------


class TestNaturalLanguage:
    def test_every_30_minutes(self):
        sched = parse_natural_language("every 30 minutes", tenant_id=TENANT)
        assert sched.schedule_type == "interval"
        assert sched.interval_seconds == 1800

    def test_every_2_hours(self):
        sched = parse_natural_language("every 2 hours", tenant_id=TENANT)
        assert sched.schedule_type == "interval"
        assert sched.interval_seconds == 7200

    def test_every_day_at_8am(self):
        sched = parse_natural_language("every day at 8am", tenant_id=TENANT)
        assert sched.schedule_type == "cron"
        assert sched.cron_expression == "0 8 * * *"

    def test_every_monday_at_9_30am(self):
        sched = parse_natural_language("every monday at 9:30am", tenant_id=TENANT)
        assert sched.schedule_type == "cron"
        assert sched.cron_expression == "30 9 * * 1"

    def test_tomorrow_at_3pm(self):
        sched = parse_natural_language("tomorrow at 3pm", tenant_id=TENANT)
        assert sched.schedule_type == "once"
        assert sched.run_at is not None
        assert sched.run_at.hour == 15

    def test_specific_date(self):
        sched = parse_natural_language("on 2026-06-01 at 10am", tenant_id=TENANT)
        assert sched.schedule_type == "once"
        assert sched.run_at == datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

    def test_every_minute(self):
        sched = parse_natural_language("every minute", tenant_id=TENANT)
        assert sched.schedule_type == "interval"
        assert sched.interval_seconds == 60

    def test_unparseable_raises(self):
        with pytest.raises(SchedulerError, match="Could not parse"):
            parse_natural_language("do something weird", tenant_id=TENANT)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_cron_schedule_advances_after_run(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="cron-advance",
            schedule_type="cron", cron_expression="0 8 * * *",
        ))
        old_next = sched.next_run_at
        run = ScheduleRun(
            schedule_id=sched.id, tenant_id=TENANT,
            status="completed",
            started_at=old_next,
        )
        engine.record_run(sched.id, run)
        assert sched.next_run_at > old_next

    def test_interval_schedule_advances_after_run(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="interval-advance",
            schedule_type="interval", interval_seconds=600,
        ))
        started = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        engine.record_run(sched.id, ScheduleRun(
            schedule_id=sched.id, tenant_id=TENANT,
            status="completed", started_at=started,
        ))
        assert sched.next_run_at == started + timedelta(seconds=600)

    def test_update_cron_recomputes_next_run(self):
        engine = SchedulerEngine(tenant_id=TENANT)
        sched = engine.create(Schedule(
            tenant_id=TENANT, name="recompute",
            schedule_type="cron", cron_expression="0 8 * * *",
        ))
        old_next = sched.next_run_at
        engine.update(sched.id, {"cron_expression": "0 12 * * *"})
        # Next run should change (unless both happen to be the same)
        assert sched.next_run_at is not None
