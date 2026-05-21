"""Scheduler engine — cron parsing, schedule management, and tick loop.

Implements a fully in-memory scheduler suitable for single-process deployments
and as the reference implementation for persistent backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID, uuid4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchedulerError(Exception):
    """Base error for scheduler operations."""


class CronParseError(SchedulerError):
    """Raised when a cron expression is invalid."""


class ScheduleNotFoundError(SchedulerError):
    """Raised when a schedule ID is not found."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

ScheduleType = Literal["cron", "interval", "once"]
RunStatus = Literal["pending", "running", "completed", "failed", "timeout"]


@dataclass
class Schedule:
    """A scheduled task definition."""

    tenant_id: UUID
    name: str
    description: str = ""
    agent_id: str | None = None
    message: str = ""
    model: str | None = None
    schedule_type: ScheduleType = "cron"
    cron_expression: str | None = None
    interval_seconds: int | None = None
    run_at: datetime | None = None
    enabled: bool = True
    timeout_seconds: int = 900
    max_retries: int = 0
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    run_count: int = 0
    failure_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)


@dataclass
class ScheduleRun:
    """Record of a single schedule execution."""

    schedule_id: UUID
    tenant_id: UUID
    status: RunStatus = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    output: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)


# ---------------------------------------------------------------------------
# Cron parser
# ---------------------------------------------------------------------------

# Named field bounds: (min, max)
_CRON_FIELDS: list[tuple[str, int, int]] = [
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day_of_month", 1, 31),
    ("month", 1, 12),
    ("day_of_week", 0, 6),  # 0=Sunday
]


class CronParser:
    """Parse and evaluate standard 5-field cron expressions.

    Supports: wildcard (*), ranges (1-5), steps (*/15, 1-5/2), lists (1,3,5).
    """

    @staticmethod
    def _parse_field(token: str, lo: int, hi: int) -> set[int]:
        """Expand a single cron field token into a set of matching integers."""
        values: set[int] = set()
        for part in token.split(","):
            part = part.strip()
            if not part:
                raise CronParseError(f"Empty element in cron field: {token!r}")

            step: int | None = None
            if "/" in part:
                base, step_str = part.split("/", 1)
                if not step_str.isdigit() or int(step_str) == 0:
                    raise CronParseError(f"Invalid step value: {step_str!r}")
                step = int(step_str)
                part = base

            if part == "*":
                start, end = lo, hi
            elif "-" in part:
                a, b = part.split("-", 1)
                if not (a.isdigit() and b.isdigit()):
                    raise CronParseError(f"Invalid range: {part!r}")
                start, end = int(a), int(b)
                if start < lo or end > hi or start > end:
                    raise CronParseError(
                        f"Range {start}-{end} out of bounds [{lo}-{hi}]"
                    )
            elif part.isdigit():
                val = int(part)
                if val < lo or val > hi:
                    raise CronParseError(f"Value {val} out of bounds [{lo}-{hi}]")
                if step is None:
                    values.add(val)
                    continue
                start, end = val, hi
            else:
                raise CronParseError(f"Invalid cron token: {part!r}")

            if step is None:
                values.update(range(start, end + 1))
            else:
                values.update(range(start, end + 1, step))

        return values

    @classmethod
    def parse(cls, expression: str) -> dict[str, set[int]]:
        """Validate and parse a cron expression into component sets.

        Returns a dict with keys: minute, hour, day_of_month, month, day_of_week.
        Each value is a set of integers that the field matches.
        """
        tokens = expression.strip().split()
        if len(tokens) != 5:
            raise CronParseError(
                f"Expected 5 fields, got {len(tokens)}: {expression!r}"
            )

        result: dict[str, set[int]] = {}
        for (name, lo, hi), token in zip(_CRON_FIELDS, tokens):
            result[name] = cls._parse_field(token, lo, hi)
        return result

    @classmethod
    def matches(cls, expression: str, dt: datetime) -> bool:
        """Check whether *dt* matches the cron expression."""
        fields = cls.parse(expression)
        return (
            dt.minute in fields["minute"]
            and dt.hour in fields["hour"]
            and dt.day in fields["day_of_month"]
            and dt.month in fields["month"]
            and dt.weekday() in _weekday_py(fields["day_of_week"])
        )

    @classmethod
    def next_run(
        cls, expression: str, after: datetime | None = None
    ) -> datetime:
        """Calculate the next datetime that matches *expression* after *after*.

        Searches minute-by-minute up to ~2 years into the future.
        """
        fields = cls.parse(expression)
        if after is None:
            after = datetime.now(timezone.utc)
        # Start from the next whole minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        py_dow = _weekday_py(fields["day_of_week"])

        # Safety cap: 2 years of minutes
        limit = 366 * 24 * 60 * 2
        for _ in range(limit):
            if (
                candidate.month in fields["month"]
                and candidate.day in fields["day_of_month"]
                and candidate.weekday() in py_dow
                and candidate.hour in fields["hour"]
                and candidate.minute in fields["minute"]
            ):
                return candidate
            candidate += timedelta(minutes=1)

        raise SchedulerError(
            f"Could not find next run for {expression!r} within 2 years of {after}"
        )


def _weekday_py(cron_dow: set[int]) -> set[int]:
    """Convert cron day-of-week (0=Sun) to Python weekday (0=Mon)."""
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return {mapping[d] for d in cron_dow}


# ---------------------------------------------------------------------------
# Natural language helpers
# ---------------------------------------------------------------------------

_INTERVAL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"every\s+(\d+)\s+second", re.I), 1),
    (re.compile(r"every\s+(\d+)\s+minute", re.I), 60),
    (re.compile(r"every\s+(\d+)\s+hour", re.I), 3600),
    (re.compile(r"every\s+minute", re.I), 60),
    (re.compile(r"every\s+hour", re.I), 3600),
]

_TIME_RE = re.compile(
    r"(?:^|at\s+)(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.I,
)

_DAILY_RE = re.compile(r"every\s+day", re.I)
_WEEKDAY_MAP = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 0,
    "mon": 1, "tue": 2, "wed": 3, "thu": 4,
    "fri": 5, "sat": 6, "sun": 0,
}
_WEEKDAY_RE = re.compile(
    r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|wed|thu|fri|sat|sun)",
    re.I,
)
_TOMORROW_RE = re.compile(r"tomorrow", re.I)
_ONCE_DATE_RE = re.compile(
    r"on\s+(\d{4}-\d{2}-\d{2})",
    re.I,
)


def _extract_time(text: str) -> tuple[int, int]:
    """Extract hour and minute from text. Defaults to 0:00."""
    m = _TIME_RE.search(text)
    if not m:
        return 0, 0
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return hour, minute


def parse_natural_language(text: str, tenant_id: UUID | None = None) -> Schedule:
    """Convert natural-language schedule text into a Schedule object.

    Supported patterns:
    - "every 30 minutes" / "every 2 hours" -> interval
    - "every day at 8am" -> cron
    - "every monday at 9:30am" -> cron
    - "tomorrow at 3pm" / "on 2026-06-01 at 10am" -> once
    """
    tid = tenant_id or uuid4()

    # --- Interval patterns ---
    for pattern, multiplier in _INTERVAL_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = m.groups()
            qty = int(groups[0]) if groups else 1
            return Schedule(
                tenant_id=tid,
                name=text.strip(),
                message=text.strip(),
                schedule_type="interval",
                interval_seconds=qty * multiplier,
            )

    # --- Weekday cron ---
    wm = _WEEKDAY_RE.search(text)
    if wm:
        dow = _WEEKDAY_MAP[wm.group(1).lower()]
        hour, minute = _extract_time(text)
        return Schedule(
            tenant_id=tid,
            name=text.strip(),
            message=text.strip(),
            schedule_type="cron",
            cron_expression=f"{minute} {hour} * * {dow}",
        )

    # --- Daily cron ---
    if _DAILY_RE.search(text):
        hour, minute = _extract_time(text)
        return Schedule(
            tenant_id=tid,
            name=text.strip(),
            message=text.strip(),
            schedule_type="cron",
            cron_expression=f"{minute} {hour} * * *",
        )

    # --- Tomorrow (once) ---
    if _TOMORROW_RE.search(text):
        hour, minute = _extract_time(text)
        now = datetime.now(timezone.utc)
        run_at = (now + timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        return Schedule(
            tenant_id=tid,
            name=text.strip(),
            message=text.strip(),
            schedule_type="once",
            run_at=run_at,
        )

    # --- Specific date (once) ---
    dm = _ONCE_DATE_RE.search(text)
    if dm:
        hour, minute = _extract_time(text)
        date = datetime.strptime(dm.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        run_at = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return Schedule(
            tenant_id=tid,
            name=text.strip(),
            message=text.strip(),
            schedule_type="once",
            run_at=run_at,
        )

    raise SchedulerError(f"Could not parse schedule from: {text!r}")


# ---------------------------------------------------------------------------
# Scheduler engine
# ---------------------------------------------------------------------------


class SchedulerEngine:
    """In-memory schedule manager with tick-based due detection."""

    def __init__(self, tenant_id: UUID) -> None:
        self.tenant_id = tenant_id
        self._schedules: dict[UUID, Schedule] = {}
        self._runs: dict[UUID, list[ScheduleRun]] = {}  # schedule_id -> runs

    # -- CRUD ---------------------------------------------------------------

    def create(self, schedule: Schedule) -> Schedule:
        """Register a new schedule, computing its initial next_run_at."""
        schedule.tenant_id = self.tenant_id
        if schedule.id in self._schedules:
            raise SchedulerError(f"Schedule {schedule.id} already exists")
        schedule.next_run_at = self._compute_next_run(schedule)
        self._schedules[schedule.id] = schedule
        self._runs[schedule.id] = []
        return schedule

    def get(self, schedule_id: UUID) -> Schedule | None:
        """Return a schedule by ID, or None."""
        return self._schedules.get(schedule_id)

    def update(self, schedule_id: UUID, updates: dict[str, Any]) -> Schedule:
        """Apply partial updates to a schedule."""
        sched = self._schedules.get(schedule_id)
        if sched is None:
            raise ScheduleNotFoundError(f"Schedule {schedule_id} not found")
        for key, value in updates.items():
            if not hasattr(sched, key):
                raise SchedulerError(f"Unknown schedule field: {key!r}")
            setattr(sched, key, value)
        # Recompute next_run if schedule definition changed
        if any(k in updates for k in ("cron_expression", "interval_seconds", "run_at", "schedule_type")):
            sched.next_run_at = self._compute_next_run(sched)
        return sched

    def delete(self, schedule_id: UUID) -> bool:
        """Delete a schedule. Returns True if it existed."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            self._runs.pop(schedule_id, None)
            return True
        return False

    def list(self, enabled_only: bool = False) -> list[Schedule]:
        """List schedules, optionally filtering to enabled only."""
        schedules = list(self._schedules.values())
        if enabled_only:
            schedules = [s for s in schedules if s.enabled]
        return schedules

    def enable(self, schedule_id: UUID) -> Schedule:
        """Enable a schedule."""
        return self.update(schedule_id, {"enabled": True})

    def disable(self, schedule_id: UUID) -> Schedule:
        """Disable a schedule."""
        return self.update(schedule_id, {"enabled": False})

    # -- Execution tracking -------------------------------------------------

    def get_due_schedules(self, now: datetime | None = None) -> list[Schedule]:
        """Return enabled schedules whose next_run_at <= now."""
        if now is None:
            now = datetime.now(timezone.utc)
        return [
            s
            for s in self._schedules.values()
            if s.enabled and s.next_run_at is not None and s.next_run_at <= now
        ]

    def record_run(self, schedule_id: UUID, run: ScheduleRun) -> None:
        """Record an execution result and update schedule counters."""
        sched = self._schedules.get(schedule_id)
        if sched is None:
            raise ScheduleNotFoundError(f"Schedule {schedule_id} not found")
        run.schedule_id = schedule_id
        run.tenant_id = self.tenant_id
        self._runs.setdefault(schedule_id, []).append(run)

        sched.run_count += 1
        if run.status == "failed" or run.status == "timeout":
            sched.failure_count += 1
        sched.last_run_at = run.started_at or datetime.now(timezone.utc)

        # Advance next_run_at (disable one-shot schedules after execution)
        if sched.schedule_type == "once":
            sched.enabled = False
            sched.next_run_at = None
        else:
            sched.next_run_at = self._compute_next_run(sched, after=sched.last_run_at)

    def get_runs(self, schedule_id: UUID, limit: int = 20) -> list[ScheduleRun]:
        """Return recent runs for a schedule, newest first."""
        runs = self._runs.get(schedule_id, [])
        return list(reversed(runs[-limit:]))

    def tick(self, now: datetime | None = None) -> list[Schedule]:
        """Called periodically. Returns schedules that are due to run NOW.

        This is the primary entry point for a scheduling loop:
            while True:
                for sched in engine.tick(datetime.now(utc)):
                    dispatch(sched)
                sleep(60)
        """
        return self.get_due_schedules(now)

    # -- Internal -----------------------------------------------------------

    @staticmethod
    def _compute_next_run(
        schedule: Schedule, after: datetime | None = None
    ) -> datetime | None:
        """Determine the next run time for a schedule."""
        if after is None:
            after = datetime.now(timezone.utc)

        if schedule.schedule_type == "cron":
            if not schedule.cron_expression:
                raise SchedulerError("Cron schedule requires cron_expression")
            return CronParser.next_run(schedule.cron_expression, after=after)

        if schedule.schedule_type == "interval":
            if not schedule.interval_seconds or schedule.interval_seconds <= 0:
                raise SchedulerError("Interval schedule requires positive interval_seconds")
            return after + timedelta(seconds=schedule.interval_seconds)

        if schedule.schedule_type == "once":
            return schedule.run_at

        raise SchedulerError(f"Unknown schedule_type: {schedule.schedule_type!r}")
