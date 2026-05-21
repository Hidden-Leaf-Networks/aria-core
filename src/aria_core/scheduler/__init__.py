"""Agent scheduling — cron jobs, recurring tasks, natural language schedules.

Provides:
- Schedule / ScheduleRun: data models for scheduled tasks and execution records
- CronParser: standard 5-field cron expression parsing and next-run calculation
- SchedulerEngine: CRUD, due-detection, tick loop, and run recording
- parse_natural_language: convert human text to Schedule objects

Usage:
    from aria_core.scheduler import SchedulerEngine, Schedule

    engine = SchedulerEngine(tenant_id=my_tenant)
    sched = engine.create(Schedule(
        tenant_id=my_tenant,
        name="morning-brief",
        agent_id="research-agent",
        message="Summarize overnight news",
        schedule_type="cron",
        cron_expression="0 8 * * *",
    ))
"""

from aria_core.scheduler.engine import (
    CronParser,
    CronParseError,
    Schedule,
    ScheduleRun,
    SchedulerEngine,
    SchedulerError,
    ScheduleNotFoundError,
    parse_natural_language,
)

__all__ = [
    "CronParser",
    "CronParseError",
    "Schedule",
    "ScheduleRun",
    "SchedulerEngine",
    "SchedulerError",
    "ScheduleNotFoundError",
    "parse_natural_language",
]
