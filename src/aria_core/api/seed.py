"""Seed demo data on first deploy.

Seeds archetypes, sample plans, sample agents, and starter events
into the default tenant so the dashboard has data immediately.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from aria_core.archetypes.registry import ArchetypeRegistry
from aria_core.persistence.event_store import EventStore
from aria_core.planning.models import Plan, PlanAction, PlanState
from aria_core.tenant.models import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)

# Deterministic UUIDs for demo agents so they are idempotent across restarts
_DEMO_AGENT_IDS = {
    "research-analyst": UUID("aaaaaaaa-0000-0000-0000-000000000001"),
    "code-assistant": UUID("aaaaaaaa-0000-0000-0000-000000000002"),
}

# Sample plan templates
_SAMPLE_PLANS: list[dict[str, Any]] = [
    {
        "name": "CI/CD Pipeline",
        "description": "Automated build, test, and deploy pipeline for the main branch.",
        "actions": [
            {"name": "Checkout Code", "skill_name": "code_generate", "index": 0},
            {"name": "Run Tests", "skill_name": "test_generate", "index": 1, "dependencies": [0]},
            {"name": "Build Artifacts", "skill_name": "code_generate", "index": 2, "dependencies": [1]},
            {"name": "Deploy to Staging", "skill_name": "code_review", "index": 3, "dependencies": [2]},
        ],
    },
    {
        "name": "Research Pipeline",
        "description": "Multi-source research synthesis with structured report output.",
        "actions": [
            {"name": "Gather Sources", "skill_name": "web_search", "index": 0},
            {"name": "Analyze Findings", "skill_name": "synthesize", "index": 1, "dependencies": [0]},
            {"name": "Generate Report", "skill_name": "generate_report", "index": 2, "dependencies": [1]},
        ],
    },
    {
        "name": "Content Pipeline",
        "description": "End-to-end content creation from outline to published post.",
        "actions": [
            {"name": "Create Outline", "skill_name": "llm_generate", "index": 0},
            {"name": "Draft Content", "skill_name": "llm_generate", "index": 1, "dependencies": [0]},
            {"name": "SEO Optimize", "skill_name": "seo_optimize", "index": 2, "dependencies": [1]},
            {"name": "Publish", "skill_name": "publish", "index": 3, "dependencies": [2]},
        ],
    },
]


async def seed_demo_data(provider: Any) -> dict[str, Any]:
    """Seed demo data for the default tenant.

    Idempotent: checks for existing data before inserting.
    Returns a summary dict of what was seeded.
    """
    tenant_id = DEFAULT_TENANT_ID

    # Ensure default tenant exists in the database
    from aria_core.tenant.models import DEFAULT_TENANT
    existing = await provider.get_tenant(tenant_id)
    if not existing:
        await provider.save_tenant(DEFAULT_TENANT)
        logger.info("Created default tenant in database")

    summary: dict[str, Any] = {
        "archetypes": 0,
        "plans": 0,
        "agents": 0,
        "events": 0,
    }

    # 1. Seed all 14 archetypes
    registry = ArchetypeRegistry()
    count = await registry.seed_defaults(tenant_id)
    summary["archetypes"] = count
    if count > 0:
        logger.info("Seeded %d archetypes for default tenant", count)

    # 2. Seed sample plans (skip if plans already exist)
    existing_plans = await provider.list_plans(tenant_id, limit=1)
    if not existing_plans:
        for plan_tmpl in _SAMPLE_PLANS:
            plan_id = uuid4()
            actions = [
                PlanAction(
                    plan_id=plan_id,
                    name=a["name"],
                    skill_name=a.get("skill_name"),
                    index=a.get("index", i),
                    dependencies=a.get("dependencies", []),
                )
                for i, a in enumerate(plan_tmpl["actions"])
            ]
            plan = Plan(
                id=plan_id,
                name=plan_tmpl["name"],
                description=plan_tmpl["description"],
                state=PlanState.DRAFT,
                actions=actions,
                created_by="seed",
            )
            await provider.save_plan(tenant_id, plan)
            summary["plans"] += 1
        logger.info("Seeded %d sample plans", summary["plans"])

    # 3. Seed sample agents from archetypes (skip if events already indicate agents)
    existing_agent_events = await provider.count_events(
        tenant_id, event_type="agent.registered"
    )
    if existing_agent_events == 0:
        for slug, agent_id in _DEMO_AGENT_IDS.items():
            archetype = await registry.get_by_slug(tenant_id, slug)
            if archetype is None:
                continue
            agent_config = await registry.create_from_archetype(
                tenant_id, archetype.id
            )
            agent_config["agent_id"] = str(agent_id)
            agent_config["archetype_id"] = str(archetype.id)

            # Persist the agent registration as an event
            await provider.save_event(
                tenant_id,
                "agent.registered",
                agent_config,
                agent_id=agent_id,
            )
            summary["agents"] += 1
        logger.info("Seeded %d sample agents", summary["agents"])

    # 4. Emit seed events so the dashboard has activity data
    existing_events = await provider.count_events(tenant_id)
    # Only emit if the store is mostly empty (just agent registrations or nothing)
    if existing_events <= summary["agents"]:
        event_store = EventStore(provider, tenant_id)

        # System boot event
        await event_store.emit("system.seed", {
            "message": "Demo data seeded on first deploy",
            "archetypes": summary["archetypes"],
            "plans": summary["plans"],
            "agents": summary["agents"],
        })
        summary["events"] += 1

        # Emit plan.created events for each sample plan
        plans = await provider.list_plans(tenant_id, limit=10)
        for plan in plans:
            await event_store.emit("plan.created", {
                "plan_id": str(plan.id),
                "name": plan.name,
                "action_count": len(plan.actions),
            })
            summary["events"] += 1

        # Emit agent activity events for dashboard sparklines
        for slug, agent_id in _DEMO_AGENT_IDS.items():
            await event_store.emit(
                "agent.heartbeat",
                {"status": "idle", "slug": slug},
                agent_id=agent_id,
            )
            summary["events"] += 1

        logger.info("Emitted %d seed events", summary["events"])

    return summary
