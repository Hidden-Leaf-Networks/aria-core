"""MCP Server — exposes aria-core capabilities as MCP tools, resources, and prompts.

Tools exposed:
- create_plan: Create an execution plan
- list_plans: List plans for a tenant
- get_plan: Get plan details
- execute_plan: Execute a plan
- create_agent: Register a new agent from archetype
- list_agents: List registered agents
- calculate_risk: Score risk for an action
- search_events: Query the event store
- get_usage: Get tenant usage report

Resources exposed:
- tenant://config — Current tenant configuration
- tenant://archetypes — Available agent archetypes
- tenant://events/latest — Latest events

Prompts exposed:
- plan_builder — Generate a plan from natural language
- risk_assessment — Assess risk for proposed actions
- agent_designer — Design an agent configuration

Usage:
    from aria_core.mcp import create_server

    server = create_server(provider, tenant_id=tid)
    server.run(transport="streamable-http")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


def create_server(
    provider: Any = None,
    tenant_id: UUID | None = None,
    server_name: str = "aria-core",
) -> Any:
    """Create an MCP server exposing aria-core capabilities.

    Args:
        provider: PersistenceProvider instance (InMemory or Postgres)
        tenant_id: Default tenant for operations
        server_name: Server name for MCP registration
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError("MCP SDK not installed. Install with: pip install 'aria-core[mcp]'")

    mcp = FastMCP(server_name, json_response=True)

    # Default tenant
    _tenant_id = tenant_id or UUID("00000000-0000-0000-0000-000000000000")

    # -------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------

    @mcp.tool()
    async def create_plan(
        name: str,
        description: str = "",
        actions: str = "[]",
    ) -> str:
        """Create an execution plan with actions and dependencies.

        Args:
            name: Plan name
            description: What the plan accomplishes
            actions: JSON array of actions. Each action: {"name": str, "skill_name": str, "dependencies": [int]}
        """
        from aria_core.planning.models import Plan, PlanAction, PlanState

        action_list = json.loads(actions)
        plan_id = uuid4()
        now = datetime.now(timezone.utc)

        plan_actions = [
            PlanAction(
                plan_id=plan_id,
                index=i,
                name=a.get("name", f"Action {i}"),
                skill_name=a.get("skill_name", ""),
                dependencies=a.get("dependencies", []),
            )
            for i, a in enumerate(action_list)
        ]

        plan = Plan(
            id=plan_id,
            name=name,
            description=description,
            state=PlanState.DRAFT,
            actions=plan_actions,
            created_at=now,
            updated_at=now,
            created_by="mcp-client",
        )

        if provider:
            await provider.save_plan(_tenant_id, plan)

        return json.dumps({
            "plan_id": str(plan_id),
            "name": name,
            "actions": len(plan_actions),
            "state": "draft",
        })

    @mcp.tool()
    async def list_plans(state: str = "", limit: int = 20) -> str:
        """List execution plans for the current tenant.

        Args:
            state: Filter by state (draft, executing, completed, failed). Empty for all.
            limit: Max results (default 20)
        """
        if not provider:
            return json.dumps([])

        plans = await provider.list_plans(
            _tenant_id,
            state=state or None,
            limit=limit,
        )
        return json.dumps([
            {
                "id": str(p.id),
                "name": p.name,
                "state": p.state.value if hasattr(p.state, "value") else str(p.state),
                "actions": len(p.actions),
                "created_at": p.created_at.isoformat(),
            }
            for p in plans
        ])

    @mcp.tool()
    async def get_plan(plan_id: str) -> str:
        """Get detailed plan information including actions and their states.

        Args:
            plan_id: UUID of the plan
        """
        if not provider:
            return json.dumps({"error": "No provider configured"})

        plan = await provider.get_plan(_tenant_id, UUID(plan_id))
        if not plan:
            return json.dumps({"error": "Plan not found"})

        return json.dumps(plan.model_dump(mode="json"))

    @mcp.tool()
    async def create_agent(
        name: str,
        description: str = "",
        model: str = "gpt-4",
        system_prompt: str = "",
        archetype_slug: str = "",
    ) -> str:
        """Register a new agent. Optionally use an archetype as a template.

        Args:
            name: Agent name
            description: What the agent does
            model: LLM model to use (e.g. gpt-4, claude-sonnet-4-20250514)
            system_prompt: System prompt for the agent
            archetype_slug: If provided, use this archetype as a base (e.g. research-analyst, code-assistant)
        """
        agent_config = {
            "name": name,
            "description": description,
            "model": model,
            "system_prompt": system_prompt,
        }

        if archetype_slug:
            from aria_core.archetypes.registry import ArchetypeRegistry
            registry = ArchetypeRegistry()
            await registry.seed_defaults(_tenant_id)
            archetype = await registry.get_by_slug(_tenant_id, archetype_slug)
            if archetype:
                base = await registry.create_from_archetype(
                    _tenant_id, archetype.id,
                    overrides={k: v for k, v in agent_config.items() if v},
                )
                return json.dumps({"agent": base, "from_archetype": archetype_slug})

        return json.dumps({"agent": agent_config, "from_archetype": None})

    @mcp.tool()
    async def list_archetypes(category: str = "") -> str:
        """List available agent archetypes (pre-built templates).

        Args:
            category: Filter by category (research, engineering, content, data, support, security, operations)
        """
        from aria_core.archetypes.registry import ArchetypeRegistry
        registry = ArchetypeRegistry()
        await registry.seed_defaults(_tenant_id)
        archetypes = await registry.list(_tenant_id, category=category or None)
        return json.dumps([
            {
                "slug": a.slug,
                "name": a.name,
                "category": a.category.value,
                "model": a.model,
                "description": a.description,
                "icon": a.icon,
            }
            for a in archetypes
        ])

    @mcp.tool()
    async def calculate_risk(
        skill_name: str,
        skill_category: str = "exec",
        impact_scope: str = "user",
    ) -> str:
        """Calculate risk score for a proposed action.

        Args:
            skill_name: Name of the skill/tool to assess
            skill_category: Category — read, write, exec, external
            impact_scope: Scope — local, user, system, external
        """
        from aria_core.permissions.risk_engine import RiskEngine
        from aria_core.permissions.models import RiskScoreInput, SkillCategory, ImpactScope

        engine = RiskEngine()
        score = engine.calculate(RiskScoreInput(
            skill_name=skill_name,
            skill_category=SkillCategory(skill_category),
            impact_scope=ImpactScope(impact_scope),
        ))
        return json.dumps({
            "score": score.score,
            "level": score.level,
            "requires_approval": score.requires_approval,
            "factors": [f.model_dump() for f in score.factors],
        })

    @mcp.tool()
    async def search_events(
        event_type: str = "",
        limit: int = 50,
    ) -> str:
        """Search the event audit trail.

        Args:
            event_type: Filter by event type (e.g. plan.created, agent.start). Empty for all.
            limit: Max results
        """
        if not provider:
            return json.dumps([])

        events = await provider.list_events(
            _tenant_id,
            event_type=event_type or None,
            limit=limit,
        )
        return json.dumps([
            {
                "event_type": e["event_type"],
                "payload": e["payload"],
                "timestamp": e["timestamp"].isoformat() if hasattr(e["timestamp"], "isoformat") else str(e["timestamp"]),
            }
            for e in events
        ])

    @mcp.tool()
    async def get_pricing(
        api_calls: int = 0,
        events: int = 0,
        agent_runs: int = 0,
        tenants: int = 1,
    ) -> str:
        """Calculate pricing based on projected usage.

        Args:
            api_calls: Expected monthly API calls
            events: Expected monthly events
            agent_runs: Expected monthly agent runs
            tenants: Number of tenants needed
        """
        from aria_core.billing.pricing import PricingCalculator
        result = PricingCalculator().recommend_tier(
            api_calls=api_calls,
            events=events,
            agent_runs=agent_runs,
            tenants=tenants,
        )
        return json.dumps(result)

    # -------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------

    @mcp.resource("aria://archetypes")
    async def archetypes_resource() -> str:
        """Available agent archetypes for this tenant."""
        from aria_core.archetypes.registry import ArchetypeRegistry
        registry = ArchetypeRegistry()
        await registry.seed_defaults(_tenant_id)
        archetypes = await registry.list(_tenant_id)
        return json.dumps([
            {"slug": a.slug, "name": a.name, "category": a.category.value, "description": a.description}
            for a in archetypes
        ])

    @mcp.resource("aria://pricing/tiers")
    async def pricing_tiers_resource() -> str:
        """Aria Core pricing tiers."""
        from aria_core.billing.pricing import PricingCalculator
        return json.dumps(PricingCalculator().get_tiers())

    @mcp.resource("aria://version")
    async def version_resource() -> str:
        """Aria Core version info."""
        import aria_core
        return json.dumps({
            "version": aria_core.__version__,
            "framework": "aria-core",
            "mcp_server": server_name,
        })

    # -------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------

    @mcp.prompt()
    def plan_builder(task_description: str) -> str:
        """Generate an execution plan from a natural language task description.

        Args:
            task_description: What you want to accomplish
        """
        return f"""You are an AI plan architect using the Aria Core agent framework.

Given this task: "{task_description}"

Design an execution plan with the following structure:
1. Break the task into discrete, ordered actions
2. Identify dependencies between actions (which must complete before others can start)
3. Assign a skill_name to each action (e.g. web_search, llm_generate, docker_build, etc.)
4. Assess the risk level of each action (safe, low, medium, high, critical)

Format your response as a JSON object:
{{
  "name": "Plan name",
  "description": "What this plan accomplishes",
  "actions": [
    {{"name": "Action name", "skill_name": "skill", "description": "what it does", "dependencies": [], "risk_level": "low"}}
  ]
}}"""

    @mcp.prompt()
    def risk_assessment(action_description: str) -> str:
        """Assess risk for a proposed agent action.

        Args:
            action_description: Description of what the agent wants to do
        """
        return f"""You are a risk assessor for an AI agent framework with these risk levels:
- safe (0-20): Read-only, local scope
- low (20-40): Write operations, user scope
- medium (40-60): Execution, system scope — requires monitoring
- high (60-80): External systems, sensitive data — requires approval
- critical (80-100): Irreversible actions, production impact — blocked without approval

Assess this proposed action: "{action_description}"

Provide:
1. Risk score (0-100)
2. Risk level
3. Key risk factors
4. Recommended approval gate (if any)
5. Mitigation suggestions"""

    @mcp.prompt()
    def agent_designer(use_case: str) -> str:
        """Design an agent configuration for a specific use case.

        Args:
            use_case: What the agent should be able to do
        """
        return f"""You are an agent architect for the Aria Core framework.

Design an agent for this use case: "{use_case}"

Available archetypes to build on: research-analyst, code-assistant, content-writer, data-engineer, support-agent, security-auditor, ops-coordinator

Provide a complete agent configuration:
{{
  "name": "Agent name",
  "slug": "kebab-case-slug",
  "description": "What it does",
  "archetype_base": "slug or null if custom",
  "model": "model-name",
  "system_prompt": "Full system prompt",
  "temperature": 0.0-2.0,
  "max_steps": 1-50,
  "allowed_skills": ["skill1", "skill2"],
  "risk_policy": {{
    "require_approval": true/false,
    "risk_threshold": 0-100
  }}
}}"""

    return mcp
