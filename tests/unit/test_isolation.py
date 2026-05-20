"""Data isolation and stress tests for the multi-tenant persistence layer.

Tests:
- TenantGuard enforcement
- Cross-tenant context mismatch detection
- Load test: 10 tenants × concurrent operations
- Migration-safe model roundtrip
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from aria_core.permissions.models import Approval, ApprovalGate, ApprovalState, RiskPolicy
from aria_core.persistence.memory import InMemoryProvider
from aria_core.persistence.event_store import EventStore
from aria_core.planning.models import Plan, PlanAction, PlanState
from aria_core.runtime.models import AgentContext, ChatMessage, MessageRole
from aria_core.tenant.config_resolver import ConfigResolver
from aria_core.tenant.guard import TenantGuard, TenantViolationError
from aria_core.tenant.models import Tenant, TenantConfig


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def guard(provider: InMemoryProvider) -> TenantGuard:
    return TenantGuard(provider)


# ---------------------------------------------------------------------------
# TenantGuard enforcement
# ---------------------------------------------------------------------------


class TestTenantGuard:
    async def test_guard_validates_tenant_id(
        self, provider: InMemoryProvider, guard: TenantGuard
    ) -> None:
        """Guard rejects None tenant_id."""
        with pytest.raises(TenantViolationError):
            await guard.save_plan(None, Plan(id=uuid4(), name="test"))  # type: ignore[arg-type]

    async def test_guard_context_tenant_mismatch(
        self, provider: InMemoryProvider, guard: TenantGuard
    ) -> None:
        """Guard catches context.tenant_id != claimed tenant_id."""
        tenant = Tenant(slug="guard-co", name="Guard Co")
        await guard.save_tenant(tenant)

        other_tenant_id = uuid4()
        ctx = AgentContext(tenant_id=other_tenant_id)

        with pytest.raises(TenantViolationError):
            await guard.save_context(tenant.id, ctx)

    async def test_guard_allows_matching_tenant(
        self, provider: InMemoryProvider, guard: TenantGuard
    ) -> None:
        """Guard passes when context.tenant_id matches."""
        tenant = Tenant(slug="match-co", name="Match Co")
        await guard.save_tenant(tenant)

        ctx = AgentContext(tenant_id=tenant.id)
        saved = await guard.save_context(tenant.id, ctx)
        assert saved.tenant_id == tenant.id

    async def test_guard_wraps_all_crud(
        self, provider: InMemoryProvider, guard: TenantGuard
    ) -> None:
        """Guard works as a drop-in replacement for provider."""
        tenant = Tenant(slug="full-co", name="Full Co")
        await guard.save_tenant(tenant)
        tid = tenant.id

        # Plan CRUD
        plan = Plan(id=uuid4(), name="Guarded Plan")
        await guard.save_plan(tid, plan)
        assert await guard.get_plan(tid, plan.id) is not None
        plans = await guard.list_plans(tid)
        assert len(plans) == 1

        # Approval
        from datetime import datetime, timedelta, timezone

        approval = Approval(
            plan_id=plan.id,
            gate_id=uuid4(),
            gate_name="test",
            risk_score=50,
            state=ApprovalState.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        await guard.save_approval(tid, approval)
        assert await guard.get_approval(tid, approval.id) is not None

        # Events
        await guard.save_event(tid, "test", {"data": 1})
        events = await guard.list_events(tid)
        assert len(events) == 1
        assert await guard.count_events(tid) == 1

        # Risk policy
        policy = RiskPolicy(name="test-policy")
        await guard.save_risk_policy(tid, policy)
        assert await guard.get_risk_policy(tid, policy.id) is not None

        # Approval gate
        gate = ApprovalGate(name="test-gate", risk_threshold=50)
        await guard.save_approval_gate(tid, gate)
        gates = await guard.list_approval_gates(tid)
        assert len(gates) == 1


# ---------------------------------------------------------------------------
# Multi-tenant load test
# ---------------------------------------------------------------------------


class TestMultiTenantLoad:
    async def test_10_tenants_concurrent_operations(
        self, provider: InMemoryProvider
    ) -> None:
        """10 tenants each performing 10 operations concurrently.
        Verify complete isolation after all operations complete.
        """
        num_tenants = 10
        ops_per_tenant = 10

        # Create tenants
        tenants = [
            Tenant(slug=f"load-{i:02d}", name=f"Load Tenant {i}")
            for i in range(num_tenants)
        ]
        for t in tenants:
            await provider.save_tenant(t)

        # Run concurrent operations for each tenant
        async def tenant_workload(tenant: Tenant, index: int) -> None:
            for j in range(ops_per_tenant):
                plan = Plan(
                    id=uuid4(),
                    name=f"Plan-{index}-{j}",
                    state=PlanState.DRAFT,
                )
                await provider.save_plan(tenant.id, plan)
                await provider.save_event(
                    tenant.id,
                    f"plan.created",
                    {"plan_name": f"Plan-{index}-{j}", "tenant_index": index},
                )
                ctx = AgentContext(
                    tenant_id=tenant.id,
                    messages=[ChatMessage(role=MessageRole.USER, content=f"msg-{j}")],
                )
                await provider.save_context(tenant.id, ctx)

        # Run all tenant workloads concurrently
        tasks = [
            tenant_workload(t, i) for i, t in enumerate(tenants)
        ]
        await asyncio.gather(*tasks)

        # Verify isolation
        for i, t in enumerate(tenants):
            plans = await provider.list_plans(t.id, limit=100)
            assert len(plans) == ops_per_tenant, (
                f"Tenant {i} has {len(plans)} plans, expected {ops_per_tenant}"
            )
            # Verify all plans belong to this tenant's workload
            for p in plans:
                assert p.name.startswith(f"Plan-{i}-"), (
                    f"Tenant {i} has plan '{p.name}' from another tenant"
                )

            events = await provider.list_events(t.id, limit=100)
            assert len(events) == ops_per_tenant
            for e in events:
                assert e["payload"]["tenant_index"] == i

            contexts = await provider.list_contexts(t.id, limit=100)
            assert len(contexts) == ops_per_tenant


# ---------------------------------------------------------------------------
# Event sourcing roundtrip
# ---------------------------------------------------------------------------


class TestEventSourcingRoundtrip:
    async def test_rebuild_state_from_events(
        self, provider: InMemoryProvider
    ) -> None:
        """Full event sourcing: emit events, then replay to rebuild state."""
        tenant = Tenant(slug="es-co", name="ES Co")
        await provider.save_tenant(tenant)
        store = EventStore(provider, tenant.id)

        # Simulate a plan lifecycle through events
        plan_id = str(uuid4())
        await store.emit("plan.created", {"plan_id": plan_id, "name": "Deploy"})
        await store.emit("plan.validated", {"plan_id": plan_id})
        await store.emit("plan.started", {"plan_id": plan_id})
        await store.emit("action.completed", {"plan_id": plan_id, "action": "build", "success": True})
        await store.emit("action.completed", {"plan_id": plan_id, "action": "test", "success": True})
        await store.emit("plan.completed", {"plan_id": plan_id})

        # Replay and rebuild
        events = await store.replay()
        assert len(events) == 6

        # Rebuild plan state from events
        state = {"state": "unknown", "actions_completed": 0}

        async def projector(event_type: str, payload: dict) -> None:
            if event_type == "plan.created":
                state["state"] = "draft"
            elif event_type == "plan.validated":
                state["state"] = "planned"
            elif event_type == "plan.started":
                state["state"] = "executing"
            elif event_type == "action.completed" and payload.get("success"):
                state["actions_completed"] += 1
            elif event_type == "plan.completed":
                state["state"] = "completed"

        count = await store.replay_with_handler(projector)
        assert count == 6
        assert state["state"] == "completed"
        assert state["actions_completed"] == 2


# ---------------------------------------------------------------------------
# Model roundtrip (Pydantic serialization)
# ---------------------------------------------------------------------------


class TestModelRoundtrip:
    async def test_plan_roundtrip(self, provider: InMemoryProvider) -> None:
        """Plan survives save → retrieve cycle with all fields intact."""
        tenant = Tenant(slug="roundtrip-co", name="Roundtrip Co")
        await provider.save_tenant(tenant)

        plan = Plan(
            id=uuid4(),
            name="Complex Plan",
            description="Has everything",
            state=PlanState.DRAFT,
            actions=[
                PlanAction(
                    plan_id=uuid4(),
                    index=0,
                    name="Step 1",
                    skill_name="build",
                    skill_args={"target": "prod"},
                    dependencies=[],
                ),
            ],
            metadata={"source": "test", "priority": "high"},
            created_by="test-suite",
        )
        await provider.save_plan(tenant.id, plan)
        retrieved = await provider.get_plan(tenant.id, plan.id)

        assert retrieved is not None
        assert retrieved.name == "Complex Plan"
        assert retrieved.description == "Has everything"
        assert len(retrieved.actions) == 1
        assert retrieved.actions[0].skill_name == "build"
        assert retrieved.actions[0].skill_args == {"target": "prod"}
        assert retrieved.metadata["priority"] == "high"
        assert retrieved.created_by == "test-suite"

    async def test_context_with_messages_roundtrip(
        self, provider: InMemoryProvider
    ) -> None:
        """AgentContext with messages and metadata roundtrips cleanly."""
        tenant = Tenant(slug="ctx-co", name="Ctx Co")
        await provider.save_tenant(tenant)

        ctx = AgentContext(
            tenant_id=tenant.id,
            messages=[
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
            ],
            metadata={"route": "direct", "response": "Hi there"},
            skill_results={"web_search": {"results": ["a", "b"]}},
        )
        await provider.save_context(tenant.id, ctx)
        retrieved = await provider.get_context(tenant.id, ctx.id)

        assert retrieved is not None
        assert len(retrieved.messages) == 2
        assert retrieved.messages[0].content == "Hello"
        assert retrieved.metadata["route"] == "direct"
        assert "web_search" in retrieved.skill_results


# ---------------------------------------------------------------------------
# Full integration: Config + Guard + EventStore
# ---------------------------------------------------------------------------


class TestFullIntegration:
    async def test_guarded_event_store_with_config(
        self, provider: InMemoryProvider
    ) -> None:
        """Full stack: Guard → Provider → EventStore → ConfigResolver."""
        guard = TenantGuard(provider)
        resolver = ConfigResolver(provider)

        # Setup tenant with custom config
        tenant = Tenant(
            slug="integrated-co",
            name="Integrated Co",
            config=TenantConfig(
                default_model="claude-sonnet-4-20250514",
                features={"planning": True},
                max_concurrent_agents=25,
            ),
        )
        await guard.save_tenant(tenant)

        # Setup risk policy
        policy = RiskPolicy(name="integrated-policy", approval_threshold=40)
        await guard.save_risk_policy(tenant.id, policy)

        # Resolve config
        config = await resolver.resolve_agent_config(tenant.id)
        assert config.model == "claude-sonnet-4-20250514"
        assert await resolver.is_feature_enabled(tenant.id, "planning") is True

        # Create event store through guard
        store = EventStore(guard, tenant.id)

        # Run through lifecycle
        await store.emit("agent.start", {"config": config.model_dump()})
        await store.emit("routing.complete", {"intent": "complex_task"})
        await store.emit("planning.complete", {"plan_id": "plan-1"})
        await store.emit("agent.complete", {"response": "Done"})

        # Verify
        events = await store.replay()
        assert len(events) == 4
        assert events[0]["event_type"] == "agent.start"
        assert events[-1]["event_type"] == "agent.complete"

        # Verify isolation — different tenant sees nothing
        other_tenant = Tenant(slug="other-co", name="Other Co")
        await guard.save_tenant(other_tenant)
        other_store = EventStore(guard, other_tenant.id)
        other_events = await other_store.replay()
        assert len(other_events) == 0
