"""Tests for the persistence layer — InMemoryProvider + tenant isolation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.permissions.models import Approval, ApprovalGate, ApprovalState, RiskPolicy
from aria_core.planning.models import Plan, PlanAction, PlanState, ActionState
from aria_core.persistence.memory import InMemoryProvider, TenantNotFoundError
from aria_core.runtime.models import AgentContext, ChatMessage, MessageRole
from aria_core.tenant.models import Tenant, TenantConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def tenant_a() -> Tenant:
    return Tenant(slug="tenant-a", name="Tenant A")


@pytest.fixture
def tenant_b() -> Tenant:
    return Tenant(slug="tenant-b", name="Tenant B")


def _make_plan(name: str = "Test Plan") -> Plan:
    plan_id = uuid4()
    return Plan(
        id=plan_id,
        name=name,
        state=PlanState.DRAFT,
        actions=[
            PlanAction(plan_id=plan_id, index=0, name="Step 1"),
        ],
    )


def _make_approval(plan_id: uuid4 | None = None) -> Approval:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return Approval(
        plan_id=plan_id or uuid4(),
        gate_id=uuid4(),
        gate_name="test-gate",
        risk_score=65,
        state=ApprovalState.PENDING,
        expires_at=now + timedelta(hours=1),
    )


def _make_context(tenant_id: uuid4 | None = None) -> AgentContext:
    kwargs = {}
    if tenant_id:
        kwargs["tenant_id"] = tenant_id
    return AgentContext(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------


class TestTenantCRUD:
    async def test_save_and_get_tenant(self, provider: InMemoryProvider) -> None:
        tenant = Tenant(slug="acme-co", name="Acme")
        saved = await provider.save_tenant(tenant)
        assert saved.id == tenant.id

        retrieved = await provider.get_tenant(tenant.id)
        assert retrieved is not None
        assert retrieved.slug == "acme-co"

    async def test_get_by_slug(self, provider: InMemoryProvider) -> None:
        tenant = Tenant(slug="by-slug", name="Test")
        await provider.save_tenant(tenant)

        found = await provider.get_tenant_by_slug("by-slug")
        assert found is not None
        assert found.id == tenant.id

        missing = await provider.get_tenant_by_slug("nope")
        assert missing is None

    async def test_list_tenants(self, provider: InMemoryProvider) -> None:
        t1 = Tenant(slug="alpha-co", name="Alpha")
        t2 = Tenant(slug="beta-co", name="Beta", is_active=False)
        await provider.save_tenant(t1)
        await provider.save_tenant(t2)

        active = await provider.list_tenants(active_only=True)
        assert len(active) == 1
        assert active[0].slug == "alpha-co"

        all_tenants = await provider.list_tenants(active_only=False)
        assert len(all_tenants) == 2

    async def test_update_tenant_config(self, provider: InMemoryProvider) -> None:
        tenant = Tenant(slug="update-me", name="Update Me")
        await provider.save_tenant(tenant)

        new_config = TenantConfig(default_model="gpt-4", max_concurrent_agents=25)
        updated = await provider.update_tenant_config(tenant.id, new_config)
        assert updated.config.default_model == "gpt-4"
        assert updated.config.max_concurrent_agents == 25

    async def test_update_nonexistent_tenant_raises(self, provider: InMemoryProvider) -> None:
        with pytest.raises(TenantNotFoundError):
            await provider.update_tenant_config(uuid4(), TenantConfig())


# ---------------------------------------------------------------------------
# Plan storage + tenant isolation
# ---------------------------------------------------------------------------


class TestPlanStorage:
    async def test_save_and_get_plan(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        plan = _make_plan("Alpha Plan")
        await provider.save_plan(tenant_a.id, plan)

        retrieved = await provider.get_plan(tenant_a.id, plan.id)
        assert retrieved is not None
        assert retrieved.name == "Alpha Plan"

    async def test_tenant_isolation_plans(
        self, provider: InMemoryProvider, tenant_a: Tenant, tenant_b: Tenant
    ) -> None:
        """Tenant A's plans are invisible to Tenant B."""
        await provider.save_tenant(tenant_a)
        await provider.save_tenant(tenant_b)

        plan_a = _make_plan("A's Plan")
        await provider.save_plan(tenant_a.id, plan_a)

        # Tenant B cannot see Tenant A's plan
        invisible = await provider.get_plan(tenant_b.id, plan_a.id)
        assert invisible is None

        # Tenant B's list is empty
        b_plans = await provider.list_plans(tenant_b.id)
        assert len(b_plans) == 0

    async def test_list_plans_with_filter(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)

        p1 = _make_plan("Draft Plan")
        p2 = Plan(id=uuid4(), name="Done Plan", state=PlanState.COMPLETED)
        await provider.save_plan(tenant_a.id, p1)
        await provider.save_plan(tenant_a.id, p2)

        drafts = await provider.list_plans(tenant_a.id, state="draft")
        assert len(drafts) == 1
        assert drafts[0].name == "Draft Plan"

    async def test_delete_plan(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        plan = _make_plan()
        await provider.save_plan(tenant_a.id, plan)

        deleted = await provider.delete_plan(tenant_a.id, plan.id)
        assert deleted is True

        gone = await provider.get_plan(tenant_a.id, plan.id)
        assert gone is None

        # Deleting again returns False
        assert await provider.delete_plan(tenant_a.id, plan.id) is False


# ---------------------------------------------------------------------------
# Approval storage + tenant isolation
# ---------------------------------------------------------------------------


class TestApprovalStorage:
    async def test_save_and_get_approval(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        approval = _make_approval()
        await provider.save_approval(tenant_a.id, approval)

        retrieved = await provider.get_approval(tenant_a.id, approval.id)
        assert retrieved is not None
        assert retrieved.risk_score == 65

    async def test_tenant_isolation_approvals(
        self, provider: InMemoryProvider, tenant_a: Tenant, tenant_b: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        await provider.save_tenant(tenant_b)

        approval = _make_approval()
        await provider.save_approval(tenant_a.id, approval)

        invisible = await provider.get_approval(tenant_b.id, approval.id)
        assert invisible is None

    async def test_list_approvals_by_state(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        a1 = _make_approval()
        a2 = _make_approval()
        # Manually set a2 to approved for filtering test
        from datetime import datetime, timezone

        a2_approved = Approval(
            id=a2.id,
            plan_id=a2.plan_id,
            gate_id=a2.gate_id,
            gate_name=a2.gate_name,
            risk_score=a2.risk_score,
            state=ApprovalState.APPROVED,
            expires_at=a2.expires_at,
            resolved_at=datetime.now(timezone.utc),
        )
        await provider.save_approval(tenant_a.id, a1)
        await provider.save_approval(tenant_a.id, a2_approved)

        pending = await provider.list_approvals(tenant_a.id, state="pending")
        assert len(pending) == 1

        approved = await provider.list_approvals(tenant_a.id, state="approved")
        assert len(approved) == 1


# ---------------------------------------------------------------------------
# Event store + tenant isolation
# ---------------------------------------------------------------------------


class TestEventStore:
    async def test_save_and_list_events(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)

        await provider.save_event(tenant_a.id, "plan.created", {"plan_id": "abc"})
        await provider.save_event(tenant_a.id, "plan.completed", {"plan_id": "abc"})
        await provider.save_event(tenant_a.id, "agent.start", {"context_id": "xyz"})

        all_events = await provider.list_events(tenant_a.id)
        assert len(all_events) == 3

        plan_events = await provider.list_events(tenant_a.id, event_type="plan.created")
        assert len(plan_events) == 1

    async def test_tenant_isolation_events(
        self, provider: InMemoryProvider, tenant_a: Tenant, tenant_b: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        await provider.save_tenant(tenant_b)

        await provider.save_event(tenant_a.id, "secret.event", {"data": "classified"})

        b_events = await provider.list_events(tenant_b.id)
        assert len(b_events) == 0

    async def test_count_events(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)

        await provider.save_event(tenant_a.id, "plan.created", {})
        await provider.save_event(tenant_a.id, "plan.created", {})
        await provider.save_event(tenant_a.id, "agent.start", {})

        total = await provider.count_events(tenant_a.id)
        assert total == 3

        plan_count = await provider.count_events(tenant_a.id, event_type="plan.created")
        assert plan_count == 2

    async def test_events_are_append_only(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        """Events can only be appended, never modified or deleted."""
        await provider.save_tenant(tenant_a)

        event = await provider.save_event(tenant_a.id, "test.event", {"val": 1})
        assert "id" in event
        assert "timestamp" in event

        # No delete or update methods exist — append-only by design
        assert not hasattr(provider, "delete_event")
        assert not hasattr(provider, "update_event")

    async def test_event_filtering_by_agent(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        agent_1 = uuid4()
        agent_2 = uuid4()

        await provider.save_event(tenant_a.id, "step.complete", {}, agent_id=agent_1)
        await provider.save_event(tenant_a.id, "step.complete", {}, agent_id=agent_2)
        await provider.save_event(tenant_a.id, "step.complete", {}, agent_id=agent_1)

        agent_1_events = await provider.list_events(tenant_a.id, agent_id=agent_1)
        assert len(agent_1_events) == 2


# ---------------------------------------------------------------------------
# Agent context storage + tenant isolation
# ---------------------------------------------------------------------------


class TestContextStorage:
    async def test_save_and_get_context(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        ctx = _make_context(tenant_a.id)
        await provider.save_context(tenant_a.id, ctx)

        retrieved = await provider.get_context(tenant_a.id, ctx.id)
        assert retrieved is not None
        assert retrieved.tenant_id == tenant_a.id

    async def test_tenant_isolation_contexts(
        self, provider: InMemoryProvider, tenant_a: Tenant, tenant_b: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        await provider.save_tenant(tenant_b)

        ctx = _make_context(tenant_a.id)
        await provider.save_context(tenant_a.id, ctx)

        invisible = await provider.get_context(tenant_b.id, ctx.id)
        assert invisible is None

    async def test_list_contexts_by_conversation(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        conv_id = uuid4()

        c1 = AgentContext(tenant_id=tenant_a.id, conversation_id=conv_id)
        c2 = AgentContext(tenant_id=tenant_a.id, conversation_id=conv_id)
        c3 = AgentContext(tenant_id=tenant_a.id, conversation_id=uuid4())

        await provider.save_context(tenant_a.id, c1)
        await provider.save_context(tenant_a.id, c2)
        await provider.save_context(tenant_a.id, c3)

        conv_contexts = await provider.list_contexts(tenant_a.id, conversation_id=conv_id)
        assert len(conv_contexts) == 2


# ---------------------------------------------------------------------------
# Risk policy + approval gate storage
# ---------------------------------------------------------------------------


class TestPolicyStorage:
    async def test_save_and_get_risk_policy(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        policy = RiskPolicy(name="strict", approval_threshold=30)
        await provider.save_risk_policy(tenant_a.id, policy)

        retrieved = await provider.get_risk_policy(tenant_a.id, policy.id)
        assert retrieved is not None
        assert retrieved.approval_threshold == 30

    async def test_get_active_risk_policy(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        p1 = RiskPolicy(name="old", is_active=False)
        p2 = RiskPolicy(name="current", is_active=True)
        await provider.save_risk_policy(tenant_a.id, p1)
        await provider.save_risk_policy(tenant_a.id, p2)

        active = await provider.get_active_risk_policy(tenant_a.id)
        assert active is not None
        assert active.name == "current"

    async def test_save_and_list_approval_gates(
        self, provider: InMemoryProvider, tenant_a: Tenant
    ) -> None:
        await provider.save_tenant(tenant_a)
        g1 = ApprovalGate(name="low-risk", risk_threshold=30)
        g2 = ApprovalGate(name="high-risk", risk_threshold=70, is_active=False)
        await provider.save_approval_gate(tenant_a.id, g1)
        await provider.save_approval_gate(tenant_a.id, g2)

        active_gates = await provider.list_approval_gates(tenant_a.id, active_only=True)
        assert len(active_gates) == 1
        assert active_gates[0].name == "low-risk"

        all_gates = await provider.list_approval_gates(tenant_a.id, active_only=False)
        assert len(all_gates) == 2


# ---------------------------------------------------------------------------
# Cross-tenant isolation stress test
# ---------------------------------------------------------------------------


class TestCrossTenantIsolation:
    async def test_full_isolation_across_all_stores(
        self, provider: InMemoryProvider
    ) -> None:
        """Comprehensive test: 3 tenants, each with data in all stores.
        Verify complete isolation.
        """
        tenants = [
            Tenant(slug=f"tenant-{i}", name=f"Tenant {i}")
            for i in range(3)
        ]
        for t in tenants:
            await provider.save_tenant(t)

        # Each tenant creates data
        for i, t in enumerate(tenants):
            await provider.save_plan(t.id, _make_plan(f"Plan {i}"))
            await provider.save_approval(t.id, _make_approval())
            await provider.save_event(t.id, f"event.{i}", {"idx": i})
            await provider.save_context(t.id, _make_context(t.id))
            await provider.save_risk_policy(
                t.id, RiskPolicy(name=f"policy-{i}")
            )
            await provider.save_approval_gate(
                t.id, ApprovalGate(name=f"gate-{i}", risk_threshold=50)
            )

        # Verify isolation for each tenant
        for i, t in enumerate(tenants):
            plans = await provider.list_plans(t.id)
            assert len(plans) == 1
            assert plans[0].name == f"Plan {i}"

            approvals = await provider.list_approvals(t.id)
            assert len(approvals) == 1

            events = await provider.list_events(t.id)
            assert len(events) == 1
            assert events[0]["payload"]["idx"] == i

            contexts = await provider.list_contexts(t.id)
            assert len(contexts) == 1

            policy = await provider.get_active_risk_policy(t.id)
            assert policy is not None
            assert policy.name == f"policy-{i}"

            gates = await provider.list_approval_gates(t.id)
            assert len(gates) == 1
            assert gates[0].name == f"gate-{i}"
