"""Pricing calculator — tier definitions and usage-based cost projections.

Aria Core pricing tiers:
- Starter: $0/mo, 1 tenant, 1k API calls, 5 agents, community support
- Pro: $99/mo, 5 tenants, 50k API calls, 25 agents, email support
- Business: $499/mo, 25 tenants, 500k API calls, 100 agents, priority support
- Enterprise: Custom, unlimited everything, SLA, dedicated support

Usage-based overages:
- API calls: $0.001/call over limit
- Events: $0.0005/event over limit
- Agent runs: $0.01/run over limit
- Storage: $0.10/GB/mo over limit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PricingTier:
    name: str
    slug: str
    monthly_price: float
    max_tenants: int
    max_api_calls: int
    max_events: int
    max_agent_runs: int
    max_agents: int
    max_storage_gb: float
    support_level: str
    features: list[str] = field(default_factory=list)
    is_custom: bool = False


TIERS: list[PricingTier] = [
    PricingTier(
        name="Starter",
        slug="starter",
        monthly_price=0,
        max_tenants=1,
        max_api_calls=1_000,
        max_events=5_000,
        max_agent_runs=100,
        max_agents=5,
        max_storage_gb=1.0,
        support_level="Community",
        features=[
            "Single tenant",
            "FSM runtime",
            "Risk scoring",
            "In-memory persistence",
            "REST API",
            "Config Portal",
        ],
    ),
    PricingTier(
        name="Pro",
        slug="pro",
        monthly_price=99,
        max_tenants=5,
        max_api_calls=50_000,
        max_events=250_000,
        max_agent_runs=5_000,
        max_agents=25,
        max_storage_gb=10.0,
        support_level="Email",
        features=[
            "Multi-tenant (5)",
            "PostgreSQL persistence",
            "Event sourcing + replay",
            "Deep Bridge consensus",
            "WebSocket streaming",
            "JWT auth + RBAC",
            "Agent archetypes",
            "Usage billing",
        ],
    ),
    PricingTier(
        name="Business",
        slug="business",
        monthly_price=499,
        max_tenants=25,
        max_api_calls=500_000,
        max_events=2_500_000,
        max_agent_runs=50_000,
        max_agents=100,
        max_storage_gb=100.0,
        support_level="Priority",
        features=[
            "Multi-tenant (25)",
            "RS256 JWKS key rotation",
            "Helm chart deployment",
            "HPA autoscaling",
            "Custom risk policies per tenant",
            "Approval gate builder",
            "Stripe billing integration",
            "99.9% SLA",
        ],
    ),
    PricingTier(
        name="Enterprise",
        slug="enterprise",
        monthly_price=0,  # Custom
        max_tenants=999_999,
        max_api_calls=999_999_999,
        max_events=999_999_999,
        max_agent_runs=999_999_999,
        max_agents=999_999,
        max_storage_gb=999_999.0,
        support_level="Dedicated",
        is_custom=True,
        features=[
            "Unlimited tenants",
            "White-label branding",
            "Custom domain",
            "Dedicated infrastructure",
            "SOC 2 compliance",
            "Custom SLA",
            "Onboarding + training",
            "24/7 dedicated support",
        ],
    ),
]

OVERAGE_RATES = {
    "api_call": 0.001,
    "event": 0.0005,
    "agent_run": 0.01,
    "storage_gb": 0.10,
}


@dataclass
class CostProjection:
    tier: str
    base_cost: float
    overage_cost: float
    total_cost: float
    breakdown: dict[str, float]
    recommended: bool = False


class PricingCalculator:
    """Calculate costs based on projected usage."""

    def __init__(self, tiers: list[PricingTier] | None = None) -> None:
        self.tiers = tiers or TIERS

    def get_tiers(self) -> list[dict[str, Any]]:
        """Get all pricing tiers as dicts."""
        return [
            {
                "name": t.name,
                "slug": t.slug,
                "monthly_price": t.monthly_price,
                "max_tenants": t.max_tenants,
                "max_api_calls": t.max_api_calls,
                "max_events": t.max_events,
                "max_agent_runs": t.max_agent_runs,
                "max_agents": t.max_agents,
                "max_storage_gb": t.max_storage_gb,
                "support_level": t.support_level,
                "features": t.features,
                "is_custom": t.is_custom,
            }
            for t in self.tiers
        ]

    def calculate(
        self,
        api_calls: int = 0,
        events: int = 0,
        agent_runs: int = 0,
        agents: int = 0,
        tenants: int = 1,
        storage_gb: float = 0,
    ) -> list[CostProjection]:
        """Project costs for each tier based on usage estimates."""
        projections: list[CostProjection] = []

        for tier in self.tiers:
            if tier.is_custom:
                projections.append(CostProjection(
                    tier=tier.name,
                    base_cost=0,
                    overage_cost=0,
                    total_cost=0,
                    breakdown={},
                    recommended=False,
                ))
                continue

            # Check if tier can handle the tenant/agent count
            if tenants > tier.max_tenants or agents > tier.max_agents:
                overage = float("inf")
            else:
                overage = 0.0

            breakdown: dict[str, float] = {}

            if api_calls > tier.max_api_calls:
                over = (api_calls - tier.max_api_calls) * OVERAGE_RATES["api_call"]
                breakdown["api_call_overage"] = round(over, 2)
                overage += over

            if events > tier.max_events:
                over = (events - tier.max_events) * OVERAGE_RATES["event"]
                breakdown["event_overage"] = round(over, 2)
                overage += over

            if agent_runs > tier.max_agent_runs:
                over = (agent_runs - tier.max_agent_runs) * OVERAGE_RATES["agent_run"]
                breakdown["agent_run_overage"] = round(over, 2)
                overage += over

            if storage_gb > tier.max_storage_gb:
                over = (storage_gb - tier.max_storage_gb) * OVERAGE_RATES["storage_gb"]
                breakdown["storage_overage"] = round(over, 2)
                overage += over

            total = tier.monthly_price + overage if overage != float("inf") else float("inf")

            projections.append(CostProjection(
                tier=tier.name,
                base_cost=tier.monthly_price,
                overage_cost=round(overage, 2) if overage != float("inf") else 0,
                total_cost=round(total, 2) if total != float("inf") else 0,
                breakdown=breakdown,
                recommended=False,
            ))

        # Mark cheapest viable option as recommended
        viable = [p for p in projections if p.total_cost > 0 and p.total_cost != float("inf")]
        if not viable:
            viable = [p for p in projections if p.total_cost == 0 and not any(t.is_custom for t in self.tiers if t.name == p.tier)]
        if viable:
            cheapest = min(viable, key=lambda p: p.total_cost)
            cheapest.recommended = True

        return projections

    def recommend_tier(
        self,
        api_calls: int = 0,
        events: int = 0,
        agent_runs: int = 0,
        agents: int = 0,
        tenants: int = 1,
        storage_gb: float = 0,
    ) -> dict[str, Any]:
        """Get the recommended tier and cost projection."""
        projections = self.calculate(
            api_calls=api_calls, events=events, agent_runs=agent_runs,
            agents=agents, tenants=tenants, storage_gb=storage_gb,
        )
        recommended = next((p for p in projections if p.recommended), projections[0])
        return {
            "recommended_tier": recommended.tier,
            "monthly_cost": recommended.total_cost,
            "base_cost": recommended.base_cost,
            "overage_cost": recommended.overage_cost,
            "breakdown": recommended.breakdown,
            "all_tiers": [
                {
                    "tier": p.tier,
                    "total_cost": p.total_cost,
                    "recommended": p.recommended,
                }
                for p in projections
            ],
        }
