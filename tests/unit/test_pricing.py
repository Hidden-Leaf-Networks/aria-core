"""Tests for pricing calculator."""

from __future__ import annotations

from aria_core.billing.pricing import PricingCalculator, TIERS, OVERAGE_RATES


class TestPricingCalculator:
    def test_get_tiers(self) -> None:
        calc = PricingCalculator()
        tiers = calc.get_tiers()
        assert len(tiers) == 4
        assert tiers[0]["slug"] == "starter"
        assert tiers[1]["slug"] == "pro"
        assert tiers[2]["slug"] == "business"
        assert tiers[3]["slug"] == "enterprise"

    def test_starter_free_within_limits(self) -> None:
        calc = PricingCalculator()
        projections = calc.calculate(api_calls=500, events=2000, agent_runs=50)
        starter = next(p for p in projections if p.tier == "Starter")
        assert starter.base_cost == 0
        assert starter.overage_cost == 0
        assert starter.total_cost == 0

    def test_overage_calculated(self) -> None:
        calc = PricingCalculator()
        # 2000 API calls on Starter (limit 1000) = 1000 * $0.001 = $1.00
        projections = calc.calculate(api_calls=2000)
        starter = next(p for p in projections if p.tier == "Starter")
        assert starter.overage_cost > 0
        assert "api_call_overage" in starter.breakdown
        assert starter.breakdown["api_call_overage"] == 1.0

    def test_pro_recommended_for_medium_usage(self) -> None:
        calc = PricingCalculator()
        result = calc.recommend_tier(
            api_calls=30000, events=100000, agent_runs=2000, agents=10, tenants=3
        )
        # Pro should be cheapest viable (Starter can't handle 3 tenants)
        assert result["recommended_tier"] == "Pro"

    def test_business_recommended_for_high_usage(self) -> None:
        calc = PricingCalculator()
        result = calc.recommend_tier(
            api_calls=200000, events=1000000, agent_runs=20000, agents=50, tenants=15
        )
        assert result["recommended_tier"] == "Business"

    def test_enterprise_excluded_from_recommendation(self) -> None:
        calc = PricingCalculator()
        projections = calc.calculate(api_calls=100)
        enterprise = next(p for p in projections if p.tier == "Enterprise")
        assert enterprise.recommended is False

    def test_all_tiers_in_result(self) -> None:
        calc = PricingCalculator()
        result = calc.recommend_tier(api_calls=100)
        assert len(result["all_tiers"]) == 4

    def test_overage_rates_defined(self) -> None:
        assert "api_call" in OVERAGE_RATES
        assert "event" in OVERAGE_RATES
        assert "agent_run" in OVERAGE_RATES
        assert "storage_gb" in OVERAGE_RATES
