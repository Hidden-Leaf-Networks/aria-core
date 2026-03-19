"""Deterministic risk scoring engine.

Calculates risk scores (0-100) for actions based on multiple factors:
1. Skill category base score (0-40 points)
2. Impact scope modifier (0-30 points)
3. Historical penalty (0-10 points)
4. Context modifiers (0-20 points)

All calculations are deterministic and reproducible via input hashing.

Usage:
    from aria_core.permissions import RiskEngine, RiskScoreInput, SkillCategory, ImpactScope

    engine = RiskEngine()
    score = engine.calculate(RiskScoreInput(
        skill_name="send_email",
        skill_category=SkillCategory.EXTERNAL,
        impact_scope=ImpactScope.EXTERNAL,
    ))
    print(f"Risk: {score.score}/100 ({score.level})")
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from uuid import uuid4

from aria_core.permissions.models import (
    ImpactScope,
    RiskFactor,
    RiskPolicy,
    RiskScore,
    RiskScoreInput,
    SkillCategory,
)

logger = logging.getLogger("aria_core.permissions")

DEFAULT_SKILL_CATEGORY_WEIGHTS: dict[str, float] = {
    SkillCategory.READ: 0.1,
    SkillCategory.WRITE: 0.3,
    SkillCategory.EXEC: 0.5,
    SkillCategory.EXTERNAL: 0.4,
}

DEFAULT_IMPACT_SCOPE_WEIGHTS: dict[str, float] = {
    ImpactScope.LOCAL: 0.1,
    ImpactScope.USER: 0.3,
    ImpactScope.SYSTEM: 0.6,
    ImpactScope.EXTERNAL: 0.5,
}


class RiskEngine:
    """Deterministic risk scoring engine.

    Calculates reproducible risk scores (0-100) for actions based on
    skill category, impact scope, history, and context modifiers.
    """

    CALCULATION_VERSION = "1.0.0"

    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy
        self._skill_weights = (
            policy.skill_category_weights if policy and policy.skill_category_weights
            else DEFAULT_SKILL_CATEGORY_WEIGHTS
        )
        self._scope_weights = (
            policy.impact_scope_weights if policy and policy.impact_scope_weights
            else DEFAULT_IMPACT_SCOPE_WEIGHTS
        )

    @property
    def approval_threshold(self) -> int:
        return self.policy.approval_threshold if self.policy else 50

    @property
    def first_execution_modifier(self) -> float:
        return self.policy.first_execution_modifier if self.policy else 1.2

    @property
    def failure_history_modifier(self) -> float:
        return self.policy.failure_history_modifier if self.policy else 0.05

    @property
    def violation_history_modifier(self) -> float:
        return self.policy.violation_history_modifier if self.policy else 0.1

    def calculate(self, input: RiskScoreInput) -> RiskScore:
        """Calculate deterministic risk score for an action."""
        factors: list[RiskFactor] = []

        # 1. Skill category (0-40)
        cat_score, cat_factor = self._calc_skill_category(input)
        factors.append(cat_factor)

        # 2. Impact scope (0-30)
        scope_score, scope_factor = self._calc_scope(input)
        factors.append(scope_factor)

        # 3. Historical penalty (0-10)
        hist_score, hist_factor = self._calc_history(input)
        factors.append(hist_factor)

        # 4. Context modifiers (0-20)
        ctx_score, ctx_factor = self._calc_context(input)
        factors.append(ctx_factor)

        total = int(min(cat_score + scope_score + hist_score + ctx_score, 100))
        level = self._level_from_score(total)
        requires_approval = total >= self.approval_threshold
        input_hash = self._hash_input(input)

        return RiskScore(
            id=uuid4(),
            score=total,
            level=level,
            requires_approval=requires_approval,
            factors=factors,
            input_hash=input_hash,
            calculated_at=datetime.now(timezone.utc),
            calculation_version=self.CALCULATION_VERSION,
        )

    def _calc_skill_category(self, input: RiskScoreInput) -> tuple[float, RiskFactor]:
        key = input.skill_category.value if hasattr(input.skill_category, "value") else str(input.skill_category)
        weight = self._skill_weights.get(key, 0.2)
        score = weight * 40.0
        return score, RiskFactor(name="skill_category", category="base", weight=weight, raw_value=40.0)

    def _calc_scope(self, input: RiskScoreInput) -> tuple[float, RiskFactor]:
        key = input.impact_scope.value if hasattr(input.impact_scope, "value") else str(input.impact_scope)
        weight = self._scope_weights.get(key, 0.3)
        score = weight * 30.0
        return score, RiskFactor(name="impact_scope", category="scope", weight=weight, raw_value=30.0)

    def _calc_history(self, input: RiskScoreInput) -> tuple[float, RiskFactor]:
        failure_penalty = min(input.historical_failures * self.failure_history_modifier * 10, 5.0)
        violation_penalty = min(input.historical_violations * self.violation_history_modifier * 10, 5.0)
        score = failure_penalty + violation_penalty
        weight = score / 10.0 if score > 0 else 0.0
        return score, RiskFactor(name="historical", category="history", weight=weight, raw_value=10.0)

    def _calc_context(self, input: RiskScoreInput) -> tuple[float, RiskFactor]:
        score = 0.0
        if input.is_first_execution:
            score += 2.0 * self.first_execution_modifier
        if input.has_sensitive_args:
            score += 3.0
        if input.targets_external_system:
            score += 2.0
        if input.requires_network:
            score += 1.5
        if input.modifies_persistent_state:
            score += 1.5
        score = min(score, 20.0)
        weight = score / 20.0 if score > 0 else 0.0
        return score, RiskFactor(name="context", category="context", weight=weight, raw_value=20.0)

    @staticmethod
    def _level_from_score(score: int) -> str:
        if score <= 20:
            return "safe"
        elif score <= 40:
            return "low"
        elif score <= 60:
            return "medium"
        elif score <= 80:
            return "high"
        else:
            return "critical"

    def _hash_input(self, input: RiskScoreInput) -> str:
        data = input.model_dump_json(exclude_none=True)
        data_with_version = f"{self.CALCULATION_VERSION}:{data}"
        return hashlib.sha256(data_with_version.encode()).hexdigest()[:16]

    def calculate_aggregate_risk(self, risk_scores: list[RiskScore]) -> tuple[int, str, bool]:
        """Calculate aggregate risk for a plan with multiple actions.

        Uses weighted average with emphasis on highest-risk actions.
        """
        if not risk_scores:
            return 0, "safe", False

        sorted_scores = sorted(risk_scores, key=lambda x: x.score, reverse=True)
        total_weight = 0.0
        weighted_sum = 0.0

        for i, rs in enumerate(sorted_scores):
            weight = 1.0 / (i + 1)
            weighted_sum += rs.score * weight
            total_weight += weight

        aggregate = int(weighted_sum / total_weight) if total_weight > 0 else 0
        aggregate = max(aggregate, sorted_scores[0].score)

        level = self._level_from_score(aggregate)
        requires_approval = aggregate >= self.approval_threshold
        return aggregate, level, requires_approval

    def evaluate_quick(
        self, skill_category: SkillCategory, impact_scope: ImpactScope
    ) -> tuple[int, str, bool]:
        """Quick risk evaluation without full input."""
        cat_key = skill_category.value if hasattr(skill_category, "value") else str(skill_category)
        scope_key = impact_scope.value if hasattr(impact_scope, "value") else str(impact_scope)

        cat_weight = self._skill_weights.get(cat_key, 0.2)
        scope_weight = self._scope_weights.get(scope_key, 0.3)

        score = int(cat_weight * 40 + scope_weight * 30)
        level = self._level_from_score(score)
        requires_approval = score >= self.approval_threshold
        return score, level, requires_approval

    def update_policy(self, policy: RiskPolicy) -> None:
        self.policy = policy
        self._skill_weights = policy.skill_category_weights or DEFAULT_SKILL_CATEGORY_WEIGHTS
        self._scope_weights = policy.impact_scope_weights or DEFAULT_IMPACT_SCOPE_WEIGHTS
