"""Risk scoring example — demonstrates the permission system.

No API keys required. Demonstrates:
- Deterministic risk calculation
- Policy customization
- Approval workflow

Run:
    python examples/risk_scoring.py
"""

from uuid import uuid4

from aria_core.permissions import (
    ApprovalEngine,
    ImpactScope,
    RiskEngine,
    RiskPolicy,
    RiskScoreInput,
    SkillCategory,
)


def main():
    print("=== Aria Core — Risk Scoring ===\n")

    engine = RiskEngine()

    # Score different actions
    actions = [
        ("read_file", SkillCategory.READ, ImpactScope.LOCAL),
        ("write_database", SkillCategory.WRITE, ImpactScope.SYSTEM),
        ("send_email", SkillCategory.EXTERNAL, ImpactScope.EXTERNAL),
        ("execute_code", SkillCategory.EXEC, ImpactScope.SYSTEM),
    ]

    print("Default Policy Scores:")
    print(f"{'Action':<20} {'Score':>5} {'Level':<10} {'Approval?':<10}")
    print("-" * 50)

    for name, category, scope in actions:
        score = engine.calculate(RiskScoreInput(
            skill_name=name,
            skill_category=category,
            impact_scope=scope,
        ))
        print(f"{name:<20} {score.score:>5} {score.level:<10} {'Yes' if score.requires_approval else 'No':<10}")

    # Custom strict policy
    print("\n\nStrict Policy (threshold=20):")
    strict = RiskPolicy(name="strict", approval_threshold=20)
    strict_engine = RiskEngine(policy=strict)

    print(f"{'Action':<20} {'Score':>5} {'Level':<10} {'Approval?':<10}")
    print("-" * 50)

    for name, category, scope in actions:
        score = strict_engine.calculate(RiskScoreInput(
            skill_name=name,
            skill_category=category,
            impact_scope=scope,
        ))
        print(f"{name:<20} {score.score:>5} {score.level:<10} {'Yes' if score.requires_approval else 'No':<10}")

    # Approval workflow
    print("\n\n=== Approval Workflow ===\n")

    approval_engine = ApprovalEngine()
    risk_score = engine.calculate(RiskScoreInput(
        skill_name="deploy_production",
        skill_category=SkillCategory.EXEC,
        impact_scope=ImpactScope.SYSTEM,
        has_sensitive_args=True,
        targets_external_system=True,
    ))

    print(f"Action: deploy_production")
    print(f"Risk: {risk_score.score}/100 ({risk_score.level})")
    print(f"Requires approval: {risk_score.requires_approval}")

    if approval_engine.requires_approval(risk_score.score):
        approval = approval_engine.create_approval(
            plan_id=uuid4(),
            risk_score=risk_score,
        )
        print(f"Approval created: {approval.id}")
        print(f"State: {approval.state}")

        response = approval_engine.approve(approval.id, approver_id="admin-1", reason="Reviewed and safe")
        print(f"After approval: {response.approval.state}")
        print(f"Plan state: {response.plan_state}")


if __name__ == "__main__":
    main()
