"""Tests for the AI Copilot builder."""

from __future__ import annotations

import pytest

from aria_core.copilot import CopilotBuilder, CopilotRequest, CopilotResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> CopilotBuilder:
    return CopilotBuilder()


def _make_request(desc: str, **kwargs) -> CopilotRequest:
    return CopilotRequest(user_description=desc, **kwargs)


# ---------------------------------------------------------------------------
# Archetype keyword mapping
# ---------------------------------------------------------------------------

class TestArchetypeSelection:
    """Rule-based generation picks the right archetype for each keyword group."""

    @pytest.mark.parametrize(
        "keyword,expected_archetype",
        [
            ("research", "research-analyst"),
            ("analyze", "research-analyst"),
            ("code", "code-assistant"),
            ("develop", "code-assistant"),
            ("build", "code-assistant"),
            ("write", "content-writer"),
            ("content", "content-writer"),
            ("blog", "content-writer"),
            ("data", "data-engineer"),
            ("sql", "data-engineer"),
            ("pipeline", "data-engineer"),
            ("support", "support-agent"),
            ("help", "support-agent"),
            ("customer", "support-agent"),
            ("security", "security-auditor"),
            ("audit", "security-auditor"),
            ("scan", "security-auditor"),
            ("ops", "ops-coordinator"),
            ("monitor", "ops-coordinator"),
            ("deploy", "ops-coordinator"),
        ],
    )
    async def test_keyword_selects_archetype(
        self, builder: CopilotBuilder, keyword: str, expected_archetype: str
    ):
        result = await builder.generate(_make_request(f"I need an agent that can {keyword} things"))
        assert result.agent_config["archetype"] == expected_archetype

    async def test_unknown_description_falls_back_to_generic(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("do something completely unrelated"))
        assert result.agent_config["archetype"] == "generic"


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

class TestComplexityEstimation:
    async def test_simple_description_low_steps(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("simple task"))
        assert result.agent_config["max_steps"] <= 10

    async def test_complex_description_high_steps(self, builder: CopilotBuilder):
        result = await builder.generate(
            _make_request("Build a comprehensive multi-step pipeline that does many complex things")
        )
        assert result.agent_config["max_steps"] >= 20

    async def test_creative_description_raises_temperature(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("creative brainstorm ideas for a campaign"))
        assert result.agent_config["temperature"] >= 0.5


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

class TestSkillExtraction:
    async def test_extracts_search_skill(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("search the web and summarize results"))
        assert "web-search" in result.agent_config["skills"]
        assert "summarization" in result.agent_config["skills"]

    async def test_extracts_code_skills(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("code review and test the module"))
        skills = result.agent_config["skills"]
        assert "code-generation" in skills
        assert "code-review" in skills
        assert "testing" in skills


# ---------------------------------------------------------------------------
# Full generate cycle
# ---------------------------------------------------------------------------

class TestFullCycle:
    async def test_result_has_all_fields(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("research market trends"))
        assert isinstance(result, CopilotResult)
        assert "model" in result.agent_config
        assert "system_prompt" in result.agent_config
        assert "temperature" in result.agent_config
        assert "max_steps" in result.agent_config
        assert "skills" in result.agent_config
        assert "name" in result.plan
        assert "description" in result.plan
        assert "actions" in result.plan
        assert "score" in result.risk_assessment
        assert "level" in result.risk_assessment
        assert "factors" in result.risk_assessment
        assert isinstance(result.suggestions, list)

    async def test_plan_actions_have_dependencies(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("code and test the app"))
        actions = result.plan["actions"]
        assert len(actions) >= 2
        assert actions[0]["id"] == "init"
        assert actions[-1]["id"] == "finalize"

    async def test_default_model(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("research something"))
        assert result.agent_config["model"] == "claude-sonnet-4-20250514"

    async def test_preferred_model_override(self, builder: CopilotBuilder):
        result = await builder.generate(
            _make_request("research something", preferred_model="gpt-4o")
        )
        assert result.agent_config["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Constraints override
# ---------------------------------------------------------------------------

class TestConstraints:
    async def test_constraints_override_config(self, builder: CopilotBuilder):
        result = await builder.generate(
            _make_request(
                "code a REST API",
                constraints={"temperature": 0.0, "max_steps": 50},
            )
        )
        assert result.agent_config["temperature"] == 0.0
        assert result.agent_config["max_steps"] == 50

    async def test_constraints_add_new_keys(self, builder: CopilotBuilder):
        result = await builder.generate(
            _make_request("research papers", constraints={"timeout": 300})
        )
        assert result.agent_config["timeout"] == 300


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    async def test_security_archetype_has_risk_factor(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("security audit the server"))
        assert result.risk_assessment["score"] > 0
        assert any("sensitive" in f for f in result.risk_assessment["factors"])

    async def test_simple_task_low_risk(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("simple task"))
        assert result.risk_assessment["level"] == "low"


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:
    async def test_generic_archetype_suggests_keywords(self, builder: CopilotBuilder):
        result = await builder.generate(_make_request("do something vague"))
        assert any("keyword" in s.lower() for s in result.suggestions)
