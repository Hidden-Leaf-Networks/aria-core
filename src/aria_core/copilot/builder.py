"""Copilot builder: converts natural language descriptions into agent configurations."""

from __future__ import annotations

import re
from typing import Any

from aria_core.runtime.models import BaseModel


# ---------------------------------------------------------------------------
# Request / Result models
# ---------------------------------------------------------------------------

class CopilotRequest(BaseModel):
    """Input for the copilot builder."""

    user_description: str
    constraints: dict[str, Any] | None = None
    preferred_model: str | None = None


class CopilotResult(BaseModel):
    """Output produced by the copilot builder."""

    agent_config: dict[str, Any]
    plan: dict[str, Any]
    risk_assessment: dict[str, Any]
    suggestions: list[str]


# ---------------------------------------------------------------------------
# Archetype definitions
# ---------------------------------------------------------------------------

_ARCHETYPES: dict[str, dict[str, Any]] = {
    "research-analyst": {
        "system_prompt": (
            "You are a meticulous research analyst. Gather information, "
            "cross-reference sources, and produce well-structured reports."
        ),
        "temperature": 0.3,
        "max_steps": 15,
        "skills": ["web-search", "summarization", "citation"],
    },
    "code-assistant": {
        "system_prompt": (
            "You are an expert software engineer. Write clean, tested, "
            "and well-documented code following best practices."
        ),
        "temperature": 0.2,
        "max_steps": 20,
        "skills": ["code-generation", "code-review", "testing"],
    },
    "content-writer": {
        "system_prompt": (
            "You are a skilled content writer. Produce engaging, clear, "
            "and audience-appropriate written content."
        ),
        "temperature": 0.7,
        "max_steps": 10,
        "skills": ["writing", "editing", "seo"],
    },
    "data-engineer": {
        "system_prompt": (
            "You are a data engineer. Design pipelines, write queries, "
            "and ensure data quality and integrity."
        ),
        "temperature": 0.2,
        "max_steps": 15,
        "skills": ["sql", "data-pipeline", "data-validation"],
    },
    "support-agent": {
        "system_prompt": (
            "You are a friendly and efficient support agent. Resolve "
            "customer issues quickly while maintaining a positive tone."
        ),
        "temperature": 0.4,
        "max_steps": 10,
        "skills": ["knowledge-base", "ticketing", "escalation"],
    },
    "security-auditor": {
        "system_prompt": (
            "You are a security auditor. Identify vulnerabilities, "
            "assess risks, and recommend mitigations."
        ),
        "temperature": 0.1,
        "max_steps": 20,
        "skills": ["vulnerability-scan", "compliance-check", "reporting"],
    },
    "ops-coordinator": {
        "system_prompt": (
            "You are an operations coordinator. Monitor systems, "
            "manage deployments, and respond to incidents."
        ),
        "temperature": 0.2,
        "max_steps": 15,
        "skills": ["monitoring", "deployment", "incident-response"],
    },
    "generic": {
        "system_prompt": (
            "You are a helpful AI assistant. Follow the user's instructions "
            "carefully and provide clear, accurate responses."
        ),
        "temperature": 0.5,
        "max_steps": 10,
        "skills": [],
    },
}

# Keyword → archetype mapping
_KEYWORD_MAP: dict[str, str] = {
    "research": "research-analyst",
    "analyze": "research-analyst",
    "code": "code-assistant",
    "develop": "code-assistant",
    "build": "code-assistant",
    "write": "content-writer",
    "content": "content-writer",
    "blog": "content-writer",
    "data": "data-engineer",
    "sql": "data-engineer",
    "pipeline": "data-engineer",
    "support": "support-agent",
    "help": "support-agent",
    "customer": "support-agent",
    "security": "security-auditor",
    "audit": "security-auditor",
    "scan": "security-auditor",
    "ops": "ops-coordinator",
    "monitor": "ops-coordinator",
    "deploy": "ops-coordinator",
}

# Skill-hint keywords
_SKILL_KEYWORDS: dict[str, str] = {
    "search": "web-search",
    "web": "web-search",
    "summarize": "summarization",
    "summarization": "summarization",
    "code": "code-generation",
    "test": "testing",
    "review": "code-review",
    "write": "writing",
    "edit": "editing",
    "seo": "seo",
    "sql": "sql",
    "data": "data-pipeline",
    "validate": "data-validation",
    "ticket": "ticketing",
    "escalate": "escalation",
    "scan": "vulnerability-scan",
    "compliance": "compliance-check",
    "monitor": "monitoring",
    "deploy": "deployment",
    "incident": "incident-response",
    "report": "reporting",
    "cite": "citation",
    "citation": "citation",
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CopilotBuilder:
    """Generates agent configurations from natural language descriptions."""

    def __init__(self, adapter: Any | None = None) -> None:
        self._adapter = adapter

    # -- public API ----------------------------------------------------------

    async def generate(self, request: CopilotRequest) -> CopilotResult:
        """Generate an agent config from *request*.

        Uses the LLM adapter when available; otherwise falls back to
        deterministic rule-based generation.
        """
        if self._adapter is not None:
            return await self._llm_generate(request)
        return self._rule_based_generate(request)

    # -- LLM path (stub — delegates to adapter) -----------------------------

    async def _llm_generate(self, request: CopilotRequest) -> CopilotResult:
        """Use an LLM adapter to produce the config.

        The adapter is expected to expose an async ``generate`` method that
        accepts a prompt string and returns a text response.  We construct a
        structured prompt so the LLM returns JSON we can parse.

        For now we fall back to rule-based if anything goes wrong.
        """
        try:
            prompt = (
                "Given the following description of an AI agent, produce a JSON object with keys: "
                "agent_config, plan, risk_assessment, suggestions.\n\n"
                f"Description: {request.user_description}\n"
            )
            if request.constraints:
                prompt += f"Constraints: {request.constraints}\n"
            if request.preferred_model:
                prompt += f"Preferred model: {request.preferred_model}\n"

            _response = await self._adapter.generate(prompt)
            # Real implementation would parse the JSON response here.
            # For safety, fall through to rule-based until parser is wired.
        except Exception:
            pass

        return self._rule_based_generate(request)

    # -- deterministic path --------------------------------------------------

    def _rule_based_generate(self, request: CopilotRequest) -> CopilotResult:
        """Deterministic config generation driven by keyword matching."""
        desc = request.user_description

        archetype_name = self._select_archetype(desc)
        archetype = _ARCHETYPES[archetype_name]
        complexity = self._estimate_complexity(desc)
        skills = self._parse_skills_from_description(desc) or list(archetype["skills"])

        model = request.preferred_model or "claude-sonnet-4-20250514"

        agent_config: dict[str, Any] = {
            "model": model,
            "system_prompt": archetype["system_prompt"],
            "temperature": complexity["temperature"],
            "max_steps": complexity["max_steps"],
            "skills": skills,
            "archetype": archetype_name,
        }

        # Apply user constraints as overrides
        if request.constraints:
            agent_config.update(request.constraints)

        plan: dict[str, Any] = {
            "name": f"{archetype_name}-plan",
            "description": f"Execution plan for: {desc[:120]}",
            "actions": self._build_actions(archetype_name, skills),
        }

        risk = self._assess_risk(archetype_name, complexity)

        suggestions = self._build_suggestions(archetype_name, skills, complexity)

        return CopilotResult(
            agent_config=agent_config,
            plan=plan,
            risk_assessment=risk,
            suggestions=suggestions,
        )

    # -- helpers -------------------------------------------------------------

    def _select_archetype(self, desc: str) -> str:
        """Match *desc* to the best archetype via keyword scanning."""
        lower = desc.lower()
        for keyword, archetype in _KEYWORD_MAP.items():
            if keyword in lower:
                return archetype
        return "generic"

    def _parse_skills_from_description(self, desc: str) -> list[str]:
        """Extract likely skills from natural language."""
        lower = desc.lower()
        found: list[str] = []
        for keyword, skill in _SKILL_KEYWORDS.items():
            if keyword in lower and skill not in found:
                found.append(skill)
        return found

    def _estimate_complexity(self, desc: str) -> dict[str, Any]:
        """Estimate max_steps and temperature from description length/keywords."""
        word_count = len(desc.split())

        # Complex indicators
        complex_markers = ["complex", "multi-step", "advanced", "comprehensive", "detailed"]
        simple_markers = ["simple", "quick", "basic", "single", "easy"]

        lower = desc.lower()
        is_complex = any(m in lower for m in complex_markers)
        is_simple = any(m in lower for m in simple_markers)

        if is_complex or word_count > 50:
            max_steps = 25
            temperature = 0.3
        elif is_simple or word_count < 10:
            max_steps = 5
            temperature = 0.5
        else:
            max_steps = 12
            temperature = 0.4

        # Creative descriptions bump temperature
        creative_markers = ["creative", "brainstorm", "innovative", "artistic"]
        if any(m in lower for m in creative_markers):
            temperature = min(temperature + 0.3, 1.0)

        archetype = self._select_archetype(desc)
        arch_temp = _ARCHETYPES.get(archetype, _ARCHETYPES["generic"])["temperature"]
        temperature = round((temperature + arch_temp) / 2, 2)

        return {"max_steps": max_steps, "temperature": temperature}

    def _build_actions(self, archetype: str, skills: list[str]) -> list[dict[str, Any]]:
        """Build a default action list for the plan."""
        actions: list[dict[str, Any]] = [
            {"id": "init", "name": "Initialize", "depends_on": []},
        ]
        for i, skill in enumerate(skills):
            actions.append({
                "id": f"step_{i + 1}",
                "name": f"Execute {skill}",
                "depends_on": ["init"] if i == 0 else [f"step_{i}"],
            })
        actions.append({
            "id": "finalize",
            "name": "Finalize output",
            "depends_on": [actions[-1]["id"]],
        })
        return actions

    def _assess_risk(self, archetype: str, complexity: dict[str, Any]) -> dict[str, Any]:
        """Produce a risk assessment dict."""
        factors: list[str] = []
        score = 0.0

        if complexity["max_steps"] > 20:
            factors.append("High step count increases execution time and cost")
            score += 0.3
        if complexity["temperature"] > 0.6:
            factors.append("Higher temperature may produce less predictable outputs")
            score += 0.2
        if archetype in ("security-auditor", "ops-coordinator"):
            factors.append("Archetype operates on sensitive systems")
            score += 0.2

        score = round(min(score, 1.0), 2)
        if score >= 0.6:
            level = "high"
        elif score >= 0.3:
            level = "medium"
        else:
            level = "low"

        return {"score": score, "level": level, "factors": factors}

    def _build_suggestions(
        self,
        archetype: str,
        skills: list[str],
        complexity: dict[str, Any],
    ) -> list[str]:
        """Generate helpful suggestions."""
        suggestions: list[str] = []
        if not skills:
            suggestions.append("Consider adding skills to extend agent capabilities.")
        if complexity["max_steps"] > 20:
            suggestions.append("Consider breaking complex tasks into sub-agents.")
        if archetype == "generic":
            suggestions.append(
                "Provide more specific keywords (e.g. 'research', 'code', 'deploy') "
                "for a tailored configuration."
            )
        if complexity["temperature"] > 0.6:
            suggestions.append("Lower the temperature if deterministic output is preferred.")
        return suggestions
