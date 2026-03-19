"""Deep Bridge multi-model consensus validation.

Queries multiple LLM providers in parallel to get independent validation
of high-stakes actions before approval. No single model decides alone.

Consensus modes:
- UNANIMOUS: All models must agree
- MAJORITY: More than half must agree
- ANY: At least one model approves

Usage:
    from aria_core.orchestration import DeepBridgeValidator, ConsensusMode

    validator = DeepBridgeValidator(
        consensus_mode=ConsensusMode.MAJORITY,
        models=["gpt-4o", "claude-3-5-sonnet-20241022"],
    )
    result = await validator.validate(
        action_id=uuid,
        plan_id=uuid,
        skill_name="email_send",
        args={"to": ["user@example.com"]},
        risk_score=55,
        risk_level="MEDIUM",
    )
    if result.consensus_approved:
        # Proceed
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member

logger = logging.getLogger("aria_core.orchestration")


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------


class ConsensusMode(StrEnum):
    """Consensus requirement modes."""

    UNANIMOUS = "unanimous"  # All models must agree
    MAJORITY = "majority"    # >50% must agree
    ANY = "any"              # At least one approves


class ModelVote(BaseModel):
    """Individual model's vote on an action."""

    model_name: str
    provider: str
    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    response_time_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeepBridgeResult(BaseModel):
    """Result of Deep Bridge consensus validation."""

    id: UUID = Field(default_factory=uuid4)
    action_id: UUID
    plan_id: UUID

    # Consensus result
    consensus_reached: bool
    consensus_approved: bool
    consensus_mode: ConsensusMode

    # Voting details
    votes: list[ModelVote] = Field(default_factory=list)
    total_votes: int = 0
    approve_votes: int = 0
    reject_votes: int = 0

    # Aggregate confidence
    average_confidence: float = 0.0
    min_confidence: float = 0.0

    # Timing
    validation_started_at: datetime
    validation_completed_at: datetime
    total_time_ms: int

    # Context
    skill_name: str
    risk_score: int


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class ModelQueryProvider(Protocol):
    """Protocol for querying a model provider.

    Implement this to add custom model backends (local models, proxies, etc).
    """

    async def query(self, model: str, prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        ...


# ---------------------------------------------------------------------------
# Built-in providers (using httpx directly, no SDK dependency)
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """Query OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"

    async def query(self, model: str, prompt: str) -> str:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.1,
                },
                timeout=25.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class AnthropicProvider:
    """Query Anthropic API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def query(self, model: str, prompt: str) -> str:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=25.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]


class XAIProvider:
    """Query xAI/Grok API (OpenAI-compatible)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("XAI_API_KEY", "")

    async def query(self, model: str, prompt: str) -> str:
        provider = OpenAIProvider(api_key=self.api_key, base_url="https://api.x.ai/v1")
        return await provider.query(model, prompt)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

# Default model-to-provider mapping
DEFAULT_PROVIDERS: dict[str, type[OpenAIProvider] | type[AnthropicProvider] | type[XAIProvider]] = {
    "gpt": OpenAIProvider,
    "claude": AnthropicProvider,
    "grok": XAIProvider,
}


class DeepBridgeValidator:
    """Multi-model consensus validator.

    Queries multiple LLM providers in parallel for independent validation
    of high-stakes actions. Supports pluggable providers and consensus modes.
    """

    def __init__(
        self,
        consensus_mode: ConsensusMode = ConsensusMode.MAJORITY,
        min_votes: int = 2,
        timeout_seconds: int = 30,
        models: list[str] | None = None,
        providers: dict[str, ModelQueryProvider] | None = None,
        prompt_builder: Callable[..., str] | None = None,
    ) -> None:
        """Initialize Deep Bridge validator.

        Args:
            consensus_mode: How to determine consensus.
            min_votes: Minimum votes required for valid consensus.
            timeout_seconds: Timeout for all model queries.
            models: List of model identifiers to query. Auto-detected if None.
            providers: Custom provider instances keyed by model name prefix.
            prompt_builder: Custom function to build validation prompts.
        """
        self.consensus_mode = consensus_mode
        self.min_votes = min_votes
        self.timeout_seconds = timeout_seconds
        self.models = models or self._detect_available_models()
        self._providers = providers or {}
        self._prompt_builder = prompt_builder

    def _detect_available_models(self) -> list[str]:
        """Auto-detect available models from environment API keys."""
        models = []
        if os.environ.get("OPENAI_API_KEY"):
            models.append("gpt-4o")
        if os.environ.get("ANTHROPIC_API_KEY"):
            models.append("claude-3-5-sonnet-20241022")
        if os.environ.get("XAI_API_KEY"):
            models.append("grok-2-latest")

        if len(models) < 2:
            logger.warning(
                "Deep Bridge requires at least 2 models for meaningful consensus. Found: %d",
                len(models),
            )
        return models

    def _get_provider(self, model: str) -> ModelQueryProvider:
        """Resolve provider for a model."""
        # Check custom providers first
        for prefix, provider in self._providers.items():
            if model.startswith(prefix) or prefix in model.lower():
                return provider

        # Fall back to built-in providers
        for prefix, provider_class in DEFAULT_PROVIDERS.items():
            if prefix in model.lower():
                return provider_class()

        raise ValueError(f"No provider found for model: {model}")

    async def validate(
        self,
        action_id: UUID,
        plan_id: UUID,
        skill_name: str,
        args: dict[str, Any],
        risk_score: int,
        risk_level: str,
        context: dict[str, Any] | None = None,
    ) -> DeepBridgeResult:
        """Validate an action through multi-model consensus.

        Args:
            action_id: Unique action identifier.
            plan_id: Plan this action belongs to.
            skill_name: Skill/tool to be executed.
            args: Skill arguments (should be sanitized).
            risk_score: Calculated risk score (0-100).
            risk_level: Risk level string (SAFE/LOW/MEDIUM/HIGH).
            context: Optional additional context for the prompt.

        Returns:
            DeepBridgeResult with consensus outcome and all votes.
        """
        start_time = datetime.now(timezone.utc)

        # Build prompt
        if self._prompt_builder:
            prompt = self._prompt_builder(
                skill_name=skill_name,
                args=args,
                risk_score=risk_score,
                risk_level=risk_level,
                context=context or {},
            )
        else:
            prompt = self._default_prompt(
                skill_name=skill_name,
                args=args,
                risk_score=risk_score,
                risk_level=risk_level,
            )

        # Query all models concurrently
        tasks = [self._query_model(model, prompt) for model in self.models]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Deep Bridge validation timed out after %ds", self.timeout_seconds)
            results = []

        # Collect votes
        votes: list[ModelVote] = []
        for result in results:
            if isinstance(result, ModelVote):
                votes.append(result)
            elif isinstance(result, Exception):
                logger.warning("Model query failed: %s", result)

        # Calculate consensus
        end_time = datetime.now(timezone.utc)
        total_time_ms = int((end_time - start_time).total_seconds() * 1000)

        approve_count = sum(1 for v in votes if v.approved)
        reject_count = len(votes) - approve_count

        consensus_reached = len(votes) >= self.min_votes
        consensus_approved = self._check_consensus(
            approve_votes=approve_count,
            total_votes=len(votes),
            mode=self.consensus_mode,
        )

        confidences = [v.confidence for v in votes] if votes else [0.0]

        return DeepBridgeResult(
            action_id=action_id,
            plan_id=plan_id,
            consensus_reached=consensus_reached,
            consensus_approved=consensus_approved,
            consensus_mode=self.consensus_mode,
            votes=votes,
            total_votes=len(votes),
            approve_votes=approve_count,
            reject_votes=reject_count,
            average_confidence=sum(confidences) / len(confidences),
            min_confidence=min(confidences),
            validation_started_at=start_time,
            validation_completed_at=end_time,
            total_time_ms=total_time_ms,
            skill_name=skill_name,
            risk_score=risk_score,
        )

    async def _query_model(self, model: str, prompt: str) -> ModelVote:
        """Query a single model and parse its vote."""
        start_time = datetime.now(timezone.utc)
        provider = self._get_provider(model)
        response = await provider.query(model, prompt)

        # Parse structured response
        try:
            data = json.loads(response)
            approved = data.get("approved", False)
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
        except (json.JSONDecodeError, KeyError, TypeError):
            approved = "approved" in response.lower() and "not approved" not in response.lower()
            confidence = 0.5
            reasoning = response[:200]

        end_time = datetime.now(timezone.utc)
        response_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Determine provider name
        provider_name = type(provider).__name__.replace("Provider", "").lower()

        return ModelVote(
            model_name=model,
            provider=provider_name,
            approved=approved,
            confidence=min(max(confidence, 0.0), 1.0),
            reasoning=reasoning,
            response_time_ms=response_time_ms,
        )

    @staticmethod
    def _default_prompt(
        skill_name: str,
        args: dict[str, Any],
        risk_score: int,
        risk_level: str,
    ) -> str:
        """Default validation prompt."""
        return f"""You are a security validation system. Analyze the following action and determine if it should be approved.

ACTION DETAILS:
- Skill: {skill_name}
- Arguments: {args}
- Risk Score: {risk_score}/100
- Risk Level: {risk_level}

GUIDELINES:
1. Consider the potential impact of this action
2. Evaluate if the action aligns with the stated context
3. Check for any suspicious patterns in the arguments
4. Consider reversibility of the action

Respond with a JSON object:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}

Only respond with the JSON object, no other text."""

    @staticmethod
    def _check_consensus(
        approve_votes: int,
        total_votes: int,
        mode: ConsensusMode,
    ) -> bool:
        """Determine if consensus approves the action."""
        if total_votes == 0:
            return False

        if mode == ConsensusMode.UNANIMOUS:
            return approve_votes == total_votes
        if mode == ConsensusMode.MAJORITY:
            return approve_votes > total_votes / 2
        if mode == ConsensusMode.ANY:
            return approve_votes > 0

        return False


def create_deep_bridge_validator(
    consensus_mode: str = "majority",
    models: list[str] | None = None,
) -> DeepBridgeValidator:
    """Create a Deep Bridge validator with sensible defaults.

    Args:
        consensus_mode: "unanimous", "majority", or "any".
        models: Model list. Auto-detected if None.

    Returns:
        Configured DeepBridgeValidator.
    """
    mode = ConsensusMode(consensus_mode.lower())
    return DeepBridgeValidator(consensus_mode=mode, models=models)
