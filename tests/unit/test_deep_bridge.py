"""Tests for Deep Bridge multi-model consensus validation."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from aria_core.orchestration.deep_bridge import (
    ConsensusMode,
    DeepBridgeValidator,
    ModelVote,
)


# ---------------------------------------------------------------------------
# Stub provider that returns deterministic responses
# ---------------------------------------------------------------------------


class StubProvider:
    """A fake model provider for testing."""

    def __init__(self, approved: bool = True, confidence: float = 0.9) -> None:
        self.approved = approved
        self.confidence = confidence
        self.call_count = 0

    async def query(self, model: str, prompt: str) -> str:
        self.call_count += 1
        return json.dumps({
            "approved": self.approved,
            "confidence": self.confidence,
            "reasoning": f"Stub says {'approve' if self.approved else 'reject'}",
        })


class FailingProvider:
    """Provider that always raises."""

    async def query(self, model: str, prompt: str) -> str:
        raise RuntimeError("Provider is down")


class MalformedProvider:
    """Provider that returns non-JSON."""

    async def query(self, model: str, prompt: str) -> str:
        return "This is not JSON but it says approved"


# ---------------------------------------------------------------------------
# Consensus logic tests
# ---------------------------------------------------------------------------


class TestConsensus:
    def test_unanimous_all_approve(self) -> None:
        assert DeepBridgeValidator._check_consensus(3, 3, ConsensusMode.UNANIMOUS) is True

    def test_unanimous_one_reject(self) -> None:
        assert DeepBridgeValidator._check_consensus(2, 3, ConsensusMode.UNANIMOUS) is False

    def test_majority_over_half(self) -> None:
        assert DeepBridgeValidator._check_consensus(2, 3, ConsensusMode.MAJORITY) is True

    def test_majority_exactly_half(self) -> None:
        assert DeepBridgeValidator._check_consensus(2, 4, ConsensusMode.MAJORITY) is False

    def test_any_one_approve(self) -> None:
        assert DeepBridgeValidator._check_consensus(1, 5, ConsensusMode.ANY) is True

    def test_any_none_approve(self) -> None:
        assert DeepBridgeValidator._check_consensus(0, 3, ConsensusMode.ANY) is False

    def test_zero_votes_always_false(self) -> None:
        for mode in ConsensusMode:
            assert DeepBridgeValidator._check_consensus(0, 0, mode) is False


# ---------------------------------------------------------------------------
# Validator integration tests (with stubs)
# ---------------------------------------------------------------------------


class TestDeepBridgeValidator:
    async def test_all_approve_majority(self) -> None:
        provider = StubProvider(approved=True, confidence=0.95)
        validator = DeepBridgeValidator(
            consensus_mode=ConsensusMode.MAJORITY,
            min_votes=2,
            models=["stub-model-a", "stub-model-b"],
            providers={"stub": provider},
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="test_skill",
            args={"key": "value"},
            risk_score=50,
            risk_level="MEDIUM",
        )

        assert result.consensus_reached is True
        assert result.consensus_approved is True
        assert result.total_votes == 2
        assert result.approve_votes == 2
        assert provider.call_count == 2

    async def test_all_reject(self) -> None:
        provider = StubProvider(approved=False, confidence=0.8)
        validator = DeepBridgeValidator(
            consensus_mode=ConsensusMode.MAJORITY,
            min_votes=2,
            models=["stub-a", "stub-b"],
            providers={"stub": provider},
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="risky_action",
            args={},
            risk_score=90,
            risk_level="HIGH",
        )

        assert result.consensus_approved is False
        assert result.reject_votes == 2

    async def test_mixed_votes_majority(self) -> None:
        approve_provider = StubProvider(approved=True)
        reject_provider = StubProvider(approved=False)

        validator = DeepBridgeValidator(
            consensus_mode=ConsensusMode.MAJORITY,
            min_votes=2,
            models=["approve-a", "approve-b", "reject-c"],
            providers={"approve": approve_provider, "reject": reject_provider},
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="test",
            args={},
            risk_score=30,
            risk_level="LOW",
        )

        assert result.total_votes == 3
        assert result.approve_votes == 2
        assert result.consensus_approved is True

    async def test_provider_failure_graceful(self) -> None:
        """Failed providers reduce vote count but don't crash."""
        good = StubProvider(approved=True)
        bad = FailingProvider()

        validator = DeepBridgeValidator(
            consensus_mode=ConsensusMode.ANY,
            min_votes=1,
            models=["good-model", "fail-model"],
            providers={"good": good, "fail": bad},
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="test",
            args={},
            risk_score=20,
            risk_level="LOW",
        )

        assert result.total_votes == 1
        assert result.consensus_approved is True

    async def test_malformed_response_parsing(self) -> None:
        """Non-JSON responses get fallback parsing."""
        provider = MalformedProvider()

        validator = DeepBridgeValidator(
            consensus_mode=ConsensusMode.ANY,
            min_votes=1,
            models=["bad-json"],
            providers={"bad": provider},
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="test",
            args={},
            risk_score=10,
            risk_level="SAFE",
        )

        assert result.total_votes == 1
        # The fallback parser finds "approved" in the text
        assert result.votes[0].confidence == 0.5

    async def test_not_enough_votes(self) -> None:
        """Consensus not reached if below min_votes."""
        validator = DeepBridgeValidator(
            consensus_mode=ConsensusMode.MAJORITY,
            min_votes=3,
            models=["stub-a", "stub-b"],
            providers={"stub": StubProvider(approved=True)},
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="test",
            args={},
            risk_score=50,
            risk_level="MEDIUM",
        )

        assert result.consensus_reached is False
        # Votes still approve but quorum not met
        assert result.total_votes == 2

    async def test_custom_prompt_builder(self) -> None:
        """Custom prompt builder is called."""
        prompts_seen: list[str] = []

        class CapturingProvider:
            async def query(self, model: str, prompt: str) -> str:
                prompts_seen.append(prompt)
                return json.dumps({"approved": True, "confidence": 1.0, "reasoning": "ok"})

        def custom_prompt(**kwargs: Any) -> str:
            return f"CUSTOM: {kwargs.get('skill_name')}"

        validator = DeepBridgeValidator(
            models=["cap-model"],
            providers={"cap": CapturingProvider()},
            prompt_builder=custom_prompt,
            min_votes=1,
        )

        await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="my_skill",
            args={},
            risk_score=10,
            risk_level="SAFE",
        )

        assert len(prompts_seen) == 1
        assert prompts_seen[0] == "CUSTOM: my_skill"

    async def test_confidence_metrics(self) -> None:
        """Average and min confidence are calculated correctly."""

        class VariableProvider:
            def __init__(self, confidence: float) -> None:
                self.confidence = confidence

            async def query(self, model: str, prompt: str) -> str:
                return json.dumps({
                    "approved": True,
                    "confidence": self.confidence,
                    "reasoning": "ok",
                })

        validator = DeepBridgeValidator(
            models=["high-model", "low-model"],
            providers={
                "high": VariableProvider(0.9),
                "low": VariableProvider(0.6),
            },
            min_votes=2,
        )

        result = await validator.validate(
            action_id=uuid4(),
            plan_id=uuid4(),
            skill_name="test",
            args={},
            risk_score=40,
            risk_level="LOW",
        )

        assert result.average_confidence == pytest.approx(0.75)
        assert result.min_confidence == pytest.approx(0.6)


class TestConsensusMode:
    def test_from_string(self) -> None:
        assert ConsensusMode("majority") == ConsensusMode.MAJORITY
        assert ConsensusMode("unanimous") == ConsensusMode.UNANIMOUS
        assert ConsensusMode("any") == ConsensusMode.ANY
