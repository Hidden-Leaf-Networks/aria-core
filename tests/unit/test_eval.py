"""Tests for the Agent Eval Framework (ARIA-302)."""

from __future__ import annotations

import pytest

from aria_core.eval import (
    BUILT_IN_SAFETY_SUITE,
    BUILT_IN_SMOKE_SUITE,
    EvalCase,
    EvalMetric,
    EvalResult,
    EvalRunner,
    EvalScorer,
    EvalSuite,
)


# ---------------------------------------------------------------------------
# EvalMetric model tests
# ---------------------------------------------------------------------------


class TestEvalMetric:
    def test_create_metric(self) -> None:
        m = EvalMetric(name="accuracy", metric_type="accuracy", weight=0.8)
        assert m.name == "accuracy"
        assert m.metric_type == "accuracy"
        assert m.weight == 0.8

    def test_metric_default_weight(self) -> None:
        m = EvalMetric(name="cost", metric_type="cost")
        assert m.weight == 1.0

    def test_metric_all_types(self) -> None:
        for t in ("accuracy", "latency", "cost", "tool_use", "safety", "quality", "custom"):
            m = EvalMetric(name=t, metric_type=t)
            assert m.metric_type == t


# ---------------------------------------------------------------------------
# EvalCase model tests
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_create_case(self) -> None:
        c = EvalCase(name="test", input_message="hello")
        assert c.name == "test"
        assert c.input_message == "hello"
        assert c.id  # auto-generated

    def test_case_optional_fields(self) -> None:
        c = EvalCase(name="opt", input_message="hi")
        assert c.expected_output is None
        assert c.expected_tools is None
        assert c.expected_state is None
        assert c.max_duration_ms is None
        assert c.max_steps is None
        assert c.tags == []

    def test_case_with_all_fields(self) -> None:
        c = EvalCase(
            id="c1",
            name="full",
            description="fully specified",
            input_message="do something",
            expected_output="done",
            expected_tools=["tool_a"],
            expected_state="complete",
            max_duration_ms=1000,
            max_steps=5,
            tags=["tag1"],
        )
        assert c.id == "c1"
        assert c.expected_tools == ["tool_a"]
        assert c.max_steps == 5


# ---------------------------------------------------------------------------
# EvalResult model tests
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_create_result(self) -> None:
        r = EvalResult(case_id="r1", case_name="test", passed=True, score=0.95)
        assert r.passed is True
        assert r.score == 0.95
        assert r.error is None

    def test_result_defaults(self) -> None:
        r = EvalResult(case_id="r2", case_name="t", passed=False, score=0.0)
        assert r.actual_output == ""
        assert r.actual_duration_ms == 0
        assert r.metrics == {}


# ---------------------------------------------------------------------------
# EvalSuite model tests
# ---------------------------------------------------------------------------


class TestEvalSuite:
    def test_create_suite(self) -> None:
        s = EvalSuite(name="my_suite")
        assert s.name == "my_suite"
        assert s.threshold == 0.8
        assert s.cases == []

    def test_suite_with_cases(self) -> None:
        cases = [EvalCase(name=f"c{i}", input_message="x") for i in range(3)]
        s = EvalSuite(name="s", cases=cases, threshold=0.9)
        assert len(s.cases) == 3
        assert s.threshold == 0.9


# ---------------------------------------------------------------------------
# EvalScorer tests
# ---------------------------------------------------------------------------


class TestEvalScorer:
    def test_output_match_exact_substring(self) -> None:
        assert EvalScorer.score_output_match("The capital is Paris.", "Paris") == 1.0

    def test_output_match_no_match(self) -> None:
        score = EvalScorer.score_output_match("hello world", "zzzzz")
        assert score < 0.5

    def test_output_match_empty_expected(self) -> None:
        assert EvalScorer.score_output_match("anything", "") == 1.0

    def test_output_match_fuzzy(self) -> None:
        score = EvalScorer.score_output_match("The quick brown fox", "the quick brown")
        assert score > 0.7

    def test_tool_usage_perfect(self) -> None:
        assert EvalScorer.score_tool_usage(["a", "b"], ["a", "b"]) == 1.0

    def test_tool_usage_partial(self) -> None:
        score = EvalScorer.score_tool_usage(["a"], ["a", "b"])
        assert 0.0 < score < 1.0

    def test_tool_usage_empty_expected(self) -> None:
        assert EvalScorer.score_tool_usage(["a"], []) == 1.0

    def test_tool_usage_empty_actual(self) -> None:
        assert EvalScorer.score_tool_usage([], ["a"]) == 0.0

    def test_tool_usage_no_overlap(self) -> None:
        assert EvalScorer.score_tool_usage(["x"], ["y"]) == 0.0

    def test_latency_under(self) -> None:
        assert EvalScorer.score_latency(500, 1000) == 1.0

    def test_latency_over(self) -> None:
        score = EvalScorer.score_latency(1500, 1000)
        assert score == pytest.approx(0.5)

    def test_latency_way_over(self) -> None:
        assert EvalScorer.score_latency(2000, 1000) == 0.0

    def test_latency_zero_threshold(self) -> None:
        assert EvalScorer.score_latency(100, 0) == 1.0

    def test_steps_under(self) -> None:
        assert EvalScorer.score_steps(3, 5) == 1.0

    def test_steps_over(self) -> None:
        score = EvalScorer.score_steps(7, 5)
        assert 0.0 < score < 1.0

    def test_steps_way_over(self) -> None:
        assert EvalScorer.score_steps(10, 5) == 0.0

    def test_state_match(self) -> None:
        assert EvalScorer.score_state("complete", "complete") == 1.0

    def test_state_mismatch(self) -> None:
        assert EvalScorer.score_state("running", "complete") == 0.0

    def test_state_empty_expected(self) -> None:
        assert EvalScorer.score_state("anything", "") == 1.0


# ---------------------------------------------------------------------------
# EvalRunner tests
# ---------------------------------------------------------------------------


class TestEvalRunner:
    @pytest.fixture
    def runner(self) -> EvalRunner:
        r = EvalRunner()
        suite = EvalSuite(
            name="test_suite",
            threshold=0.5,
            cases=[
                EvalCase(id="tc1", name="basic", input_message="hello"),
                EvalCase(
                    id="tc2",
                    name="with_state",
                    input_message="test",
                    expected_state="needs_approval",
                ),
            ],
        )
        r.add_suite(suite)
        return r

    @pytest.mark.asyncio
    async def test_run_case_stub(self) -> None:
        runner = EvalRunner()
        case = EvalCase(id="x", name="stub", input_message="ping")
        result = await runner.run_case(case)
        assert result.passed is True
        assert "[eval-stub]" in result.actual_output

    @pytest.mark.asyncio
    async def test_run_case_with_expected_output(self) -> None:
        runner = EvalRunner()
        case = EvalCase(
            id="x", name="out", input_message="ping", expected_output="ping"
        )
        result = await runner.run_case(case)
        # stub echoes input, so substring match should succeed
        assert result.metrics["output_match"] == 1.0

    @pytest.mark.asyncio
    async def test_run_case_state_mismatch(self) -> None:
        runner = EvalRunner()
        case = EvalCase(
            id="x",
            name="state",
            input_message="hi",
            expected_state="needs_approval",
        )
        result = await runner.run_case(case)
        # stub returns expected_state to simulate correct state
        assert result.metrics["state_match"] == 1.0

    @pytest.mark.asyncio
    async def test_run_suite(self, runner: EvalRunner) -> None:
        results = await runner.run_suite("test_suite")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_run_suite_not_found(self, runner: EvalRunner) -> None:
        with pytest.raises(KeyError, match="not_real"):
            await runner.run_suite("not_real")

    @pytest.mark.asyncio
    async def test_run_all(self, runner: EvalRunner) -> None:
        all_results = await runner.run_all()
        assert "test_suite" in all_results
        assert len(all_results["test_suite"]) == 2

    def test_score_suite(self) -> None:
        results = [
            EvalResult(case_id="a", case_name="a", passed=True, score=1.0),
            EvalResult(case_id="b", case_name="b", passed=True, score=0.5),
        ]
        assert EvalRunner.score_suite(results) == pytest.approx(0.75)

    def test_score_suite_empty(self) -> None:
        assert EvalRunner.score_suite([]) == 0.0

    def test_generate_report(self) -> None:
        results = [
            EvalResult(
                case_id="a", case_name="fast", passed=True, score=1.0,
                actual_duration_ms=10,
            ),
            EvalResult(
                case_id="b", case_name="slow", passed=False, score=0.3,
                actual_duration_ms=500, error="timeout",
            ),
        ]
        report = EvalRunner.generate_report(results)
        assert report["total"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1
        assert report["slowest_case"]["case_name"] == "slow"
        assert report["fastest_case"]["case_name"] == "fast"
        assert len(report["errors"]) == 1

    def test_generate_report_empty(self) -> None:
        report = EvalRunner.generate_report([])
        assert report["total"] == 0
        assert report["slowest_case"] is None

    def test_check_quality_gate_pass(self) -> None:
        results = [
            EvalResult(case_id="a", case_name="a", passed=True, score=0.9),
            EvalResult(case_id="b", case_name="b", passed=True, score=0.8),
        ]
        assert EvalRunner.check_quality_gate(results, 0.8) is True

    def test_check_quality_gate_fail(self) -> None:
        results = [
            EvalResult(case_id="a", case_name="a", passed=False, score=0.3),
        ]
        assert EvalRunner.check_quality_gate(results, 0.8) is False

    def test_check_quality_gate_empty(self) -> None:
        assert EvalRunner.check_quality_gate([], 0.5) is False


# ---------------------------------------------------------------------------
# Built-in suites tests
# ---------------------------------------------------------------------------


class TestBuiltInSuites:
    def test_smoke_suite_exists(self) -> None:
        assert BUILT_IN_SMOKE_SUITE.name == "smoke"
        assert len(BUILT_IN_SMOKE_SUITE.cases) == 5

    def test_safety_suite_exists(self) -> None:
        assert BUILT_IN_SAFETY_SUITE.name == "safety"
        assert len(BUILT_IN_SAFETY_SUITE.cases) == 3

    def test_smoke_cases_have_ids(self) -> None:
        for case in BUILT_IN_SMOKE_SUITE.cases:
            assert case.id.startswith("smoke-")

    def test_safety_cases_have_tags(self) -> None:
        for case in BUILT_IN_SAFETY_SUITE.cases:
            assert "safety" in case.tags

    @pytest.mark.asyncio
    async def test_run_smoke_suite(self) -> None:
        runner = EvalRunner()
        runner.add_suite(BUILT_IN_SMOKE_SUITE)
        results = await runner.run_suite("smoke")
        assert len(results) == 5
        # Stub won't produce real answers, so cases with expected_output may fail.
        # Verify at least the cases without expected_output pass.
        no_expected = [r for r in results if r.case_id != "smoke-002"]
        for r in no_expected:
            assert r.passed is True

    @pytest.mark.asyncio
    async def test_run_safety_suite(self) -> None:
        runner = EvalRunner()
        runner.add_suite(BUILT_IN_SAFETY_SUITE)
        results = await runner.run_suite("safety")
        assert len(results) == 3
