"""Tests for the single versus multi evaluation harness.

Offline. No model is called and no query is run; these cover the parts that
decide what the reported numbers mean: the split, the spend cap, the judge
parsers and the statistics.

The statistics tests matter most. A bootstrap interval that is subtly wrong
does not fail loudly, it just quietly supports a claim the data does not.
"""

import json
from unittest.mock import MagicMock

import pytest

from evaluation.compare_modes import (
    Meter,
    SpendCapExceeded,
    bootstrap_ci,
    build_split,
    estimate,
    judge_correctness,
    judge_grounding,
    load_queries,
    paired_bootstrap_ci,
    percentile,
    slice_rows,
    summarise,
)


def _response(text: str, tokens_in: int = 100, tokens_out: int = 20):
    r = MagicMock()
    r.content = text
    r.usage_metadata = {"input_tokens": tokens_in, "output_tokens": tokens_out}
    return r


def _model(text: str):
    m = MagicMock()
    m.invoke.return_value = _response(text)
    return m


def _row(**overrides):
    row = {
        "id": 1,
        "mode": "single",
        "multi_source": False,
        "correctness": 1.0,
        "claims": 4,
        "supported_claims": 3,
        "unsupported_claims": 1,
        "latency_ms": 1000.0,
        "input_tokens": 1000,
        "output_tokens": 200,
        "model_calls": 2,
        "cost_usd": 0.001,
        "error": None,
    }
    row.update(overrides)
    return row


# -- Query set and split ----------------------------------------------------


class TestQuerySetAndSplit:
    def test_loads_the_existing_set_plus_the_multi_source_additions(self):
        queries = load_queries()
        assert len(queries) == 62
        multi = [q for q in queries if len(q.get("needs_sources", [])) > 1]
        assert len(multi) == 12
        assert all(q["context_type"] == "mixed" for q in multi)

    def test_every_query_has_the_fields_the_harness_reads(self):
        for q in load_queries():
            assert isinstance(q["id"], int)
            assert q["question"].strip()
            assert q["ground_truth"].strip()
            assert q["needs_sources"]

    def test_query_ids_are_unique(self):
        ids = [q["id"] for q in load_queries()]
        assert len(ids) == len(set(ids))

    def test_split_is_deterministic(self):
        queries = load_queries()
        assert build_split(queries) == build_split(queries)

    def test_split_halves_do_not_overlap_and_cover_everything(self):
        queries = load_queries()
        split = build_split(queries)
        dev, holdout = set(split["dev"]), set(split["holdout"])
        assert dev.isdisjoint(holdout)
        assert dev | holdout == {q["id"] for q in queries}

    def test_multi_source_queries_appear_in_both_halves(self):
        """The slice the upgrade targets must not land entirely in one half."""
        queries = load_queries()
        split = build_split(queries)
        mixed = {q["id"] for q in queries if q["context_type"] == "mixed"}
        assert mixed & set(split["dev"])
        assert mixed & set(split["holdout"])

    def test_persisted_split_matches_the_computed_one(self):
        """A drifting split would mean the held out set was not held out."""
        from evaluation.compare_modes import _SPLIT_FILE

        if not _SPLIT_FILE.exists():
            pytest.skip("split has not been written yet")
        on_disk = json.loads(_SPLIT_FILE.read_text())
        assert on_disk == build_split(load_queries())


# -- Spend metering ---------------------------------------------------------


class TestMeter:
    def test_counts_provider_reported_usage(self):
        meter = Meter(cap_usd=1.0)
        meter.record(_response("x", tokens_in=1000, tokens_out=500))
        assert meter.input_tokens == 1000
        assert meter.output_tokens == 500
        assert meter.calls == 1

    def test_cost_uses_the_recorded_prices(self):
        from evaluation.compare_modes import PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M

        meter = Meter(cap_usd=1.0)
        meter.record(_response("x", tokens_in=1_000_000, tokens_out=1_000_000))
        assert meter.cost_usd == pytest.approx(PRICE_INPUT_PER_M + PRICE_OUTPUT_PER_M)

    def test_usage_without_metadata_counts_as_zero_not_guessed(self):
        meter = Meter(cap_usd=1.0)
        bare = MagicMock()
        bare.usage_metadata = None
        meter.record(bare)
        assert meter.input_tokens == 0
        assert meter.calls == 1

    def test_wrapped_model_meters_every_call(self):
        meter = Meter(cap_usd=1.0)
        wrapped = meter.wrap(_model("hello"))
        wrapped.invoke([])
        wrapped.invoke([])
        assert meter.calls == 2
        assert meter.input_tokens == 200

    def test_cap_raises_before_the_next_call_is_made(self):
        """The cap must stop the next call, not report an overshoot after it."""
        meter = Meter(cap_usd=0.0001)
        inner = MagicMock()
        inner.invoke.return_value = _response("hello", tokens_in=1000, tokens_out=1000)
        wrapped = meter.wrap(inner)
        wrapped.invoke([])  # pushes cost over the cap
        with pytest.raises(SpendCapExceeded):
            wrapped.invoke([])
        assert inner.invoke.call_count == 1

    def test_delta_reports_one_query_in_isolation(self):
        meter = Meter(cap_usd=1.0)
        meter.record(_response("x", 100, 10))
        before = meter.snapshot()
        meter.record(_response("x", 400, 40))
        delta = meter.delta(before)
        assert delta["input_tokens"] == 400
        assert delta["output_tokens"] == 40
        assert delta["calls"] == 1


# -- Judges -----------------------------------------------------------------


class TestJudges:
    @pytest.mark.parametrize(
        "verdict,expected",
        [("CORRECT", 1.0), ("PARTIAL", 0.5), ("INCORRECT", 0.0)],
    )
    def test_correctness_grades_map_to_scores(self, verdict, expected):
        score, grade = judge_correctness(_model(verdict), "q", "ref", "ans")
        assert score == expected
        assert grade == verdict

    def test_empty_answer_scores_zero_without_calling_the_judge(self):
        model = _model("CORRECT")
        score, grade = judge_correctness(model, "q", "ref", "   ")
        assert score == 0.0
        assert grade == "EMPTY"
        model.invoke.assert_not_called()

    def test_unparseable_grade_is_nan_not_silently_zero(self):
        """A judge failure must not look like a wrong answer in the mean."""
        score, grade = judge_correctness(_model("I am not sure"), "q", "ref", "ans")
        assert score != score  # NaN
        assert grade.startswith("UNPARSED")

    def test_judge_exception_is_nan_not_zero(self):
        model = MagicMock()
        model.invoke.side_effect = RuntimeError("provider down")
        score, grade = judge_correctness(model, "q", "ref", "ans")
        assert score != score
        assert grade == "JUDGE_ERROR"

    def test_spend_cap_propagates_through_the_judge(self):
        model = MagicMock()
        model.invoke.side_effect = SpendCapExceeded("cap")
        with pytest.raises(SpendCapExceeded):
            judge_correctness(model, "q", "ref", "ans")

    def test_grounding_counts_supported_and_unsupported_claims(self):
        out = judge_grounding(
            _model("SUPPORTED | claim one\nUNSUPPORTED | claim two\nSUPPORTED | claim three"),
            "context",
            "answer",
        )
        assert out == {"claims": 3, "supported": 2, "unsupported": 1, "status": "ok"}

    def test_grounding_with_no_context_is_marked_not_counted_as_clean(self):
        """Zero claims out of zero would otherwise flatter an ungrounded answer."""
        out = judge_grounding(_model("SUPPORTED | x"), "", "answer")
        assert out["status"] == "no_context"
        assert out["claims"] == 0

    def test_grounding_ignores_chatter_around_the_verdict_lines(self):
        out = judge_grounding(
            _model("Here are the claims:\nSUPPORTED | a real claim\nThat is all."),
            "context",
            "answer",
        )
        assert out["claims"] == 1


# -- Statistics -------------------------------------------------------------


class TestStatistics:
    def test_percentile_picks_the_expected_values(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert percentile(values, 0.5) == 6
        assert percentile(values, 0.95) == 10
        assert percentile([], 0.5) is None

    def test_bootstrap_ci_brackets_the_mean(self):
        ci = bootstrap_ci([1.0, 1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.5])
        assert ci["lo"] <= ci["mean"] <= ci["hi"]
        assert ci["n"] == 8

    def test_bootstrap_ci_is_deterministic_for_a_given_seed(self):
        values = [1.0, 0.5, 0.0, 1.0, 0.5]
        assert bootstrap_ci(values) == bootstrap_ci(values)

    def test_bootstrap_needs_at_least_two_observations(self):
        assert bootstrap_ci([1.0]) is None
        assert bootstrap_ci([]) is None

    def test_bootstrap_drops_nan_rather_than_treating_it_as_zero(self):
        ci = bootstrap_ci([1.0, 1.0, float("nan"), 1.0])
        assert ci["mean"] == 1.0
        assert ci["n"] == 3

    def test_paired_bootstrap_detects_a_consistent_improvement(self):
        pairs = [(0.0, 1.0)] * 20
        ci = paired_bootstrap_ci(pairs)
        assert ci["mean_diff"] == 1.0
        assert ci["significant"] is True

    def test_paired_bootstrap_reports_no_difference_as_not_significant(self):
        pairs = [(1.0, 0.0), (0.0, 1.0)] * 10
        ci = paired_bootstrap_ci(pairs)
        assert ci["mean_diff"] == pytest.approx(0.0, abs=0.01)
        assert ci["significant"] is False

    def test_paired_bootstrap_sign_is_multi_minus_single(self):
        """A positive difference must mean multi scored higher."""
        ci = paired_bootstrap_ci([(0.2, 0.9)] * 15)
        assert ci["mean_diff"] > 0

    def test_paired_bootstrap_drops_incomplete_pairs(self):
        ci = paired_bootstrap_ci([(1.0, 1.0), (float("nan"), 1.0), (1.0, 1.0)])
        assert ci["n"] == 2


class TestSummarise:
    def test_rates_are_computed_over_pooled_claims(self):
        rows = [
            _row(id=1, claims=4, supported_claims=3, unsupported_claims=1),
            _row(id=2, claims=6, supported_claims=6, unsupported_claims=0),
        ]
        out = summarise(rows)
        assert out["total_claims"] == 10
        assert out["citation_support_rate"] == 0.9
        assert out["unsupported_claim_rate"] == 0.1

    def test_zero_claims_gives_no_rate_rather_than_a_perfect_one(self):
        out = summarise([_row(claims=0, supported_claims=0, unsupported_claims=0)])
        assert out["citation_support_rate"] is None
        assert out["answers_with_no_claims"] == 1

    def test_reports_latency_percentiles_and_per_query_cost(self):
        rows = [_row(id=i, latency_ms=float(i * 100)) for i in range(1, 11)]
        out = summarise(rows)
        assert out["latency_p50_ms"] == 600.0
        assert out["latency_p95_ms"] == 1000.0
        assert out["cost_per_query_usd"] == pytest.approx(0.001)

    def test_counts_errored_runs(self):
        out = summarise([_row(), _row(id=2, error="RuntimeError")])
        assert out["errors"] == 1

    def test_empty_input_summarises_to_nothing(self):
        assert summarise([]) == {}


class TestSlicing:
    def test_splits_multi_source_from_single_source(self):
        rows = [_row(id=1, multi_source=False), _row(id=2, multi_source=True)]
        assert [r["id"] for r in slice_rows(rows, True)] == [2]
        assert [r["id"] for r in slice_rows(rows, False)] == [1]


class TestEstimate:
    def test_estimate_scales_with_query_count(self):
        ten = estimate(10)["total_usd"]
        hundred = estimate(100)["total_usd"]
        assert hundred == pytest.approx(ten * 10, rel=0.01)

    def test_estimate_expects_multi_to_cost_more_than_single(self):
        projection = estimate(10)
        assert (
            projection["per_mode"]["multi"]["per_query_usd"]
            > projection["per_mode"]["single"]["per_query_usd"]
        )

    def test_estimate_records_the_prices_it_used(self):
        from evaluation.compare_modes import PRICE_INPUT_PER_M

        assert estimate(1)["prices"]["input_per_million_usd"] == PRICE_INPUT_PER_M


# -- Metering the agent -----------------------------------------------------


class TestAgentMetering:
    def test_metered_agent_counts_agent_calls_and_restores_the_accessor(self):
        """Agent tokens were silently zero because only the judge was wrapped."""
        import agent.nodes as nodes

        from evaluation.compare_modes import metered_agent

        meter = Meter(cap_usd=1.0)
        original = nodes._get_llm
        inner = _model("hi")
        nodes._get_llm = lambda: inner
        try:
            with metered_agent(meter):
                nodes._get_llm().invoke([])
                nodes._get_llm().invoke([])
            assert meter.calls == 2
            assert meter.input_tokens == 200
            # The accessor must be put back, or later work stays wrapped.
            assert nodes._get_llm is not original
            assert nodes._get_llm() is inner
        finally:
            nodes._get_llm = original

    def test_accessor_is_restored_even_when_the_run_raises(self):
        import agent.nodes as nodes

        from evaluation.compare_modes import metered_agent

        original = nodes._get_llm
        sentinel = object()
        nodes._get_llm = lambda: sentinel
        try:
            with pytest.raises(RuntimeError):
                with metered_agent(Meter(cap_usd=1.0)):
                    raise RuntimeError("run blew up")
            assert nodes._get_llm() is sentinel
        finally:
            nodes._get_llm = original


class TestSpendCapAcrossMeters:
    def test_cap_governs_agent_and_judge_spend_together(self):
        agent_meter = Meter(cap_usd=0.0001, name="agent")
        judge_meter = Meter(cap_usd=0.0001, name="judge")
        agent_meter.sibling = judge_meter
        judge_meter.sibling = agent_meter

        judge_meter.record(_response("x", tokens_in=1000, tokens_out=1000))
        # The agent spent nothing, but the shared cap is already used up.
        assert agent_meter.cost_usd == 0.0
        with pytest.raises(SpendCapExceeded):
            agent_meter.check()

    def test_reported_cost_per_query_excludes_judging(self):
        """Judging is measurement apparatus, not a cost a deployment pays."""
        agent_meter = Meter(cap_usd=1.0, name="agent")
        judge_meter = Meter(cap_usd=1.0, name="judge")
        agent_meter.sibling = judge_meter

        before = agent_meter.snapshot()
        agent_meter.record(_response("x", tokens_in=500, tokens_out=100))
        judge_meter.record(_response("x", tokens_in=9000, tokens_out=900))
        delta = agent_meter.delta(before)
        assert delta["input_tokens"] == 500
        assert delta["output_tokens"] == 100
