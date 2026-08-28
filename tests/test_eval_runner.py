"""
tests/test_eval_runner.py — Unit tests for evaluation/eval_runner.py's
router accuracy computation.

The route_distribution aggregate (a raw count per route) can look correct
even when individual queries are misrouted, if two errors happen to cancel
out in the totals. These tests verify _compute_router_accuracy catches that
case by comparing route_decision to context_type per query.
"""

from evaluation.eval_runner import _compute_router_accuracy


def _result(id_, expected, actual, question="q"):
    return {"id": id_, "question": question, "context_type": expected, "route_decision": actual}


def test_all_correct_gives_100_percent():
    results = [
        _result(1, "retrieval", "retrieval"),
        _result(2, "direct", "direct"),
        _result(3, "web_search", "web_search"),
    ]

    accuracy = _compute_router_accuracy(results)

    assert accuracy["overall_accuracy"] == 1.0
    assert accuracy["correct"] == 3
    assert accuracy["total"] == 3
    assert accuracy["misrouted_queries"] == []


def test_swapped_routes_cancel_out_in_raw_distribution_but_not_accuracy():
    """This is the exact gap the accuracy computation exists to catch: a
    query expected as retrieval routed to direct, and vice versa, leaves the
    raw per-route counts unchanged (1 retrieval, 1 direct either way) while
    both queries are actually wrong."""
    results = [
        _result(1, "retrieval", "direct"),
        _result(2, "direct", "retrieval"),
    ]

    accuracy = _compute_router_accuracy(results)

    assert accuracy["overall_accuracy"] == 0.0
    assert accuracy["correct"] == 0
    assert len(accuracy["misrouted_queries"]) == 2


def test_per_route_accuracy_is_isolated_per_expected_label():
    results = [
        _result(1, "retrieval", "retrieval"),
        _result(2, "retrieval", "web_search"),
        _result(3, "direct", "direct"),
    ]

    accuracy = _compute_router_accuracy(results)

    assert accuracy["per_route_accuracy"]["retrieval"] == 0.5
    assert accuracy["per_route_accuracy"]["direct"] == 1.0


def test_confusion_matrix_tracks_expected_vs_actual():
    results = [
        _result(1, "retrieval", "retrieval"),
        _result(2, "retrieval", "web_search"),
    ]

    accuracy = _compute_router_accuracy(results)

    assert accuracy["confusion_matrix"]["retrieval"]["retrieval"] == 1
    assert accuracy["confusion_matrix"]["retrieval"]["web_search"] == 1


def test_misrouted_queries_list_includes_question_text():
    results = [_result(7, "web_search", "direct", question="what's the weather today?")]

    accuracy = _compute_router_accuracy(results)

    assert accuracy["misrouted_queries"] == [
        {"id": 7, "question": "what's the weather today?", "expected": "web_search", "actual": "direct"}
    ]


def test_empty_results_do_not_divide_by_zero():
    accuracy = _compute_router_accuracy([])

    assert accuracy["overall_accuracy"] == 0.0
    assert accuracy["total"] == 0
