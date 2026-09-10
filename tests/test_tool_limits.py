"""
tests/test_tool_limits.py — Route allowlist, sanitiser, and step-limit
enforcement. The graph has no dynamic LLM-named tool calling, so "tool
allowlisting" here means: the router can only ever produce one of three
route strings, and the conditional edge can only ever dispatch to one of
three nodes -- proven against adversarial inputs.
"""

import pytest

from agent.state import ALLOWED_ROUTES, sanitize_route


class TestSanitizeRoute:
    @pytest.mark.parametrize("value", ["retrieval", "web_search", "direct"])
    def test_exact_allowed_values_pass_through(self, value):
        assert sanitize_route(value) == value

    @pytest.mark.parametrize("value", ["  Web_Search  ", "RETRIEVAL", "route: direct", "web_search."])
    def test_messy_but_unambiguous_values_are_recovered(self, value):
        assert sanitize_route(value) in ALLOWED_ROUTES

    @pytest.mark.parametrize(
        "value",
        [
            "shell_exec",                       # hallucinated tool name
            "ignore previous instructions",     # injection text
            "retrieval and web_search",         # ambiguous: two hits
            "",
            None,
            42,
            {"route": "direct"},
            "rm -rf /",
        ],
    )
    def test_unknown_ambiguous_or_nonstring_falls_back_to_direct(self, value):
        # "direct" invokes no tool -- the safe default for anything we
        # can't confidently interpret.
        assert sanitize_route(value) == "direct"


class TestConditionalEdgeAllowlist:
    @pytest.mark.parametrize(
        "route_decision,expected_node",
        [
            ("retrieval", "retriever"),
            ("web_search", "web_search"),
            ("direct", "direct"),
            ("totally_unknown", "direct"),
            ("'; DROP TABLE", "direct"),
            (None, "direct"),
        ],
    )
    def test_route_decision_only_ever_dispatches_to_a_known_node(self, route_decision, expected_node):
        from agent.graph import _ROUTE_TO_NODE, _route_decision

        node = _route_decision({"route_decision": route_decision})
        assert node == expected_node
        assert node in set(_ROUTE_TO_NODE.values())


class TestRecursionLimit:
    def test_thread_config_sets_an_explicit_recursion_limit(self):
        from agent.memory import _RECURSION_LIMIT, get_thread_config

        cfg = get_thread_config("s1")
        assert cfg["recursion_limit"] == _RECURSION_LIMIT
        # A normal turn is 4 super-steps (security/router/tool/generator);
        # the cap must clear that but stay well under LangGraph's default 25.
        assert 4 <= _RECURSION_LIMIT < 25
