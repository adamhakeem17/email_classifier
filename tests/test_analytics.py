"""
tests for analytics.py - summary stats and chart data
no LLM calls, uses manually built EmailResult objects
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from processor import EmailResult
from analytics import AnalyticsSummary, high_churn_table


def _make_result(**kwargs):
    """Helper to build an EmailResult with sensible defaults."""
    defaults = dict(
        raw_email="test", email_preview="test…",
        category="general_inquiry", urgency="low",
        sentiment="neutral", churn_probability=0.2,
        confidence=0.85, language="English",
        key_issue="test issue", customer_tone="polite",
        needs_human_review=False, parse_error=False,
        draft_reply="Thank you for contacting us.",
        processing_ms=500,
    )
    defaults.update(kwargs)
    return EmailResult(**defaults)


class TestAnalyticsSummary:
    def test_empty_results(self):
        s = AnalyticsSummary.from_results([])
        assert s.total == 0
        assert s.needs_review == 0
        assert s.high_churn == 0

    def test_total_count(self):
        results = [_make_result() for _ in range(5)]
        s = AnalyticsSummary.from_results(results)
        assert s.total == 5

    def test_needs_review_count(self):
        results = [
            _make_result(needs_human_review=True),
            _make_result(needs_human_review=True),
            _make_result(needs_human_review=False),
        ]
        s = AnalyticsSummary.from_results(results)
        assert s.needs_review == 2

    def test_high_churn_count(self):
        results = [
            _make_result(churn_probability=0.8),
            _make_result(churn_probability=0.9),
            _make_result(churn_probability=0.3),
        ]
        s = AnalyticsSummary.from_results(results)
        assert s.high_churn == 2

    def test_avg_churn(self):
        results = [
            _make_result(churn_probability=0.4),
            _make_result(churn_probability=0.6),
        ]
        s = AnalyticsSummary.from_results(results)
        assert s.avg_churn == pytest.approx(0.5, abs=0.01)

    def test_auto_resolved_pct(self):
        results = [
            _make_result(needs_human_review=False),
            _make_result(needs_human_review=False),
            _make_result(needs_human_review=True),
            _make_result(needs_human_review=True),
        ]
        s = AnalyticsSummary.from_results(results)
        assert s.auto_resolved_pct == pytest.approx(50.0, abs=0.1)

    def test_top_category(self):
        results = [
            _make_result(category="billing"),
            _make_result(category="billing"),
            _make_result(category="bug_report"),
        ]
        s = AnalyticsSummary.from_results(results)
        assert s.top_category == "billing"


class TestHighChurnTable:
    def test_filters_above_threshold(self):
        results = [
            _make_result(churn_probability=0.8, key_issue="wants to cancel"),
            _make_result(churn_probability=0.3, key_issue="billing question"),
            _make_result(churn_probability=0.7, key_issue="considering alternatives"),
        ]
        df = high_churn_table(results, threshold=0.65)
        assert len(df) == 2

    def test_sorted_descending(self):
        results = [
            _make_result(churn_probability=0.7),
            _make_result(churn_probability=0.9),
            _make_result(churn_probability=0.8),
        ]
        df = high_churn_table(results, threshold=0.65)
        probs = df["churn_probability"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_empty_when_no_high_churn(self):
        results = [_make_result(churn_probability=0.2) for _ in range(5)]
        df = high_churn_table(results, threshold=0.65)
        assert len(df) == 0

    def test_expected_columns(self):
        results = [_make_result(churn_probability=0.9)]
        df = high_churn_table(results)
        expected = {"email_preview", "category", "urgency", "churn_probability", "key_issue", "sentiment"}
        assert expected.issubset(set(df.columns))
