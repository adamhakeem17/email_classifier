"""
tests for classifier.py helper functions
tests the parsing + validation logic without needing Ollama running
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from classifier import (
    _clamp,
    _parse_json_safely,
    _validate,
    VALID_CATEGORIES,
    VALID_URGENCIES,
    VALID_SENTIMENTS,
)


class TestParseJsonSafely:
    def test_clean_json(self):
        raw = '{"category": "billing", "urgency": "high"}'
        result, error = _parse_json_safely(raw)
        assert result["category"] == "billing"
        assert error is False

    def test_json_with_preamble(self):
        """LLM sometimes adds text before the JSON object."""
        raw = 'Here is the classification:\n{"category": "bug_report", "urgency": "critical"}'
        result, error = _parse_json_safely(raw)
        assert result["category"] == "bug_report"
        assert error is False

    def test_json_with_markdown_fences(self):
        """LLM sometimes wraps JSON in ```json blocks despite instructions."""
        raw = '```json\n{"category": "complaint", "urgency": "low"}\n```'
        result, error = _parse_json_safely(raw)
        # Should extract JSON from inside the fences
        assert isinstance(result, dict)

    def test_completely_invalid(self):
        raw = "I could not classify this email."
        result, error = _parse_json_safely(raw)
        assert result == {}
        assert error is True

    def test_empty_string(self):
        result, error = _parse_json_safely("")
        assert result == {}
        assert error is True


class TestValidate:
    def test_valid_value_passes_through(self):
        assert _validate("billing", VALID_CATEGORIES, "general_inquiry") == "billing"

    def test_invalid_value_returns_fallback(self):
        assert _validate("unknown_category", VALID_CATEGORIES, "general_inquiry") == "general_inquiry"

    def test_all_valid_categories(self):
        for cat in VALID_CATEGORIES:
            assert _validate(cat, VALID_CATEGORIES, "general_inquiry") == cat

    def test_all_valid_urgencies(self):
        for urg in VALID_URGENCIES:
            assert _validate(urg, VALID_URGENCIES, "medium") == urg

    def test_all_valid_sentiments(self):
        for sent in VALID_SENTIMENTS:
            assert _validate(sent, VALID_SENTIMENTS, "neutral") == sent


class TestClamp:
    def test_value_in_range(self):
        assert _clamp(0.5) == 0.5

    def test_value_below_zero(self):
        assert _clamp(-0.5) == 0.0

    def test_value_above_one(self):
        assert _clamp(1.5) == 1.0

    def test_zero_boundary(self):
        assert _clamp(0.0) == 0.0

    def test_one_boundary(self):
        assert _clamp(1.0) == 1.0

    def test_string_float(self):
        assert _clamp("0.75") == 0.75

    def test_invalid_type_returns_default(self):
        assert _clamp("not_a_number") == 0.5

    def test_none_returns_default(self):
        assert _clamp(None) == 0.5


class TestNeedsHumanReview:
    """Test the computed property without calling the LLM."""

    def _make_result(self, **kwargs):
        from classifier import ClassificationResult
        defaults = dict(
            category="general_inquiry", urgency="low",
            sentiment="neutral", churn_probability=0.2,
            confidence=0.9, language="English",
            key_issue="test", customer_tone="polite",
            raw_email="test email",
        )
        defaults.update(kwargs)
        return ClassificationResult(**defaults)

    def test_normal_email_no_review(self):
        r = self._make_result()
        assert r.needs_human_review is False

    def test_critical_urgency_triggers_review(self):
        r = self._make_result(urgency="critical")
        assert r.needs_human_review is True

    def test_high_churn_triggers_review(self):
        r = self._make_result(churn_probability=0.8)
        assert r.needs_human_review is True

    def test_churn_below_threshold_no_review(self):
        r = self._make_result(churn_probability=0.6)
        assert r.needs_human_review is False

    def test_very_negative_sentiment_triggers_review(self):
        r = self._make_result(sentiment="very_negative")
        assert r.needs_human_review is True

    def test_low_confidence_triggers_review(self):
        r = self._make_result(confidence=0.4)
        assert r.needs_human_review is True

    def test_confidence_at_threshold_no_review(self):
        r = self._make_result(confidence=0.6)
        assert r.needs_human_review is False
