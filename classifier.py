"""
classifier.py - handles email classification with Ollama LLM
"""

import json
import re
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


# allowed values for each field
VALID_CATEGORIES = {
    "billing", "bug_report", "feature_request", "churn_risk",
    "general_inquiry", "complaint", "praise", "refund_request",
}
VALID_URGENCIES = {"low", "medium", "high", "critical"}
VALID_SENTIMENTS = {"very_positive", "positive", "neutral", "negative", "very_negative"}
VALID_TONES = {"polite", "frustrated", "angry", "urgent", "casual"}


@dataclass
class ClassificationResult:
    category: str
    urgency: str
    sentiment: str
    churn_probability: float
    confidence: float
    language: str
    key_issue: str
    customer_tone: str
    raw_email: str
    parse_error: bool = False

    @property
    def needs_human_review(self):
        """Flag emails that probably need a real person to look at them."""
        if self.urgency == "critical":
            return True
        if self.churn_probability > 0.65:
            return True
        if self.sentiment == "very_negative":
            return True
        if self.confidence < 0.6:
            return True
        return False

    def to_dict(self):
        return {
            "category": self.category,
            "urgency": self.urgency,
            "sentiment": self.sentiment,
            "churn_probability": self.churn_probability,
            "confidence": self.confidence,
            "language": self.language,
            "key_issue": self.key_issue,
            "customer_tone": self.customer_tone,
            "needs_human_review": self.needs_human_review,
            "parse_error": self.parse_error,
        }


# the prompt we send to the LLM
CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["email", "company"],
    template="""You are an expert email classification AI for {company}.

Analyse this customer email carefully:
---
{email}
---

Return ONLY valid JSON — no markdown fences, no explanation, nothing else.
Use exactly these field names and value options:

{{
  "category":          "<billing|bug_report|feature_request|churn_risk|general_inquiry|complaint|praise|refund_request>",
  "urgency":           "<low|medium|high|critical>",
  "sentiment":         "<very_positive|positive|neutral|negative|very_negative>",
  "churn_probability": <float 0.0 to 1.0>,
  "confidence":        <float 0.0 to 1.0>,
  "language":          "<detected language e.g. English, Bahasa Indonesia, Mandarin>",
  "key_issue":         "<one sentence: the core problem or request>",
  "customer_tone":     "<polite|frustrated|angry|urgent|casual>"
}}""",
)


class EmailClassifier:
    """Wraps the Ollama LLM + classification prompt."""

    def __init__(self, llm, company_name="Acme Corp"):
        self.company_name = company_name
        self._chain = CLASSIFY_PROMPT | llm  # langchain pipe syntax

    def classify(self, email_text):
        """Classify a single email. Always returns a ClassificationResult even if parsing fails."""
        raw = self._chain.invoke({"email": email_text, "company": self.company_name})
        parsed, parse_error = _parse_json_safely(raw)

        return ClassificationResult(
            category=_validate(parsed.get("category", "general_inquiry"), VALID_CATEGORIES, "general_inquiry"),
            urgency=_validate(parsed.get("urgency", "medium"), VALID_URGENCIES, "medium"),
            sentiment=_validate(parsed.get("sentiment", "neutral"), VALID_SENTIMENTS, "neutral"),
            churn_probability=_clamp(parsed.get("churn_probability", 0.3)),
            confidence=_clamp(parsed.get("confidence", 0.5)),
            language=parsed.get("language", "English"),
            key_issue=parsed.get("key_issue", "Unable to determine"),
            customer_tone=_validate(parsed.get("customer_tone", "casual"), VALID_TONES, "casual"),
            raw_email=email_text,
            parse_error=parse_error,
        )


# --- helper functions ---

def _parse_json_safely(text):
    """Try to pull a JSON object out of whatever the LLM gave us."""
    text = text.strip()

    # best case: the LLM just returned clean JSON
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    # sometimes there's extra text around the JSON, try to grab just the {...} part
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), False
        except json.JSONDecodeError:
            pass

    # give up
    return {}, True


def _validate(value, valid_set, fallback):
    """Return the value if it's in the allowed set, otherwise use the fallback."""
    if value in valid_set:
        return value
    return fallback


def _clamp(value, lo=0.0, hi=1.0):
    """Clamp a number between lo and hi. If it's not a number, return 0.5."""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 0.5
