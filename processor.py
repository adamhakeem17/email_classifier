"""
processor.py - ties together the classifier and responder into one pipeline
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from langchain_community.llms import Ollama

from classifier import EmailClassifier
from responder import EmailResponder


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EmailResult:
    # input
    raw_email: str
    email_preview: str

    # classification stuff
    category: str
    urgency: str
    sentiment: str
    churn_probability: float
    confidence: float
    language: str
    key_issue: str
    customer_tone: str
    needs_human_review: bool
    parse_error: bool

    # reply
    draft_reply: str

    # meta
    processed_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    processing_ms: int = 0

    def to_dict(self):
        return {
            "email_preview": self.email_preview,
            "category": self.category,
            "urgency": self.urgency,
            "sentiment": self.sentiment,
            "churn_probability": round(self.churn_probability, 3),
            "confidence": round(self.confidence, 3),
            "language": self.language,
            "key_issue": self.key_issue,
            "customer_tone": self.customer_tone,
            "needs_human_review": self.needs_human_review,
            "draft_reply": self.draft_reply,
            "processed_at": self.processed_at,
            "processing_ms": self.processing_ms,
            "parse_error": self.parse_error,
        }


# ── Processor ─────────────────────────────────────────────────────────────────

class EmailProcessor:
    """
    Main class that runs classification + reply drafting.
    Use from_config() to create one easily.
    """

    def __init__(self, classifier, responder, reply_language="auto"):
        self._classifier = classifier
        self._responder = responder
        self._reply_language = reply_language

    @classmethod
    def from_config(cls, model="llama3", temperature=0.0, company_name="Acme Corp",
                    agent_name="Support Team", max_reply_words=150, reply_language="auto"):
        """Build a processor from simple config values."""
        llm = Ollama(model=model, temperature=temperature)
        classifier = EmailClassifier(llm=llm, company_name=company_name)
        responder = EmailResponder(
            llm=llm,
            company_name=company_name,
            agent_name=agent_name,
            max_words=max_reply_words,
        )
        return cls(classifier=classifier, responder=responder, reply_language=reply_language)

    def process(self, email_text):
        """Classify one email and draft a reply for it."""
        start = time.time()
        classification = self._classifier.classify(email_text)
        draft = self._responder.draft_reply(classification, self._reply_language)
        elapsed_ms = round((time.time() - start) * 1000)

        # build the preview (first 120 chars)
        if len(email_text) > 120:
            preview = email_text[:120] + "…"
        else:
            preview = email_text

        return EmailResult(
            raw_email=email_text,
            email_preview=preview,
            category=classification.category,
            urgency=classification.urgency,
            sentiment=classification.sentiment,
            churn_probability=classification.churn_probability,
            confidence=classification.confidence,
            language=classification.language,
            key_issue=classification.key_issue,
            customer_tone=classification.customer_tone,
            needs_human_review=classification.needs_human_review,
            parse_error=classification.parse_error,
            draft_reply=draft,
            processing_ms=elapsed_ms,
        )

    def process_batch(self, emails, on_progress=None):
        """
        Process a list of emails one by one.
        on_progress is an optional callback(current, total) for progress bars.
        """
        results = []
        total = len(emails)
        for i, email in enumerate(emails):
            result = self.process(email)
            results.append(result)
            if on_progress:
                on_progress(i + 1, total)
        return results

    def process_dataframe(self, df, email_column="email", on_progress=None):
        """Process all emails in a DataFrame column and return the df with new columns added."""
        if email_column not in df.columns:
            raise ValueError(f"Column '{email_column}' not found in DataFrame.")

        results = self.process_batch(
            emails=df[email_column].astype(str).tolist(),
            on_progress=on_progress,
        )
        result_df = pd.DataFrame([r.to_dict() for r in results])
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)

    def stream_batch(self, emails):
        """Yield results one at a time — handy for live-updating UIs."""
        for email in emails:
            yield self.process(email)
