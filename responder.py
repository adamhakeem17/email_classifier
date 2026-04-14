"""
responder.py - generates a draft reply for a classified email
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from classifier import ClassificationResult


# prompt for generating the reply
REPLY_PROMPT = PromptTemplate(
    input_variables=[
        "email", "category", "urgency", "sentiment",
        "company", "agent_name", "language", "max_words",
    ],
    template="""You are a professional customer support agent at {company}.

Write a reply to the following {category} email.
Urgency level: {urgency}. Customer sentiment: {sentiment}.
Reply in {language}. Maximum {max_words} words.

Original email:
---
{email}
---

Writing rules:
- Start by acknowledging their specific issue (not a generic "Thank you for contacting us")
- Be empathetic and solution-focused
- Offer exactly ONE clear next step the customer can take
- If urgency is "critical" or sentiment is "very_negative", convey genuine urgency in resolving this
- If category is "churn_risk", subtly reinforce the product's value without being salesy
- If category is "praise", be warm and genuine — not corporate
- End with: "Best regards,\\n{agent_name} | {company} Support"

Reply:""",
)


class EmailResponder:
    """Generates draft replies using the classified email's metadata."""

    def __init__(self, llm, company_name="Acme Corp", agent_name="Support Team", max_words=150):
        self.company_name = company_name
        self.agent_name = agent_name
        self.max_words = max_words
        self._chain = REPLY_PROMPT | llm  # pipe syntax

    def draft_reply(self, classification, reply_language="auto"):
        """
        Generate a draft reply for a classified email.
        If reply_language is "auto", we use whatever language the classifier detected.
        """
        if reply_language == "auto":
            language = classification.language
        else:
            language = reply_language

        result = self._chain.invoke({
            "email": classification.raw_email,
            "category": classification.category.replace("_", " "),
            "urgency": classification.urgency,
            "sentiment": classification.sentiment.replace("_", " "),
            "company": self.company_name,
            "agent_name": self.agent_name,
            "language": language,
            "max_words": self.max_words,
        })
        return result.strip()
