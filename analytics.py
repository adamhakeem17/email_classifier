"""
analytics.py - summary stats and plotly charts from classification results
"""

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from processor import EmailResult


# colours I picked for the charts
URGENCY_COLOURS = {
    "critical": "#fa6d6d",
    "high": "#fad96d",
    "medium": "#7c6dfa",
    "low": "#6dfabd",
}
SENTIMENT_COLOURS = {
    "very_positive": "#6dfabd",
    "positive": "#adfadb",
    "neutral": "#a0a0c0",
    "negative": "#fad96d",
    "very_negative": "#fa6d6d",
}
PLOTLY_TEMPLATE = "plotly_dark"


@dataclass
class AnalyticsSummary:
    total: int
    needs_review: int
    high_churn: int
    avg_churn: float
    avg_confidence: float
    auto_resolved_pct: float
    top_category: str
    top_urgency: str

    @classmethod
    def from_results(cls, results):
        if not results:
            return cls(0, 0, 0, 0.0, 0.0, 0.0, "—", "—")
        df = _to_df(results)
        return cls(
            total=len(df),
            needs_review=int(df["needs_human_review"].sum()),
            high_churn=int((df["churn_probability"] > 0.65).sum()),
            avg_churn=round(df["churn_probability"].mean(), 3),
            avg_confidence=round(df["confidence"].mean(), 3),
            auto_resolved_pct=round((1 - df["needs_human_review"].mean()) * 100, 1),
            top_category=df["category"].mode().iloc[0] if len(df) else "—",
            top_urgency=df["urgency"].mode().iloc[0] if len(df) else "—",
        )


# chart builders

def chart_category_pie(results):
    """Pie chart showing how many emails fall into each category."""
    df = _to_df(results)
    freq = df["category"].value_counts().reset_index()
    freq.columns = ["Category", "Count"]
    return px.pie(
        freq,
        names="Category",
        values="Count",
        title="Email Category Breakdown",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )


def chart_urgency_by_category(results):
    """Stacked bar chart: urgency levels broken down by category."""
    df = _to_df(results)
    grouped = df.groupby(["category", "urgency"]).size().reset_index(name="count")
    fig = px.bar(
        grouped,
        x="category",
        y="count",
        color="urgency",
        title="Urgency by Category",
        template=PLOTLY_TEMPLATE,
        color_discrete_map=URGENCY_COLOURS,
    )
    fig.update_layout(xaxis_tickangle=-30, xaxis_title="", yaxis_title="Count")
    return fig


def chart_churn_vs_confidence(results):
    """Scatter plot: churn probability vs confidence, colored by category."""
    df = _to_df(results)
    fig = px.scatter(
        df,
        x="confidence",
        y="churn_probability",
        color="category",
        hover_data=["key_issue", "urgency", "sentiment"],
        title="Churn Risk vs Classification Confidence",
        template=PLOTLY_TEMPLATE,
    )
    # draw a horizontal line at the churn alert threshold
    fig.add_hline(
        y=0.65,
        line_dash="dash",
        line_color=URGENCY_COLOURS["critical"],
        annotation_text="Churn alert threshold (0.65)",
        annotation_position="top right",
    )
    fig.add_vline(
        x=0.6,
        line_dash="dot",
        line_color="#7c6dfa",
        annotation_text="Min confidence (0.60)",
        annotation_position="top left",
    )
    fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Churn Probability")
    return fig


def chart_sentiment_distribution(results):
    """Horizontal bar chart of sentiment counts."""
    df = _to_df(results)
    order = ["very_positive", "positive", "neutral", "negative", "very_negative"]
    freq = df["sentiment"].value_counts().reindex(order, fill_value=0).reset_index()
    freq.columns = ["Sentiment", "Count"]
    fig = px.bar(
        freq,
        x="Count",
        y="Sentiment",
        orientation="h",
        title="Sentiment Distribution",
        template=PLOTLY_TEMPLATE,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLOURS,
    )
    fig.update_layout(showlegend=False, yaxis_title="")
    return fig


def high_churn_table(results, threshold=0.65):
    """Get a table of emails with churn probability above the threshold."""
    df = _to_df(results)
    high_risk = df[df["churn_probability"] >= threshold]
    cols = ["email_preview", "category", "urgency", "churn_probability", "key_issue", "sentiment"]
    return high_risk[cols].sort_values("churn_probability", ascending=False).reset_index(drop=True)


def _to_df(results):
    """Turn a list of EmailResult into a DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])
