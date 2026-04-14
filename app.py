"""
app.py - Streamlit UI for the email classifier
"""

import streamlit as st
import pandas as pd

from analytics import (
    AnalyticsSummary,
    chart_category_pie,
    chart_churn_vs_confidence,
    chart_sentiment_distribution,
    chart_urgency_by_category,
    high_churn_table,
)
from exporter import to_analytics_pdf_bytes, to_csv_bytes
from processor import EmailProcessor, EmailResult

# page setup
st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="📧",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background: #0d0d14; color: #e0e0f0; }
    .email-card {
        background: #1a1a26; border: 1px solid #2a2a40;
        border-radius: 8px; padding: 20px;
    }
    .tag {
        display: inline-block; font-size: 11px; padding: 3px 10px;
        border-radius: 4px; margin: 2px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# colour maps for the tags
URGENCY_COLOUR = {"critical": "#fa6d6d", "high": "#fad96d", "medium": "#7c6dfa", "low": "#6dfabd"}
CATEGORY_COLOUR = {
    "billing": "#fa6d6d", "bug_report": "#fad96d", "churn_risk": "#fa6d6d",
    "complaint": "#fad96d", "feature_request": "#7c6dfa",
    "general_inquiry": "#6dfabd", "praise": "#6dfabd", "refund_request": "#fa6d6d",
}
SENTIMENT_COLOUR = {
    "very_positive": "#6dfabd", "positive": "#adfadb", "neutral": "#a0a0c0",
    "negative": "#fad96d", "very_negative": "#fa6d6d",
}

# keep track of results across reruns
if "results" not in st.session_state:
    st.session_state.results = []

# sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    model          = st.selectbox("Ollama Model",   ["llama3", "mistral", "phi3"])
    company_name   = st.text_input("Company Name",  "Acme Corp")
    agent_name     = st.text_input("Agent Name",    "Support Team")
    reply_language = st.selectbox(
        "Reply Language",
        ["auto", "English", "Bahasa Indonesia", "Mandarin", "Thai", "Vietnamese"],
    )
    max_words = st.slider("Max Reply Words", 50, 300, 150, 25)

    st.divider()
    st.header("📊 Session Stats")
    if st.session_state.results:
        summary = AnalyticsSummary.from_results(st.session_state.results)
        st.metric("Processed",    summary.total)
        st.metric("Need Review",  summary.needs_review)
        st.metric("High Churn",   summary.high_churn)
        st.metric("Auto-Resolved", f"{summary.auto_resolved_pct:.0f}%")

    if st.session_state.results and st.button("🗑️ Clear All"):
        st.session_state.results = []
        st.rerun()


# cache the processor so we don't rebuild it every time streamlit reruns
@st.cache_resource(show_spinner="Loading AI model…")
def get_processor(model_: str, company_: str, agent_: str, lang_: str, words_: int):
    return EmailProcessor.from_config(
        model=model_,
        company_name=company_,
        agent_name=agent_,
        reply_language=lang_,
        max_reply_words=words_,
    )


# header
st.title("📧 AI Email Classifier & Auto-Responder")
st.caption(
    "Classify emails by intent, urgency & churn risk · "
    "Auto-draft tailored replies · Batch-process CSVs · Analytics dashboard"
)

tab_single, tab_batch, tab_analytics = st.tabs(
    ["📧 Single Email", "📁 Batch CSV", "📊 Analytics"]
)

# ---- TAB 1: Single email ----
with tab_single:
    SAMPLES = {
        "Angry billing complaint": (
            "Subject: Charged TWICE this month — completely unacceptable!\n\n"
            "I have been a loyal customer for 3 years and you have just charged me $199 TWICE "
            "this month. Nobody is responding to my emails. If this isn't fixed TODAY I am "
            "cancelling and disputing with my bank. This is disgraceful."
        ),
        "Polite feature request": (
            "Hi team, loving the product overall! One quick request — it would be really "
            "helpful if we could export reports as PDF directly. Currently we screenshot "
            "everything. Thanks for all the great work!"
        ),
        "Churn risk": (
            "I've been evaluating whether to renew our contract next month. Honestly, "
            "the price has gone up and I'm not sure we're getting the same value anymore. "
            "Our team hasn't been using it as much. Just flagging this before we decide."
        ),
        "Critical technical bug": (
            "Hi, getting a 500 error every single time I try to export data. This started "
            "after yesterday's update. I have a client presentation in 90 minutes and I "
            "desperately need this working. Please help ASAP."
        ),
        "Happy customer / praise": (
            "Just wanted to say — your support team was absolutely incredible yesterday. "
            "The issue was resolved in under 10 minutes. Keep up the fantastic work! "
            "Will definitely be recommending you to colleagues."
        ),
    }

    col_in, col_out = st.columns([1, 1])

    with col_in:
        sample_key = st.selectbox("Load a sample", ["— write your own —"] + list(SAMPLES))
        email_text = st.text_area(
            "Email content",
            value=SAMPLES.get(sample_key, ""),
            height=240,
            placeholder="Paste a customer email here…",
        )
        classify_btn = st.button("🔍 Classify & Draft Reply", type="primary")

    with col_out:
        if classify_btn and email_text.strip():
            processor = get_processor(model, company_name, agent_name, reply_language, max_words)
            with st.spinner("Classifying…"):
                result = processor.process(email_text)
            st.session_state.results.insert(0, result)

        if st.session_state.results:
            r = st.session_state.results[0]

            # Colour-coded tag row
            uc = URGENCY_COLOUR.get(r.urgency, "#7c6dfa")
            cc = CATEGORY_COLOUR.get(r.category, "#7c6dfa")
            sc = SENTIMENT_COLOUR.get(r.sentiment, "#a0a0c0")

            st.markdown(
                f'<div class="email-card" style="border-left:4px solid {uc}">'
                f'<span class="tag" style="background:{cc}22;color:{cc};border:1px solid {cc}55">'
                f'{r.category.replace("_"," ").title()}</span>'
                f'<span class="tag" style="background:{uc}22;color:{uc};border:1px solid {uc}55">'
                f'⚡ {r.urgency.upper()}</span>'
                f'<span class="tag" style="background:{sc}22;color:{sc};border:1px solid {sc}55">'
                f'{r.sentiment.replace("_"," ")}</span>'
                f'<span class="tag" style="background:#2a2a4033;color:#a0a0c0;border:1px solid #3a3a60">'
                f'🌐 {r.language}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Risk",  f"{r.churn_probability:.0%}")
            m2.metric("Confidence",  f"{r.confidence:.0%}")
            m3.metric("Tone",        r.customer_tone.title())

            if r.needs_human_review:
                st.error("⚠️ FLAG: Requires human review")
            if r.parse_error:
                st.warning("⚠️ JSON parse error — classification may be approximate")

            st.info(f"**Core issue:** {r.key_issue}")

            st.subheader("✉️ Draft Reply")
            edited = st.text_area("Review and edit before sending", r.draft_reply, height=200)
            st.caption(f"⏱ Processed in {r.processing_ms}ms")

# ---- TAB 2: Batch CSV ----
with tab_batch:
    st.subheader("📁 Batch Classification")
    st.caption("Upload a CSV with an `email` column. All rows are classified and a results CSV is generated.")

    # Generate demo CSV
    if st.button("📋 Generate Demo CSV"):
        demo_df = pd.DataFrame({
            "email_id": range(1, len(SAMPLES) + 1),
            "email":    list(SAMPLES.values()),
        })
        st.download_button(
            "⬇️ Download demo_emails.csv",
            data=demo_df.to_csv(index=False).encode("utf-8"),
            file_name="demo_emails.csv",
            mime="text/csv",
        )

    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        df_in = pd.read_csv(csv_file)

        if "email" not in df_in.columns:
            st.error("CSV must contain an `email` column.")
        else:
            st.dataframe(df_in.head(5), use_container_width=True)
            st.caption(f"{len(df_in)} emails ready to process.")

            if st.button(f"🚀 Classify All {len(df_in)} Emails", type="primary"):
                processor  = get_processor(model, company_name, agent_name, reply_language, max_words)
                progress   = st.progress(0, text="Starting…")
                batch_res: list[EmailResult] = []

                def _on_progress(current: int, total: int) -> None:
                    progress.progress(current / total, text=f"Processing {current}/{total}…")

                batch_res = processor.process_batch(
                    df_in["email"].astype(str).tolist(),
                    on_progress=_on_progress,
                )
                progress.progress(1.0, text="✅ Done!")
                st.session_state.results.extend(batch_res)

                # Show results table
                result_df = df_in.copy()
                result_df = pd.concat(
                    [result_df.reset_index(drop=True),
                     pd.DataFrame([r.to_dict() for r in batch_res])],
                    axis=1,
                )
                display_cols = [
                    c for c in
                    ["email_id", "category", "urgency", "sentiment",
                     "churn_probability", "confidence", "needs_human_review", "key_issue"]
                    if c in result_df.columns
                ]
                st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Download Results CSV",
                    data=to_csv_bytes(batch_res),
                    file_name=f"classified_{csv_file.name}",
                    mime="text/csv",
                )

# ---- TAB 3: Analytics ----
with tab_analytics:
    st.subheader("📊 Analytics Dashboard")

    if not st.session_state.results:
        st.info("Process emails in the other tabs first to see analytics here.")
    else:
        results = st.session_state.results
        summary = AnalyticsSummary.from_results(results)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Processed", summary.total)
        k2.metric("Need Review",     summary.needs_review)
        k3.metric("Avg Churn Risk",  f"{summary.avg_churn:.0%}")
        k4.metric("Auto-Resolved",   f"{summary.auto_resolved_pct:.0f}%")

        # Charts — row 1
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_category_pie(results), use_container_width=True)
        with c2:
            st.plotly_chart(chart_urgency_by_category(results), use_container_width=True)

        # Charts — row 2
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_churn_vs_confidence(results), use_container_width=True)
        with c4:
            st.plotly_chart(chart_sentiment_distribution(results), use_container_width=True)

        # High churn table
        churn_df = high_churn_table(results)
        if not churn_df.empty:
            st.divider()
            st.subheader(f"🚨 High Churn Risk ({len(churn_df)} emails)")
            st.dataframe(churn_df, use_container_width=True, hide_index=True)

        # Exports
        st.divider()
        ex1, ex2 = st.columns(2)
        ex1.download_button(
            "📥 Export All Results (CSV)",
            data=to_csv_bytes(results),
            file_name="all_classified_emails.csv",
            mime="text/csv",
        )
        ex2.download_button(
            "📄 Export Analytics Report (PDF)",
            data=to_analytics_pdf_bytes(results, company_name),
            file_name="email_analytics_report.pdf",
            mime="application/pdf",
        )
