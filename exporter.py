"""
exporter.py - CSV and PDF export for classification results
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF

from analytics import AnalyticsSummary, high_churn_table
from processor import EmailResult


def to_csv_bytes(results):
    """Convert results to CSV bytes (for download buttons etc)."""
    df = pd.DataFrame([r.to_dict() for r in results])
    return df.to_csv(index=False).encode("utf-8")


def save_csv(results, path=None):
    """Save results to a CSV file. Returns the path it saved to."""
    if path is None:
        path = Path(f"exports/classified_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_csv_bytes(results))
    return path


def to_analytics_pdf_bytes(results, company_name="Acme Corp"):
    """Build a PDF analytics report and return the raw bytes."""
    summary = AnalyticsSummary.from_results(results)
    churn_table = high_churn_table(results)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, f"{company_name} — Email Analytics Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(0, 7, f"Generated: {now}  |  Emails analysed: {summary.total}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # KPI section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    kpis = [
        ("Total emails processed", str(summary.total)),
        ("Requiring human review", f"{summary.needs_review}  ({100 - summary.auto_resolved_pct:.1f}%)"),
        ("Auto-resolved", f"{summary.auto_resolved_pct:.1f}%"),
        ("High churn-risk emails", str(summary.high_churn)),
        ("Average churn probability", f"{summary.avg_churn:.0%}"),
        ("Average model confidence", f"{summary.avg_confidence:.0%}"),
        ("Most common category", summary.top_category.replace("_", " ").title()),
        ("Most common urgency", summary.top_urgency.title()),
    ]
    for label, value in kpis:
        pdf.cell(90, 7, label, border="B")
        pdf.cell(0, 7, value, border="B", ln=True)
    pdf.ln(6)

    # category breakdown
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Category Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 10)

    df_all = pd.DataFrame([r.to_dict() for r in results])
    if not df_all.empty:
        for cat, count in df_all["category"].value_counts().items():
            pct = count / summary.total * 100
            label = cat.replace("_", " ").title()
            pdf.cell(90, 7, label)
            pdf.cell(0, 7, f"{count} emails  ({pct:.1f}%)", ln=True)
    pdf.ln(6)

    # high churn risk section
    if not churn_table.empty:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "High Churn Risk Emails (>65%)", ln=True)
        pdf.set_font("Helvetica", "", 9)

        for _, row in churn_table.iterrows():
            line = (
                f"[{row['urgency'].upper()}] "
                f"{row['key_issue']}  "
                f"(churn: {row['churn_probability']:.0%})"
            )
            # fpdf doesn't handle unicode well, so replace what we can
            safe = line.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe)
            pdf.ln(1)

    return bytes(pdf.output())


def save_analytics_pdf(results, company_name="Acme Corp", path=None):
    """Save the PDF report to disk and return the path."""
    if path is None:
        path = Path(f"exports/analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_analytics_pdf_bytes(results, company_name))
    return path
