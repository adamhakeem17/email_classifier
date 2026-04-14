"""
cli.py - command line interface for the email classifier

Examples:
    python cli.py --text "I was charged twice this month!"
    python cli.py --csv emails.csv --output results.csv
    python cli.py --csv emails.csv --model mistral --company "TechCorp"
"""

import argparse
import json
import sys

import pandas as pd

from exporter import save_analytics_pdf, save_csv
from processor import EmailProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Email Classifier & Auto-Responder",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Single email text to classify")
    group.add_argument("--csv", help="Path to CSV file with 'email' column")

    parser.add_argument("--email-column", default="email", help="CSV column name (default: email)")
    parser.add_argument("--output", default=None, help="Output CSV path for batch results")
    parser.add_argument("--model", default="llama3", help="Ollama model (default: llama3)")
    parser.add_argument("--company", default="Acme Corp", help="Company name for replies")
    parser.add_argument("--agent", default="Support Team", help="Agent sign-off name")
    parser.add_argument("--language", default="auto", help="Reply language (default: auto)")
    parser.add_argument("--max-words", type=int, default=150, help="Max reply words")
    parser.add_argument("--pdf-report", action="store_true", help="Save analytics PDF after batch")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = EmailProcessor.from_config(
        model=args.model,
        company_name=args.company,
        agent_name=args.agent,
        reply_language=args.language,
        max_reply_words=args.max_words,
    )

    # single email mode
    if args.text or not args.csv:
        if args.text:
            email_text = args.text
        else:
            email_text = _read_stdin()

        print(f"\nClassifying email...\n{'─' * 50}")
        result = processor.process(email_text)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Category:     {result.category.replace('_', ' ').title()}")
            print(f"Urgency:      {result.urgency.upper()}")
            print(f"Sentiment:    {result.sentiment.replace('_', ' ')}")
            print(f"Churn risk:   {result.churn_probability:.0%}")
            print(f"Confidence:   {result.confidence:.0%}")
            print(f"Language:     {result.language}")
            print(f"Key issue:    {result.key_issue}")
            print(f"Needs review: {result.needs_human_review}")
            print(f"\n{'─' * 50}")
            print("DRAFT REPLY:\n")
            print(result.draft_reply)
        return

    # batch CSV mode
    print(f"\nBatch mode: {args.csv}")
    df_in = pd.read_csv(args.csv)

    if args.email_column not in df_in.columns:
        print(f"Error: column '{args.email_column}' not found. Available: {list(df_in.columns)}", file=sys.stderr)
        sys.exit(1)

    total = len(df_in)
    print(f"{'─' * 50}")
    print(f"Emails:  {total}")
    print(f"Model:   {args.model}")
    print(f"Company: {args.company}")
    print(f"{'─' * 50}\n")

    results = []
    for i, email in enumerate(df_in[args.email_column].astype(str), 1):
        print(f"[{i:>4}/{total}] Processing...", end="\r")
        results.append(processor.process(email))

    print(f"\nDone! Classified {total} emails.")

    output_path = save_csv(results, path=args.output)
    print(f"Saved results: {output_path}")

    if args.pdf_report:
        pdf_path = save_analytics_pdf(results, company_name=args.company)
        print(f"Saved analytics PDF: {pdf_path}")

    # quick summary
    needs_review = sum(1 for r in results if r.needs_human_review)
    high_churn = sum(1 for r in results if r.churn_probability > 0.65)
    print(f"\nSummary:")
    print(f"  Needs human review: {needs_review}/{total}  ({needs_review/total:.0%})")
    print(f"  High churn risk:    {high_churn}/{total}")


def _read_stdin() -> str:
    print("Paste email text (Ctrl+D when done):\n")
    return sys.stdin.read().strip()


if __name__ == "__main__":
    main()
