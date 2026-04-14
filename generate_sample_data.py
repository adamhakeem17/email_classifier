"""
generate_sample_data.py
-----------------------
Generates a demo CSV of 20 realistic customer emails so you can
test the batch classification immediately without a real dataset.

Run:
    python generate_sample_data.py
"""

import os
import pandas as pd

SAMPLE_EMAILS = [
    # Billing
    "Subject: Charged twice this month!\n\nI have been charged $199 twice in November. "
    "Please refund one of these immediately. My invoice number is INV-2024-0892.",

    "Hi, I just noticed a charge on my credit card from your company that I don't recognise. "
    "Could you please clarify what this is for? Amount: $49.00 on 15 Nov.",

    # Bug reports
    "Getting a 500 Internal Server Error every time I click Export → CSV. "
    "This has been happening since yesterday's update. Using Chrome 120 on Windows 11.",

    "The mobile app crashes instantly when I open it on iOS 17.1. "
    "I've tried reinstalling three times. Completely unusable right now.",

    "Dashboard is loading extremely slowly — 30+ seconds. Was fine last week. "
    "No changes on my end. Team is complaining.",

    # Churn risk
    "We're coming up to renewal next month and I wanted to flag that we've been "
    "evaluating alternatives. The price increase this year was significant and "
    "I'm not sure the new features justify it for our team size.",

    "Our usage has dropped quite a bit since the UI redesign — the team finds "
    "it harder to navigate now. We're reconsidering whether to continue.",

    # Feature requests
    "Would it be possible to add a dark mode? We use the tool all day and "
    "it would make a big difference for eye strain. Many of my colleagues have asked for this.",

    "Feature request: bulk export. Right now we can only export one report at a time "
    "which is very time-consuming when we need to send weekly reports to 10 clients.",

    "Is there any plan to add SSO / SAML support? Our IT team requires it before "
    "we can onboard the wider organisation.",

    # Complaints
    "I am extremely frustrated. I've sent 4 support emails over the past 2 weeks "
    "and not a single response. This is completely unacceptable for a paid subscription.",

    "Your recent update completely broke our workflow. Nobody warned us this was coming "
    "and now our team has lost half a day trying to adapt. We need to be notified "
    "before changes like this are pushed.",

    # General inquiries
    "Hi, I'm evaluating your product for our team of 50 people. "
    "Do you offer volume discounts? And is there a way to trial the Enterprise plan?",

    "Can you tell me what your data retention policy is? We're in a regulated industry "
    "and need to know how long you store our data and where your servers are located.",

    "What are your support hours? And do you offer phone support or only email/chat?",

    # Refund requests
    "I signed up for the annual plan by mistake — I meant to select monthly. "
    "I only realised two days later. Is it possible to get a refund and switch to monthly?",

    "We had to cancel our account due to company restructuring. We still have 8 months "
    "left on our annual subscription. Would you be able to issue a pro-rated refund?",

    # Praise
    "Just wanted to say — your support agent Sarah was absolutely brilliant yesterday. "
    "Resolved a complex issue in under 15 minutes. Really impressed. Keep it up!",

    "The new reporting features you released last week are exactly what we needed. "
    "Our team is very happy. Thank you for listening to our feedback!",

    # Urgent / critical
    "URGENT: Our entire team is locked out of their accounts since 9am. "
    "We have a board presentation at 2pm and need access immediately. "
    "This is a critical business issue. Please escalate NOW.",
]

if __name__ == "__main__":
    os.makedirs("sample_data", exist_ok=True)
    df = pd.DataFrame({
        "email_id": range(1, len(SAMPLE_EMAILS) + 1),
        "email":    SAMPLE_EMAILS,
    })
    path = "sample_data/demo_emails.csv"
    df.to_csv(path, index=False)
    print(f"✅ Generated {path} ({len(df)} rows)")
    print(df[["email_id", "email"]].assign(
        preview=df["email"].str[:60] + "…"
    )[["email_id", "preview"]].to_string(index=False))
