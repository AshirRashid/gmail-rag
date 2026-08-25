"""Deterministic synthetic email corpus + query ground truth for the retrieval benchmark."""
import itertools
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

SENDERS = {
    "events": ["parties@evite-clone.com", "alumni@nyuad-network.org", "team-social@acme-labs.io"],
    "deadlines": ["research-office@university.edu", "no-reply@conference-system.org", "hr@acme-labs.io"],
    "newsletters": ["digest@techweekly.com", "updates@fintech-brief.io", "news@opensource-monthly.org"],
    "receipts": ["receipts@cloudhost.com", "billing@streamplus.io", "orders@gearshop.com"],
}

EVENT_TEMPLATES = [
    ("You're invited: {kind} on {date}", "RSVP by {date} - {kind} kicks off at {time} at {venue}. Bring a friend."),
    ("Save the date - {kind} at {venue}", "We're hosting {kind} on {date} at {venue}. Details and RSVP link inside."),
    ("Reminder: {kind} starts {time}", "Just a reminder that {kind} starts at {time} on {date}. See you there."),
    ("Join us for {kind}", "{kind} is happening {date} at {venue}, starting {time}. Hope to see you."),
]
EVENT_FILL = [
    {"kind": "the team dinner", "date": "Friday", "time": "7pm", "venue": "The Local Bistro"},
    {"kind": "a dentist appointment", "date": "Tuesday", "time": "3pm", "venue": "Downtown Dental"},
    {"kind": "the alumni mixer", "date": "next Thursday", "time": "6:30pm", "venue": "The Rooftop"},
    {"kind": "a birthday party", "date": "Saturday", "time": "noon", "venue": "Riverside Park"},
]

DEADLINE_TEMPLATES = [
    ("Re: Grant renewal - documents due {date}", "Reminder that the renewal packet, including the budget justification, is due before end of day {date}."),
    ("Your submission window closes {date}", "This is a reminder that the camera-ready deadline for accepted papers is {date}. Please upload your final PDF."),
    ("Action required: onboarding paperwork due {date}", "Please complete and return the onboarding forms by {date} so payroll can be set up on time."),
    ("Lease renewal - sign by {date}", "Please return the signed renewal by {date} so we can lock in the current rate before it expires."),
]
DEADLINE_FILL = [
    {"date": "this Friday"}, {"date": "next Monday"}, {"date": "the 30th"}, {"date": "end of week"},
]

NEWSLETTER_TEMPLATES = [
    ("This week in {topic}", "Top stories in {topic} this week: a roundup of what mattered and why."),
    ("{topic} digest - issue #{n}", "Your {topic} digest for this week. Curated links and short takes below."),
    ("The {topic} roundup", "Five things worth knowing in {topic} this week, summarized."),
    ("{topic} weekly", "Here's what happened in {topic} this week, in five minutes or less."),
]
NEWSLETTER_FILL = [
    {"topic": "AI research", "n": 42}, {"topic": "fintech", "n": 17},
    {"topic": "open source", "n": 88}, {"topic": "climate tech", "n": 9},
]

RECEIPT_TEMPLATES = [
    ("Your receipt for {item}", "Thanks for your purchase. {item} - total {amount}, charged to your card ending in 4412."),
    ("Payment confirmation - {item}", "We've received your payment of {amount} for {item}. This receipt is for your records."),
    ("Order confirmed: {item}", "Your order for {item} ({amount}) has been confirmed and will ship shortly."),
    ("Subscription renewed - {item}", "Your subscription to {item} renewed for {amount}. Manage your plan anytime."),
]
RECEIPT_FILL = [
    {"item": "cloud storage plan", "amount": "$9.99"}, {"item": "wireless keyboard", "amount": "$64.00"},
    {"item": "streaming subscription", "amount": "$12.99"}, {"item": "monitor stand", "amount": "$38.50"},
]


def _build_category(category: str, templates: list[tuple[str, str]], fills: list[dict]) -> list[dict]:
    senders = SENDERS[category]
    emails = []
    # Take 12 combinations: 4 templates with 3 fills each
    # Rotate which fill is skipped per template so all fills are represented
    selected_combos = []
    for t_idx, (subject_t, body_t) in enumerate(templates):
        skip_idx = t_idx  # Skip a different fill per template to ensure all appear
        for f_idx in range(len(fills)):
            if f_idx == skip_idx:
                continue
            fill = fills[f_idx]
            i = t_idx * 3 + (f_idx if f_idx < skip_idx else f_idx - 1)
            sender = senders[i % len(senders)]
            selected_combos.append(((subject_t, body_t), fill, sender))

    for i, ((subject_t, body_t), fill, sender) in enumerate(selected_combos, start=1):
        eid = f"{category}-{i:03d}"
        emails.append({
            "id": eid,
            "thread_id": f"t-{eid}",
            "subject": subject_t.format(**fill),
            "sender": sender,
            "date": "Mon, 1 Jan 2026",
            "body": body_t.format(**fill),
            "is_reply": False,
        })
    return emails


def _build_reply_pairs() -> list[dict]:
    emails = []
    for i in range(1, 7):
        orig_id, reply_id = f"thread-orig-{i:03d}", f"thread-reply-{i:03d}"
        emails.append({
            "id": orig_id, "thread_id": f"thread-{i:03d}",
            "subject": f"Project update #{i}",
            "sender": "manager@acme-labs.io", "date": "Mon, 1 Jan 2026",
            "body": f"Here's the status on project {i}: on track, no blockers, next check-in Friday.",
            "is_reply": False,
        })
        emails.append({
            "id": reply_id, "thread_id": f"thread-{i:03d}",
            "subject": f"Re: Project update #{i}",
            "sender": "you@acme-labs.io", "date": "Tue, 2 Jan 2026",
            "body": f"> Here's the status on project {i}: on track...\n\nThanks, sounds good.",
            "is_reply": True,
        })
    return emails


def _find_id(emails: list[dict], subject_contains: str) -> str:
    for e in emails:
        if subject_contains in e["subject"]:
            return e["id"]
    raise ValueError(f"no email found with subject containing {subject_contains!r}")


def build_corpus() -> list[dict]:
    emails = []
    emails += _build_category("events", EVENT_TEMPLATES, EVENT_FILL)
    emails += _build_category("deadlines", DEADLINE_TEMPLATES, DEADLINE_FILL)
    emails += _build_category("newsletters", NEWSLETTER_TEMPLATES, NEWSLETTER_FILL)
    emails += _build_category("receipts", RECEIPT_TEMPLATES, RECEIPT_FILL)
    emails += _build_reply_pairs()
    return emails


def build_queries(emails: list[dict]) -> list[dict]:
    by_category = {
        cat: [e["id"] for e in emails if e["id"].startswith(cat) and not e["is_reply"]]
        for cat in ("events", "deadlines", "newsletters", "receipts")
    }
    queries = [
        {"query": "an email inviting me to an event, party, or gathering with a specific date", "relevant_ids": by_category["events"]},
        {"query": "an email about an upcoming deadline, due date, or submission cutoff", "relevant_ids": by_category["deadlines"]},
        {"query": "a newsletter or weekly digest roundup", "relevant_ids": by_category["newsletters"]},
        {"query": "a receipt or payment confirmation for something I bought", "relevant_ids": by_category["receipts"]},
        {"query": "an email confirming a dentist or doctor appointment",
         "relevant_ids": [_find_id(emails, "dentist appointment")]},
        {"query": "an email about a lease or rental agreement renewal",
         "relevant_ids": [_find_id(emails, "Lease renewal")]},
        {"query": "an email about a grant or research funding deadline",
         "relevant_ids": [_find_id(emails, "Grant renewal")]},
        {"query": "a reminder about onboarding paperwork for a new job",
         "relevant_ids": [_find_id(emails, "onboarding paperwork")]},
        {"query": "an email about AI or machine learning research news",
         "relevant_ids": [e["id"] for e in emails if "AI research" in e["body"]]},
        {"query": "an email about a subscription renewal charge",
         "relevant_ids": [e["id"] for e in emails if "Subscription renewed" in e["subject"]]},
        {"query": "a birthday party invitation",
         "relevant_ids": [_find_id(emails, "birthday party")]},
        {"query": "a conference paper camera-ready submission reminder",
         "relevant_ids": [_find_id(emails, "submission window")]},
        {"query": "an email about a team dinner or social event",
         "relevant_ids": [_find_id(emails, "team dinner")]},
        {"query": "a purchase confirmation for office equipment",
         "relevant_ids": [_find_id(emails, "wireless keyboard")]},
        {"query": "an alumni network event invitation",
         "relevant_ids": [_find_id(emails, "alumni mixer")]},
        {"query": "a fintech industry news roundup",
         "relevant_ids": [e["id"] for e in emails if "fintech" in e["body"]]},
    ]
    return queries


if __name__ == "__main__":
    emails = build_corpus()
    queries = build_queries(emails)
    (OUT_DIR / "emails.json").write_text(json.dumps(emails, indent=2))
    (OUT_DIR / "queries.json").write_text(json.dumps(queries, indent=2))
    print(f"Wrote {len(emails)} emails and {len(queries)} queries to {OUT_DIR}")
