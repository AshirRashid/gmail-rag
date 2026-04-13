"""
Gmail OAuth2 client.

Handles credential refresh, token caching, MIME traversal, and email fetching.
Individual message fetch failures are skipped so one bad message doesn't abort
the whole run.
"""

import base64
import json
import os
from dataclasses import dataclass

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import CREDENTIALS_FILE, SCOPES


@dataclass
class Email:
    id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    body: str


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _load_credentials() -> Credentials | None:
    if not os.path.exists("token.json"):
        return None
    with open("token.json") as f:
        return Credentials.from_authorized_user_info(json.load(f))


def _save_credentials(creds: Credentials) -> None:
    with open("token.json", "w") as f:
        f.write(creds.to_json())


def get_credentials() -> Credentials:
    creds = _load_credentials()
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_credentials(creds)
        return creds
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    _save_credentials(creds)
    return creds


# ---------------------------------------------------------------------------
# MIME helpers
# ---------------------------------------------------------------------------

def _decode_b64url(data: str) -> str:
    # Gmail omits base64 padding; add "==" and let urlsafe_b64decode trim it
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text(payload: dict) -> str:
    """
    Recursively walk a Gmail MIME payload tree.
    Returns plain text when available, falls back to HTML → text conversion.
    """
    plain = html = None
    stack = [payload]
    while stack:
        part = stack.pop()
        # Descend into multipart containers
        if part.get("parts"):
            stack.extend(part["parts"])
            continue
        mime = (part.get("mimeType") or "").lower()
        data = part.get("body", {}).get("data")
        if not data:
            continue
        text = _decode_b64url(data)
        if "text/plain" in mime and plain is None:
            plain = text
        elif "text/html" in mime and html is None:
            html = BeautifulSoup(text, "html.parser").get_text("\n", strip=True)
    return plain or html or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_emails(n: int = 50) -> list[Email]:
    """Fetch the most recent *n* inbox emails and return them as Email objects."""
    creds = get_credentials()
    svc = build("gmail", "v1", credentials=creds)

    try:
        resp = svc.users().messages().list(
            userId="me", maxResults=n, labelIds=["INBOX"]
        ).execute()
    except HttpError as exc:
        raise RuntimeError(f"Failed to list Gmail messages: {exc}") from exc

    emails: list[Email] = []
    for ref in resp.get("messages", []):
        try:
            msg = svc.users().messages().get(
                userId="me", id=ref["id"], format="full"
            ).execute()
        except HttpError:
            # Skip individual messages that fail (e.g. permissions, deleted)
            continue

        headers = {
            h["name"].lower(): h["value"]
            for h in msg.get("payload", {}).get("headers", [])
        }
        body = _extract_text(msg.get("payload", {})) or msg.get("snippet", "")
        if not body.strip():
            continue

        emails.append(Email(
            id=ref["id"],
            thread_id=msg.get("threadId", ""),
            subject=headers.get("subject", "No Subject"),
            sender=headers.get("from", "Unknown"),
            date=headers.get("date", "Unknown"),
            body=body,
        ))

    print(f"Fetched {len(emails)} emails")
    return emails
