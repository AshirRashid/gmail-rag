# Security review

Scope: the Gmail RAG pipeline at commit `d041aa1`. Three concrete checks, not a generic checklist.

## 1. OAuth scope

`config.py:8` sets `SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]`.
Read-only, confirmed by `tests/test_security_injection.py::test_oauth_scope_is_read_only`.
The pipeline cannot modify, send, or delete anything in the connected account.

## 2. At-rest storage

`pipeline.py`'s `ingest()` upserts full cleaned email text into ChromaDB, which persists to
`./chroma-data` on local disk with no encryption layer.
This is an accepted tradeoff for a local-only, single-user tool - the alternative (encrypting
at rest) would need a key-management story this project has no user-facing need for yet.
Worth stating plainly rather than leaving implicit: anyone with filesystem access to the
machine running this tool can read the indexed email content directly from `chroma-data`.

## 3. Output injection via unsanitized snippets

`app.py` renders retrieved email snippets directly into `gr.Markdown`.
`query.search_collection()` truncates and returns the raw document text as `"snippet"` with no
HTML/Markdown escaping at any stage of the pipeline.
`tests/test_security_injection.py::test_html_payload_in_email_body_survives_unescaped_into_snippet`
confirms an `<img onerror=...>` / `javascript:` payload embedded in an email body survives intact
into the value that reaches `gr.Markdown`.

This is the applicable security surface at this commit - there is no LLM generation step, so
classic prompt-injection-into-generation doesn't apply here. The risk is UI-level: if Gradio's
Markdown renderer executes injected HTML/JS (not verified here - this test stops at confirming
the payload reaches the render boundary unsanitized, not at confirming exploitation in a live
browser), a crafted email could affect the local UI session. Given this is a local, single-user
tool, blast radius is limited to the operator's own machine - but it's a real, documented gap,
not a theoretical one. Recommended follow-up (out of scope for this evidence pack): escape
snippet content before rendering, or switch the result display from `gr.Markdown` to a plain
`gr.Textbox`.
