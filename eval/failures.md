# Failure analysis

Source: `eval/results/synthetic_benchmark.json`, the 5 lowest-recall queries per method (k = 5).

## Semantic pipeline

For each of the worst-performing queries below: the query text, what came back, what should
have come back, and why.

### "an email inviting me to an event, party, or gathering with a specific date"

- Recall@k: 0.333
- Returned: `['events-003', 'events-009', 'events-006', 'events-001', 'deadlines-008']`
- Expected: `['events-001', 'events-002', 'events-003', 'events-004', 'events-005', 'events-006', 'events-007', 'events-008', 'events-009', 'events-010', 'events-011', 'events-012']`
- Root cause: this query has 12 relevant emails in the corpus, but k = 5. Even a perfect
  retriever can score at most 5/12 = 0.417 recall here, so the low number is mostly a benchmark
  ceiling, not a retrieval failure. What actually happened: 4 of the 5 returned ids are correct
  (events-003, 009, 006, 001), and the fifth slot went to `deadlines-008` ("Action required:
  onboarding paperwork due next Monday") instead of a 5th event. Checked the text: `deadlines-008`
  is a date-bound reminder phrased almost like an invite ("due next Monday"), and the query itself
  asks for something "with a specific date" - so the embedding is picking up on the shared
  "date-bound reminder" framing between an event invite and a deadline reminder, not on a
  vocabulary overlap. That's a real, if minor, category-boundary miss, but it cost exactly one
  slot out of five.

### "an email about an upcoming deadline, due date, or submission cutoff"

- Recall@k: 0.417
- Returned: `['deadlines-008', 'deadlines-005', 'deadlines-006', 'deadlines-007', 'deadlines-009']`
- Expected: `['deadlines-001', 'deadlines-002', ..., 'deadlines-012']` (all 12 deadlines-* ids)
- Root cause: all 5 returned ids are correct-category (deadlines-*). Recall@5 = 5/12 = 0.417 is
  exactly the maximum achievable score given only 5 return slots against 12 relevant emails. This
  is not a retrieval failure at all - it's the k=5 ceiling from the benchmark's own construction
  (12 near-duplicate emails per category, evaluated at k=5).

### "a newsletter or weekly digest roundup"

- Recall@k: 0.417
- Returned: `['newsletters-005', 'newsletters-002', 'newsletters-001', 'newsletters-004', 'newsletters-008']`
- Expected: all 12 newsletters-* ids
- Root cause: same as above. All 5 returned ids are correct-category; this is the k=5 ceiling
  (5/12 = 0.417), not a retrieval defect.

### "a receipt or payment confirmation for something I bought"

- Recall@k: 0.417
- Returned: `['receipts-006', 'receipts-004', 'receipts-003', 'receipts-001', 'receipts-005']`
- Expected: all 12 receipts-* ids
- Root cause: same as above again - all 5 returned ids are correct-category receipts. The score
  is the k=5 ceiling, not a quality problem.

### "an email confirming a dentist or doctor appointment"

- Recall@k: 1.0
- Returned: `['events-001', 'events-008', 'events-011', 'deadlines-008', 'deadlines-009']`
- Expected: `['events-001']`
- Root cause: this is not actually a failure. `lowest_scoring` returns the 5 lowest-recall rows
  by sort order, and this query's recall is a perfect 1.0 - `events-001`, the single relevant id,
  is returned in position 1. It only appears here because, once the 4 genuinely imperfect queries
  above are accounted for, every other query in the 16-query set also scores 1.0 recall and this
  one happened to land in the top-5-lowest slice on sort order. Including it here for
  transparency rather than dropping it, since the extraction script produced it as-is: it
  demonstrates that outside the 4 broad category queries above, the semantic pipeline retrieves
  the correct single email 100% of the time in this benchmark, including for other
  near-duplicate-in-category items like `events-008` and `events-011` (also dentist-appointment
  emails) ranking right behind it.

## BM25 baseline

### "an email about a grant or research funding deadline"

- Recall@k: 0.0
- Returned: `['newsletters-010', 'newsletters-007', 'newsletters-004', 'deadlines-004', 'deadlines-005']`
- Expected: `['deadlines-001']`
- Root cause: genuine miss, not a ceiling artifact (only 1 relevant id, plenty of return slots).
  `deadlines-001` ("Re: Grant renewal - documents due next Monday" / "the renewal packet ...
  budget justification, is due before end of day next Monday") shares only the single token
  "grant" with the query - it never says "research," "funding," or "deadline" (it says "due").
  Meanwhile `newsletters-010/007/004` are all "AI research weekly/roundup/digest" newsletters
  that repeat the literal token "research" in both subject and body. BM25 is a bag-of-words
  matcher: it has no way to know that "grant renewal ... due" and "grant ... research funding
  deadline" refer to the same concept, so raw term frequency on "research" wins. This is exactly
  the kind of vocabulary-gap case semantic search is supposed to help with, and it did - the
  semantic pipeline scored recall 1.0 on this identical query.

### "a purchase confirmation for office equipment"

- Recall@k: 0.0
- Returned: `['receipts-005', 'receipts-006', 'receipts-004', 'events-011', 'events-008']`
- Expected: `['receipts-001']`
- Root cause: another genuine miss. `receipts-001` ("Your receipt for wireless keyboard" /
  "Thanks for your purchase. wireless keyboard - total $64.00...") never uses the word
  "confirmation" - its subject says "receipt." The three receipts that outrank it
  (`receipts-005/006/004`) all have the literal subject line "Payment confirmation - ...", so
  they match the query token "confirmation" directly while the actual answer does not. Same
  vocabulary-gap problem as above: the query's word choice doesn't match the correct document's
  word choice, even though a human would recognize a wireless keyboard purchase as "office
  equipment." Semantic search again scored 1.0 recall on this exact query.

### "an email inviting me to an event, party, or gathering with a specific date"

- Recall@k: 0.167
- Returned: `['events-006', 'newsletters-011', 'events-011', 'newsletters-010', 'newsletters-012']`
- Expected: all 12 events-* ids
- Root cause: checked actual token overlap between the query and each returned newsletter. The
  only overlap between the query and `newsletters-011/010/012` is the single stopword "or" - none
  of "event," "party," "gathering," or "date" appear in these newsletter bodies at all (bodies
  read like "Here's what happened in fintech this week, in five minutes or less."). The BM25
  baseline (`eval/baseline_bm25.py`) does no stopword filtering, and these newsletter bodies are
  very short (about 14 tokens). BM25's length-normalization term inflates the score for any
  matched token in a short document relative to the corpus average, so a single hit on "or" in a
  14-word newsletter outscores real topical overlap in a longer, correctly-categorized email.
  Verified this by computing raw BM25 scores directly: `newsletters-011` scores 3.14, ahead of
  most genuine events-* emails, purely off matching "or."

### "an email about an upcoming deadline, due date, or submission cutoff"

- Recall@k: 0.167
- Returned: `['newsletters-011', 'newsletters-010', 'newsletters-012', 'deadlines-001', 'deadlines-002']`
- Expected: all 12 deadlines-* ids
- Root cause: same mechanism as the previous case. The same three short newsletter emails
  (`newsletters-011/010/012`) crowd out real deadlines-* emails via a stray "or" match, boosted
  by BM25's short-document length normalization. Two real deadlines emails do make it into the
  top 5 (deadlines-001, 002), but three of five slots are wasted on newsletters that share no
  topical content with the query.

### "a receipt or payment confirmation for something I bought"

- Recall@k: 0.25
- Returned: `['receipts-005', 'receipts-006', 'receipts-004', 'newsletters-011', 'newsletters-010']`
- Expected: all 12 receipts-* ids
- Root cause: the first three returned ids are genuine receipts-* hits (all "Payment
  confirmation - ..." subjects, matching "confirmation" directly). The last two slots go to the
  same `newsletters-011/010` pair as above, again via the stray "or" match inflated by
  short-document length normalization, displacing two receipts that would otherwise have scored.

## Pattern

The data does not support the "whole-email-embedding dilution" theory - none of the semantic
pipeline's low-recall queries show a sub-topic getting lost inside a longer email. Three of its
four genuinely low scores are simple k=5-vs-12-relevant-ids ceiling effects where the pipeline
actually returned all correct-category results it had room for, and the fourth is a one-slot
miss from a real but narrow category-boundary overlap (a deadline reminder phrased with a
specific date, confused with an event invite "with a specific date"). On every query with only
1-3 relevant ids, semantic recall was a perfect 1.0, including on the two queries where BM25
failed outright (recall 0.0) due to vocabulary mismatch. BM25's failures, by contrast, cluster
around two distinct and verifiable causes: literal keyword mismatch between the query and the
correct email's actual wording (query says "confirmation," answer says "receipt"; query says
"research funding deadline," answer says "grant ... due"), and a lack of stopword filtering that
lets short, topically unrelated newsletter emails outrank longer, correct emails on a single
stray match to a word like "or," amplified by BM25's document-length normalization.
