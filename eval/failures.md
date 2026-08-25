# Failure analysis

Source: `eval/results/synthetic_benchmark.json`, the 5 lowest-recall queries per method (k = 5).

## Semantic pipeline

For each of the worst-performing queries below: the query text, what came back, what should
have come back, and why.

### "an email inviting me to an event, party, or gathering with a specific date"

- Recall@k: 0.333
- Returned: `['events-003', 'events-009', 'events-006', 'events-001', 'deadlines-008']`
- Expected: `['events-001', 'events-002', 'events-003', 'events-004', 'events-005', 'events-006', 'events-007', 'events-008', 'events-009', 'events-010', 'events-011', 'events-012']`
- Root cause: this query has 12 relevant emails in the corpus, but k = 5.
  Even a perfect retriever can score at most 5/12 = 0.417 recall here, so the low number is mostly a benchmark ceiling, not a retrieval failure.
  What actually happened: 4 of the 5 returned ids are correct (events-003, 009, 006, 001), and the fifth slot went to `deadlines-008` ("Action required: onboarding paperwork due next Monday") instead of a 5th event.
  Checked the text: `deadlines-008` is a date-bound reminder phrased almost like an invite ("due next Monday"), and the query itself asks for something "with a specific date."
  So the embedding is likely picking up on the shared "date-bound reminder" framing between an event invite and a deadline reminder, not a vocabulary overlap.
  That is a real, if minor, category-boundary miss, but it cost exactly one slot out of five.

### "an email about an upcoming deadline, due date, or submission cutoff"

- Recall@k: 0.417
- Returned: `['deadlines-008', 'deadlines-005', 'deadlines-006', 'deadlines-007', 'deadlines-009']`
- Expected: `['deadlines-001', 'deadlines-002', 'deadlines-003', 'deadlines-004', 'deadlines-005', 'deadlines-006', 'deadlines-007', 'deadlines-008', 'deadlines-009', 'deadlines-010', 'deadlines-011', 'deadlines-012']`
- Root cause: all 5 returned ids are correct-category (deadlines-*).
  Recall@5 = 5/12 = 0.417 is exactly the maximum achievable score given only 5 return slots against 12 relevant emails.
  This is not a retrieval failure at all.
  It is the k = 5 ceiling from the benchmark's own construction (12 near-duplicate emails per category, evaluated at k = 5).

### "a newsletter or weekly digest roundup"

- Recall@k: 0.417
- Returned: `['newsletters-005', 'newsletters-002', 'newsletters-001', 'newsletters-004', 'newsletters-008']`
- Expected: `['newsletters-001', 'newsletters-002', 'newsletters-003', 'newsletters-004', 'newsletters-005', 'newsletters-006', 'newsletters-007', 'newsletters-008', 'newsletters-009', 'newsletters-010', 'newsletters-011', 'newsletters-012']`
- Root cause: same as above.
  All 5 returned ids are correct-category.
  The score is the k = 5 ceiling, not a quality problem.

### "a receipt or payment confirmation for something I bought"

- Recall@k: 0.417
- Returned: `['receipts-006', 'receipts-004', 'receipts-003', 'receipts-001', 'receipts-005']`
- Expected: `['receipts-001', 'receipts-002', 'receipts-003', 'receipts-004', 'receipts-005', 'receipts-006', 'receipts-007', 'receipts-008', 'receipts-009', 'receipts-010', 'receipts-011', 'receipts-012']`
- Root cause: same as above again.
  All 5 returned ids are correct-category receipts, including `receipts-001` ("Thanks for your purchase. wireless keyboard...") even though the corpus mixes four different receipt phrasings ("Thanks for your purchase...", "We've received your payment of $X...", "Your order for X has been confirmed...", and "Your subscription to X renewed...").
  The score is the k = 5 ceiling, not a quality problem.

### "an email confirming a dentist or doctor appointment"

- Recall@k: 1.0
- Returned: `['events-001', 'events-008', 'events-011', 'deadlines-008', 'deadlines-009']`
- Expected: `['events-001']`
- Root cause: this is not actually a failure.
  `lowest_scoring` returns the 5 lowest-recall rows by sort order, and this query's recall is a perfect 1.0. `events-001`, the single relevant id, is returned in position 1 (reciprocal rank 1.0).
  It only appears here because, once the 4 genuinely imperfect queries above are accounted for, every other query in the 16-query set also scores 1.0 recall, and this one happened to land in the top-5-lowest slice on sort order.
  It is included here for transparency rather than dropped, since the extraction script produced it as-is.
  It demonstrates that outside the 4 broad-category queries above, the semantic pipeline retrieves the correct single email 100% of the time in this benchmark, including for other near-duplicate-in-category items like `events-008` and `events-011` (also dentist-appointment emails) ranking right behind it.

## BM25 baseline

### "an email about a grant or research funding deadline"

- Recall@k: 0.0
- Returned: `['newsletters-010', 'newsletters-007', 'newsletters-004', 'deadlines-004', 'deadlines-005']`
- Expected: `['deadlines-001']`
- Root cause: a genuine miss, not a ceiling artifact (only 1 relevant id, plenty of return slots).
  `deadlines-001` ("Re: Grant renewal - documents due next Monday" / "the renewal packet ... budget justification, is due before end of day next Monday") shares only the single token "grant" with the query.
  It never says "research," "funding," or "deadline" (it says "due").
  Meanwhile `newsletters-010/007/004` are all "AI research weekly/roundup/digest" newsletters that repeat the literal token "research" in both subject and body.
  BM25 is a bag-of-words matcher: it has no way to know that "grant renewal ... due" and "grant ... research funding deadline" refer to the same concept, so raw term frequency on "research" wins.
  This is exactly the kind of vocabulary-gap case semantic search is supposed to help with, and it did.
  The semantic pipeline scored recall 1.0 on this identical query.

### "a purchase confirmation for office equipment"

- Recall@k: 0.0
- Returned: `['receipts-005', 'receipts-006', 'receipts-004', 'events-011', 'events-008']`
- Expected: `['receipts-001']`
- Root cause: another genuine miss.
  `receipts-001` ("Your receipt for wireless keyboard" / "Thanks for your purchase. wireless keyboard - total $64.00...") never uses the word "confirmation."
  Its subject says "receipt."
  The three receipts that outrank it (`receipts-005/006/004`) all have the literal subject line "Payment confirmation - ...," so they match the query token "confirmation" directly while the actual answer does not.
  Same vocabulary-gap problem as above: the query's word choice does not match the correct document's word choice, even though a human would recognize a wireless keyboard purchase as "office equipment."
  Semantic search again scored 1.0 recall on this exact query (at rank 2).

### "an email inviting me to an event, party, or gathering with a specific date"

- Recall@k: 0.167
- Returned: `['events-006', 'newsletters-011', 'events-011', 'newsletters-010', 'newsletters-012']`
- Expected: `['events-001', 'events-002', 'events-003', 'events-004', 'events-005', 'events-006', 'events-007', 'events-008', 'events-009', 'events-010', 'events-011', 'events-012']`
- Root cause: checked actual token overlap between the query and each returned newsletter (both lowercased and split the same way `eval/baseline_bm25.py` does).
  The only overlap between the query and `newsletters-011/010/012` is the single stopword "or."
  None of "event," "party," "gathering," or "date" appear in these newsletter bodies at all (bodies read like "Here's what happened in fintech this week, in five minutes or less.").
  `eval/baseline_bm25.py` does no stopword filtering, and these newsletter bodies are very short (about 14 tokens).
  BM25's length-normalization term inflates the score for a matched token in a short document relative to the corpus average, so a single hit on "or" in a 14-word newsletter outscores real topical overlap in a longer, correctly-categorized email.
  Verified this by computing raw BM25 scores directly with `BM25Retriever`: `newsletters-011` scores 3.143, ahead of most genuine events-* emails, purely off matching "or."

### "an email about an upcoming deadline, due date, or submission cutoff"

- Recall@k: 0.167
- Returned: `['newsletters-011', 'newsletters-010', 'newsletters-012', 'deadlines-001', 'deadlines-002']`
- Expected: `['deadlines-001', 'deadlines-002', 'deadlines-003', 'deadlines-004', 'deadlines-005', 'deadlines-006', 'deadlines-007', 'deadlines-008', 'deadlines-009', 'deadlines-010', 'deadlines-011', 'deadlines-012']`
- Root cause: same mechanism as the previous case.
  The same three short newsletter emails (`newsletters-011/010/012`) crowd out real deadlines-* emails via the stray "or" match, boosted by BM25's short-document length normalization (verified: `newsletters-011` scores 3.143, ahead of `deadlines-003` through `deadlines-012`).
  Two real deadlines emails do make it into the top 5 (`deadlines-001`, `deadlines-002`), but three of five slots are wasted on newsletters that share no topical content with the query.

### "a receipt or payment confirmation for something I bought"

- Recall@k: 0.25
- Returned: `['receipts-005', 'receipts-006', 'receipts-004', 'newsletters-011', 'newsletters-010']`
- Expected: `['receipts-001', 'receipts-002', 'receipts-003', 'receipts-004', 'receipts-005', 'receipts-006', 'receipts-007', 'receipts-008', 'receipts-009', 'receipts-010', 'receipts-011', 'receipts-012']`
- Root cause: the first three returned ids are genuine receipts-* hits (all "Payment confirmation - ..." subjects, matching "confirmation" directly, and scoring far higher at 9.283-8.943 than anything else in the corpus).
  The last two slots go to the same `newsletters-011/010` pair as above, again via the stray "or" match inflated by short-document length normalization, displacing receipts-001/002/003 (the "Thanks for your purchase" phrasing, which shares no literal token with "receipt or payment confirmation") and the rest of the receipts-007-012 range.

## Pattern

The data does not support the "whole-email-embedding dilution" theory: none of the semantic pipeline's low-recall queries show a sub-topic lost inside a longer email, since every corpus email is one or two sentences long.
Three of its four genuinely low scores are k=5-vs-12-relevant-ids benchmark ceiling effects (precision@k of 1.00 on those three), and the fourth is a one-slot miss from a narrow category-boundary overlap; on every query with only 1-3 relevant ids, semantic recall was a perfect 1.0.
BM25's failures, by contrast, cluster around two verified causes: literal keyword mismatch between the query and the correct email's actual wording, and a lack of stopword filtering that lets short, topically unrelated newsletters outrank longer, correct emails on a single stray match to the word "or," amplified by BM25's document-length normalization.
