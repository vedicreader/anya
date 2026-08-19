# Results

## Fitting a PII detector, against a pattern bank

`evals/pii_learned.py` on `evals/mkpii.py`: 240 examples, 11,922 characters, 9 span kinds over 53
sentence templates. Five splits, 25% of templates held out each time, so no system is scored on a
sentence form it was fitted on. Metrics are the mean over splits. Intervals are a paired bootstrap
over the 295 pooled held-out documents of all five splits, 2000 resamples, 2.5th to 97.5th
percentile. An interval spanning zero is reported as no difference.

"Invented" counts every false-positive span over all five splits, which is the number that decides
whether a fitted layer is safe to put in front of a retrieval gate.

### With `vishalakshi` installed, so the baseline is the real bank

    python evals/pii_learned.py --seeds 5

| arbitration | span P | span R | span F1 | invented | ms/doc | F1 vs baseline |
|---|---|---|---|---|---|---|
| baseline | 0.800 | 0.305 | 0.399 | 0 | 0.08 | |
| learned | 0.941 | 0.756 | 0.826 | 10 | 3.64 | +0.334 [+0.236, +0.428] |
| **defer** | **1.000** | 0.846 | **0.907** | **0** | 3.69 | **+0.418 [+0.341, +0.498]** |
| union | 0.946 | 0.846 | 0.876 | 10 | 3.70 | +0.388 [+0.307, +0.471] |
| hybrid | 0.946 | 0.779 | 0.837 | 10 | 3.68 | +0.346 [+0.253, +0.441] |

The bank finds nothing at all of `badge`, `case_ref`, `emp_id`, `patient_ref` or `staff_login`, which
is the whole reason to fit: 0.305 recall, and none of it recoverable by more patterns, because these
identifiers were invented by one organisation.

Every invented span in the table is a `card`. A tagger trained on payment cards learns that sixteen
Luhn-valid digits are a card, and then labels the device serials the bank had correctly gated behind
its `serial` designator. `defer` removes all ten by never emitting a kind the baseline can produce,
and loses nothing for it: same recall as `union`, 0.054 more precision, 0.031 more F1.

`hybrid` is the worst of both here. It drops the bank's `card` spans (recall 1.00 to 0.00 on that
kind) and keeps the tagger's invented ones.

Span F1 by split, `defer`: 0.983, 0.743, 1.000, 0.824, 0.987. One split in five is much worse than
the mean, and 240 examples is why.

### With neither installed, so the baseline is anya's six-pattern floor

    python evals/pii_learned.py --seeds 5 --floor

| arbitration | span P | span R | span F1 | invented | F1 vs baseline |
|---|---|---|---|---|---|
| baseline | 0.590 | 0.305 | 0.370 | 20 | |
| learned | 0.941 | 0.756 | 0.826 | 10 | +0.376 [+0.284, +0.467] |
| defer | 0.871 | 0.846 | 0.843 | 20 | +0.402 [+0.329, +0.479] |
| union | 0.871 | 0.846 | 0.843 | 20 | +0.402 [+0.329, +0.479] |
| hybrid | **0.946** | 0.779 | 0.837 | **10** | +0.388 [+0.300, +0.476] |

The floor has no designator gate, so it reads the same device serials as payment cards: 20 invented
spans, and `defer` inherits every one because it never overrules the baseline. This is the case
`hybrid` exists for. It halves the invented spans and takes precision from 0.871 to 0.946, for 0.067
recall and 0.006 F1.

So the mode follows from the baseline. `defer` where the pattern bank is the stronger layer, `hybrid`
where it is the problem.

## Nothing to fit on

`fit([])` returns `PiiDetector(unfitted, ...)`, and on 69 held-out documents its spans are identical
to the baseline's, span for span. The fallback is the same object in a different state rather than a
second code path.

## Ablations

Mean span F1 over the same five splits, `defer` against the real bank, one change at a time.

| | mean span F1 | by split |
|---|---|---|
| as shipped | 0.907 | 0.983 0.743 1.000 0.824 0.987 |
| unweighted classes | 0.557 | 0.929 0.305 0.450 0.345 0.758 |
| `C=1` not 4 | 0.864 | 0.915 0.725 0.982 0.765 0.933 |
| `window=0` | 0.847 | 0.906 0.650 0.873 0.818 0.987 |
| `window=4` | 0.811 | 0.915 0.696 0.846 0.667 0.933 |
| no baseline features | 0.900 | 0.983 0.743 1.000 0.788 0.987 |

`class_weight='balanced'` is worth 0.350 F1 and is the one setting that is not a preference: `O`
outnumbers every tag about ten to one, and an unweighted fit answers `O` to almost everything. Two
tokens of context beat none by 0.060 and four by 0.096. The baseline's own verdicts as features are
worth 0.007, which is inside the split-to-split spread; they are still there because they cost one
pass the arbitration already pays for.

## Two things that were tried and are not there

**Tuning the decoding bias.** `tune_bias` sweeps the bias added to `O` and takes the argmax on a
split held out of the training half. Adopting it cost 0.110 F1 (0.907 to 0.797, worse on all five
splits): 34 dev examples do not settle where the operating point belongs. `fit_run` measures the
sweep and draws it in the report, and `tune=True` adopts it for a corpus where the choice transfers.

**A probability-gated veto.** `hybrid` drops a baseline span whose kind the tagger was trained on.
The obvious refinement is to drop it only when the tagger is confident the span is nothing, keeping
it where the tagger is unsure. Measured at thresholds 0.05, 0.15 and 0.3 on three splits, this was
identical to `union` in every cell: with balanced class weights the tagger is never confidently `O`
on a span the baseline claimed, so the gate never fires. It is not in the code.

## What this corpus does not measure

`evals/mkpii.py` is synthetic and written by the same hand as the code. Documents are single
sentences, so the document-level numbers track the span-level ones and the document classifier is
only exercised through its own class accuracy (0.764 against the bank, 0.724 against the floor,
where the spans score 0.907). Real tickets carry several identifiers each and a lot more prose
between them. The templates are held out, but the identifier *formats* per kind are only two deep,
so this measures generalisation across sentence forms and barely at all across formats. Treat the
deltas as evidence that fitting on an organisation's own labels finds what no pattern can, not as a
number to expect on your own corpus.
