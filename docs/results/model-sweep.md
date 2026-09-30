# Model and prompt sweep: quality vs. cost

**Run:** 2026-09-30 · `fh-experiment experiments/model_sweep.yaml` · raw output in
[`runs/model-sweep/`](../../runs/model-sweep/results.md)

## Question

Which model and prompt version should the extractor use? "Best" means the most accurate
extraction per dollar, measured against expert labels rather than judged by eye.

## Setup

| | |
|---|---|
| Patents | 5 file histories: US 8,046,721 (Apple), 8,702,308 (Poly-America), 8,900,294 (Colibri), 10,469,966 (Sonos), 10,858,176 (K-fee) |
| Gold labels | 243 scored items in 77 documents, reviewed by a patent analyst (the author) in `label_app.py` |
| Finding types scored | Rejections/objections, allowable claims, claim amendments, independent claim text, reasons for allowance, examiner's amendments. Estoppel is switched off for this run. |
| Grid | 3 models (Sonnet 5, Haiku 4.5, Opus 5.5) × 2 prompt versions = 6 configurations |
| Prompt versions | `2026-09-25.2` (current) and `…-no-rule3`, an ablation that removes the rule "report ALL of it; the summary is not a substitute for the lists" |
| Fixed | Completeness retry on, strict tool use, no LLM judge, one run per configuration |
| Input | The same saved OCR text for every configuration (no re-download, no re-OCR) |
| Metric | Micro-averaged F1 over matched items; 90% bootstrap interval over patents; dollars = measured API spend per patent |

## Results

![F1 vs. dollars per patent](model-sweep-frontier.png)

| Configuration | F1 (90% CI) | Precision | Recall | $ / patent | Citations verified |
|---|---:|---:|---:|---:|---:|
| **Sonnet 5, current prompt** ★ | **92%** (89–96) | 87% | **96%** | $0.34 | **99%** |
| Sonnet 5, no rule 3 ★ | 87% (82–94) | **96%** | 79% | $0.32 | 99% |
| Haiku 4.5, current prompt ★ | 68% (57–86) | 73% | 63% | $0.13 | 84% |
| Haiku 4.5, no rule 3 ★ | 63% (50–78) | 65% | 61% | $0.13 | 91% |
| Opus 5.5, both prompts ⚠ | not measured: every call was rejected by the API (see below) | | | | |

★ = on the Pareto frontier. The Haiku ablation is on it only because it is 0.3¢ cheaper;
in practice it is dominated.

F1 by finding type:

| Configuration | Rejection | Amendment | Claim text | Allowable | Allowance reason | Examiner's amendment |
|---|---:|---:|---:|---:|---:|---:|
| Sonnet, current | 93% | 84% | 97% | 100% | 100% | 100% |
| Sonnet, no rule 3 | 81% | 82% | 94% | 100% | 80% | 100% |
| Haiku, current | 70% | 66% | 66% | 100% | 100% | 100% |
| Haiku, no rule 3 | 76% | 44% | 67% | 100% | 100% | 100% |

The last three columns rest on only a handful of items per patent, so read them as "no
failures seen", not as 100% accuracy.

## Findings

**1. Sonnet with the current prompt is the choice: 92% F1 for $0.34 per patent.**
The whole five-patent sweep cost $4.59. At about a third of a dollar per file history,
cost is not the constraint for this task; accuracy is.

**2. Haiku is 60% cheaper but not good enough.** It saves about $0.20 per patent and loses
24 points of F1. The drop is uneven:

- Short, clean file histories were fine (Apple: F1 0.90; Poly-America: 0.90).
- The two long, messy ones collapsed. K-fee (a 17-ground office action) fell to 0.50 F1,
  with 26 independent-claim entries and 14 rejections missed. On Colibri, it confused claim
  listings with amendments (17 spurious claim entries, 19 missed amendments).

Haiku's quotes are also less trustworthy: only 84% are found verbatim on the page it cites,
against 99% for Sonnet. A cheaper model is not just less complete; it cites less reliably.

**3. The "report everything" rule trades precision for recall, and its net value depends on
how the labels count things.** Removing rule 3 lowers recall from 96% to 79% and raises
precision from 87% to 96%, at a saving of $0.02 per patent. Error analysis shows where the
recall goes:

- **15 of the 51 items the ablation misses are finer-grained entries, not missed grounds.**
  In Sonos, the gold labels count one §103 ground plus 7 per-claim-group entries over the
  same reference. In K-fee, they count one §112(b) ground plus 8 sub-points. With rule 3 the
  model enumerates these sub-entries; without it, it reports the ground once.
- **The other 36 are real misses:** 22 amendments (20 of them in Colibri's large
  2014-08-21 response), 9 claim-listing entries in K-fee, and 5 drawing, specification and
  claim-numbering objections in K-fee.
- **Sensitivity check:** if the gold counted one item per ground, as the extraction prompt
  asks, a rough recount puts the two prompts about even (F1 about 0.88 with rule 3 against
  0.90 without).

The conclusion is not "rule 3 helps". It is that the labeling convention for granularity
drives the result, and it has to be fixed before prompt variants can be ranked. That is now
the first follow-up.

**4. Opus 5.5 could not be tested with the current client.** Every call returned
`400: tool_choice: type "tool" and "any" are not supported for this model`. The extractor
forced tool use, which this model does not accept. All 77 calls per configuration failed,
so it cost $0 and scored 0.

The runner originally drew those configurations on the Pareto frontier, because $0 is the
cheapest point. The fix: a configuration with failed steps is now reported with ⚠ and kept
off the frontier and the chart. The client now also falls back to `tool_choice: auto` with
an explicit instruction to call the tool (with a test). Re-running the grid retries only the
failed runs.

## Where Sonnet (current prompt) still goes wrong

Of 268 predictions, 34 were unmatched and 9 gold items were missed.

- **Amendments that are only "new claim added" (23 unmatched).** The gold labels were
  inconsistent here. Apple's new dependent claims were labeled as amendments; most other
  new claims were not. This needs a labeling rule, not a model fix.
- **An advisory action read as a rejection (Apple).** This is the same error the labeler
  marked incorrect in the original run, so it is systematic, not random.
- **Drawing and specification objections** at the end of long office actions were
  occasionally dropped (K-fee).
- **Field accuracy on matched items is high:** 99% of rejection claim sets and 96% of
  reference lists match exactly, and 99% of claim texts are within 95% similarity of the
  labeled text.
- **One labeling error found during this analysis:** the K-fee rejection over Kruger was
  labeled claims 9, 18, 26, 29. The office action says "Claims 9 — 18 and 26 — 29"; the
  spaced dashes in the OCR defeated both the model and the reviewer. It will be corrected
  in the gold file.

## Caveats

- **Five patents, one run each.** The intervals are wide (Haiku's F1 spans 57–86%).
  Run-to-run noise was not measured; `repeats: 3` would measure it.
- **The gold labels started from Sonnet's own output** (the current prompt), which the
  labeler then confirmed or corrected. That favors Sonnet with the current prompt. Recall
  is not 100% here only because a fresh run differs from the run the labels came from. No
  missed items were added by hand, so misses that every model shares are invisible.
- **Labels were reviewed quickly** (about 3–11 seconds per item) and have not been
  double-checked by a second reviewer. The known issues are the per-ground granularity and
  new-claim amendments described above.
- **Estoppel analysis was switched off** and is not covered by these results.

## Next steps

1. **Fix the labeling conventions:** one item per ground, and a rule on new claims. Correct
   the Kruger item and re-score. Scoring is free: `fh-score --experiment runs/model-sweep`.
2. **Re-run Opus 5.5** with the tool-choice fallback, which re-runs only the failed
   configurations.
3. **Run `repeats: 3`** for the Sonnet configurations, to see whether the rule-3 gap is
   larger than run-to-run noise.
4. **Try routing by document type:** Haiku for short, formulaic documents (notices of
   allowance, claim listings); Sonnet for office actions and responses.
