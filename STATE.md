# Current state

Last updated: 2026-09-01

## Completed

- Task 1 dataset audit complete (`notes/manual_dataset_audit.md`).
- H0/H1/H2 frozen in `PROJECT_BRIEF.md` before any activation result existed.
- Extraction, probe fitting and the frozen 4,000-row train sample are validated and
  committed (`5413fa1`, `d4b72a4`, `7367186`, `416fe40`).
  Sample is deterministic from seed 42: train-only 4,000 rows `9ae14f9e27a5f66d`;
  full 4,216-row worklist `c261306dde08c8b9` (this is what a run `config.json`
  records). The value `d9eb713bcd366b6a` recorded here previously was the 10-row
  `--smoke` worklist hash, not the frozen sample; corrected 2026-09-01, no result
  depended on it.
- **R001 complete on the Mila cluster** (A100-80GB, 53.7 min, job 10621220).
  See `RESULTS.md` R001 for the full table and caveats.
- **R002 complete** (no GPU): D007 embargo lifted, output-level baselines fit,
  comparator/inference rules recorded as D010. See `RESULTS.md` R002.
- **R003 complete** (no GPU): the preregistered D008 within-question ood control.
  The probe separates 15/16 questions -- and so does raw `token_length`, exactly.
  Recorded as D011; `ood_test` is now closed to further inspection.
  R003 controls question/prompt/topic, NOT the reasoning trace (paired rows come
  from different rollouts). See the R003 wording correction in `RESULTS.md`.
- **R004 complete** (no GPU): the length control on `test` (primary) and `val`.
  Length is at chance within question on test (0.494) while the probe is 0.874,
  and the probe is right on 92% of the pairs where length points the wrong way.
- **R005 complete** (no GPU, nothing fit): benchmark-construction audit of the
  released v8 builders. Global 500-token bucket balance coexists with a
  within-question length-label association in ood_test (0.938) but not in test
  (0.494). Recorded as D012.

## R001 headline

Max OOD AUROC **0.964 at depth 56** [0.899, 1.000], question-clustered.
Depth profile 8/24/40/56/64 -> 0.699 / 0.832 / 0.904 / 0.964 / 0.892 OOD.
Val-selected depth is 40 (OOD 0.904); the two are not separable given the CIs.
Reproduction bar (>= 0.85) cleared; >= 0.90 rule reached, so **probe optimisation
stops**. All pre-run gates passed (CUDA residency, activation site == model's own
`last_hidden_state`, no truncation, 14.4 GB free at the longest example).

## Key audit findings that constrain interpretation

- Zero `question_id` overlap between any pair of splits.
- Raw length is a train-only shortcut: AUROC 0.606 train, 0.510 test, 0.584 ood.
- Empirical purity differs by split: val/test 40–50/50, ood_test 45–50/50.
  val is both the noisiest-label split and the selection split.
- `yes_count + no_count < total_resamples` in 139/216 eval rows, so `yes_count/50`
  is not a calibrated termination probability.
- 16 of 32 ood_test questions carry both labels (D008 paired control, now available).

## Current beliefs

Informal research beliefs, not statistical probabilities.

- H0 immediate-output: 15% -- `think_logprob` collapses to 0.57-0.64 exactly where
  the probe stays above 0.9 (R004 B/C).
- H1 broader termination-ready latent state: 55%
- H2 generic progress / other shortcut: 30% -- its cheapest form (raw length) is
  falsified on `test`; a question-relative progress variable remains conceivable but
  is not cleanly testable on the released splits.

The more important update is not a probability: **the released ood_test split cannot
discriminate H1 from H2**, because its within-question length-label association
(0.938) survives the builders' global 500-token bucket balancing. That is an
identification limit of the evaluation, recorded as D012.

## Where the evidence stands

| claim | status |
|---|---|
| the phenomenon reproduces (OOD AUROC 0.90+) | R001, solid |
| reducible to immediate `</think>` propensity | R002+R003+R004, no |
| more than between-question topic structure | R003, yes (15/16, p=0.0005) |
| more than raw prefix length, in-domain | R004 on `test`, yes (0.874 vs 0.494; 0.914 on 52 length-discordant pairs) |
| more than reasoning progress, cross-domain | **not establishable on the released ood split** (R005 / D012) |
| the probe is merely a length detector | contradicted by R004 |

## R005 headline

| split | pooled length AUROC | within-question length concordance | depth-40 probe, same metric | mean Δlen (YES−NO) |
|---|---|---|---|---|
| val | 0.450 | 0.653 | 0.907 | +226 |
| test | 0.499 | **0.494** | 0.874 | +207 |
| ood_test | 0.587 | **0.938** | 0.938 | **+432** |

The builders filter length globally to [500, 3000), balance classes per prompt by
label quality, then balance length in **global 500-token buckets**; no step matches
YES/NO length within a prompt. ood_test's +432-token mean within-question gap is
smaller than the 500-token bucket width, so global balancing cannot see it.

## Next experiment (exactly one, then stop -- D012)

**R006 — representation-level control on `test`.** Does the depth-40 score retain
held-out predictive signal after removing its marginal linear associations with
`think_logprob` and `token_length`? Fit the nuisance relation without `test` labels,
document the split that determines it, evaluate on `test`, and describe the result
narrowly: it removes fitted marginal associations, not "termination propensity" and
not length information from the representation. No GPU.

Then stop experimenting and write up R001-R006.

## Current blocker

None. R006 needs no GPU: frozen R001 probe scores, cached D007 features, released
`token_length`.
