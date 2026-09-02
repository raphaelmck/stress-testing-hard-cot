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
  (0.494). Recorded as D012. (Wording correction appended to `RESULTS.md`: global
  bucket balancing does not constrain within-prompt or within-bucket label-length
  differences -- the bucket width is a plausible explanation, not a demonstrated
  mechanical cause.)
- **R006 complete** (no GPU, nothing refit): score-level nuisance control on `test`.
  Joint residual 0.831 [0.716, 0.928] vs raw 0.909; drop 0.077, middle band.
  **Experimentation stops here (D012).**

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

Final for this sprint. Informal research beliefs, not statistical probabilities.

- H0 immediate-output: 15%. `think_logprob` reaches 0.807 OOD alone but is at chance
  on the training distribution and collapses to 0.57-0.64 where the probe holds
  above 0.9 (R002, R004). It is the largest single nuisance in R006 nonetheless.
- H1 broader termination-ready latent state: 50%.
- H2 generic progress / other shortcut: 35%. Raw length is falsified in-domain
  (R004); a question-relative progress variable is neither tested nor cleanly
  testable on the released splits.

The load-bearing conclusion is not a probability: **the released ood_test split
cannot discriminate H1 from H2** (D012), while in-domain `test` shows signal beyond
length, immediate stop propensity, and their fitted linear associations.

## Where the evidence stands

| claim | status |
|---|---|
| the phenomenon reproduces (OOD AUROC 0.90+) | R001, solid |
| reducible to immediate `</think>` propensity | R002+R003+R004, no |
| more than between-question topic structure | R003, yes (15/16, p=0.0005) |
| more than raw prefix length, in-domain | R004 on `test`, yes (0.874 vs 0.494) |
| survives a linear control for length + stop propensity | R006, yes, at a cost (0.909 -> 0.831) |
| more than reasoning progress, cross-domain | not establishable on the released ood split (R005/D012) |
| a termination-specific latent state exists | **not established**; no intervention was run |

## R006 headline (test; nuisance fit on val, labels unused)

Val nuisance fit R^2 = 0.334 (length-only 0.098, think-only 0.274).

| score | pooled AUROC [CI] | within-question |
|---|---|---|
| raw depth-40 | 0.909 [0.835, 0.966] | 0.874 |
| joint residual | **0.831 [0.716, 0.928]** | 0.818 |
| length-only residual | 0.882 [0.779, 0.954] | 0.918 |
| think-only residual | 0.841 [0.717, 0.934] | 0.868 |

Paired delta raw − joint residual: +0.077 [−0.006, +0.173]. Almost all of the cost
comes from `think_logprob`; removing length alone costs 0.027 and *raises*
within-question concordance.

## Next step: no more experiments (D012)

Write up R001-R006. The contribution is not "we found the termination state"; it is:
strong cross-domain linear decodability does not by itself identify a domain-general
termination representation. This project reproduces the result, rules out the
simple output-proximal and raw-length explanations in-domain, and shows that the
released OOD evaluation leaves a question-conditional length confound that global
length balancing does not constrain.

Open questions worth naming rather than running: a question-relative progress
control that does not use future information; a representation-level (not
score-level) nuisance projection; and any causal intervention, of which this project
ran none.

## Current blocker

None. All six experiments are complete; the remaining work is the writeup.
