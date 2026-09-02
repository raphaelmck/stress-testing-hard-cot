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
- **R007 complete** (D013, one A100 run): causal steering along the frozen depth-40
  direction. The edit lands and propagates but does not move behaviour; a
  matched-norm orthogonal control moves it slightly more.
  **Experimentation is now closed permanently.**

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

Final. Informal research beliefs, not statistical probabilities.

- H0 immediate-output: 15%
- H1 broader termination-ready latent state: 45% (down from 50% -- R007 found no
  causal signature, which is weak evidence but not zero evidence)
- H2 generic progress / other shortcut: 40%

Two load-bearing conclusions are not probabilities: the released `ood_test` split
cannot discriminate H1 from H2 (D012), and the project has **no** causal evidence
either way (R007 is a weak null by design).

## Where the evidence stands

| claim | status |
|---|---|
| the phenomenon reproduces (OOD AUROC 0.90+) | R001, solid |
| reducible to immediate `</think>` propensity | R002+R003+R004, no |
| more than between-question topic structure | R003, yes (15/16, p=0.0005) |
| more than raw prefix length, in-domain | R004 on `test`, yes (0.874 vs 0.494) |
| survives a linear control for length + stop propensity | R006, yes, at a cost (0.909 -> 0.831) |
| more than reasoning progress, cross-domain | not establishable on the released ood split (R005/D012) |
| the decoded direction is causally used for stopping | **no evidence** (R007 null; weak by construction) |

## R007 headline (test, 2,408 generations, 7 conditions)

Baseline protocol validation: released YES prefixes terminate within 60 tokens at
**0.895**, released NO at **0.000** -- the generation setup reproduces the labels.

P(`</think>` within 60): beta -2/-1/0/+1/+2 = 0.439 / 0.436 / 0.448 / 0.459 / 0.453.
Primary contrast +2beta - -2beta = **+0.0145 [+0.0019, +0.0306]**; matched-norm
orthogonal contrast **+0.0174 [+0.0000, +0.0404]**, i.e. larger. On released-NO
prefixes, +2beta produced termination in **0 of 172** generations. 85-88% of steered
continuations are token-identical to baseline under common random numbers.

The edit verifiably lands (Δs to 0.16%) and propagates (~2% of the depth-64 norm), so
this is a real null rather than a broken intervention -- but it is a null for one
direction, one layer, one schedule, at 2.6% of the residual norm, and it does not
establish the absence of a causal termination mechanism.

## Next step: no more experiments

Write up R001-R007. The contribution: strong cross-domain linear decodability does
not by itself identify a domain-general termination representation. This project
reproduces the result, rules out the output-proximal and raw-length explanations
in-domain, shows the released OOD evaluation leaves a question-conditional length
confound that global length balancing does not constrain, and finds no causal effect
of the decoded direction under a small, verified, matched-control intervention.

## Current blocker

None. All seven experiments are complete; the remaining work is the writeup.
