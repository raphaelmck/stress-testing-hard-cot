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

Informal research beliefs, not statistical probabilities. R004 reversed most of the
R003 update: the cheap length explanation failed on the split with power.

- H0 immediate-output: 15% -- `think_logprob` collapses to 0.57-0.64 exactly where
  the probe stays above 0.9 (R004 B/C).
- H1 broader termination-ready latent state: 45% (up from 35%)
- H2 as *raw depth into the trace*: largely falsified on `test` (length 0.494).
- H2 as *question-relative reasoning progress*: 40% -- untested, and now the main
  competitor to H1. This is the version that could explain a linear probe that
  generalises to unseen questions, which raw token count cannot.

## Where the evidence stands

| claim | status |
|---|---|
| the phenomenon reproduces (OOD AUROC 0.90+) | R001, solid |
| reducible to immediate `</think>` propensity | R002+R003+R004, no |
| more than between-question topic structure | R003, yes (15/16, p=0.0005) |
| more than raw prefix length | R004 on `test`, yes (0.874 vs 0.494; 0.914 on 52 length-discordant pairs) |
| more than question-relative reasoning progress | **untested -- this is R005** |
| cross-domain claim survives the length control | untested; `ood_test` is closed |

## R004 headline (primary split = `test`, 17 questions, 137 pairs)

Within-question macro concordance: activation **0.874** [0.727, 0.977];
`think_logprob` 0.794 [0.664, 0.908]; `token_length` **0.494** [0.321, 0.667].
Paired delta activation − length [+0.137, +0.605], P(Δ>0) = 0.998.
On the 52 length-discordant pairs (14 questions): activation 0.923 pooled / 0.914
macro, above 0.5 on 12/14 questions; `think_logprob` 0.635 pooled.
Near-matched |Δlen| ≤ 100: activation 0.983, think 0.567 (23 pairs, 10 questions).

## Next experiment (exactly one)

**R005 — question-relative progress.** Build a progress variable that a global linear
probe could plausibly track across unseen questions: prefix length as a fraction of
its own rollout's total, and/or the prefix's rank among the prefixes of its question.
Run exactly the R004 machinery on `test`: within-question concordance for progress,
then the probe's concordance on the pairs where *progress* is discordant or matched.
No GPU, no refitting.

Decision rule:

- probe still wins on progress-discordant pairs -> the signal is not generic progress
  either; H1 leads and activation-level residualization becomes R006.
- probe collapses toward 0.5 there -> the honest headline is that a strong
  cross-domain termination probe largely tracks question-relative reasoning progress
  rather than a termination-specific state.
- too few discordant pairs -> declare underpowered, do not infer either.

Do not reopen `ood_test`. Do not fit new activation probes at other depths.

## Current blocker

None. R005 needs no GPU: frozen R001 probe scores and the released rollout fields.
