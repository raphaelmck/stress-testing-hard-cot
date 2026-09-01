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

Informal research beliefs, not statistical probabilities. R001 was predicted by all
three hypotheses, so it moved them little; the small shift is from the depth profile
(peak at 56, not at the final layer), which is weak evidence at these CIs.

- H0 output-proximal: 35%
- H1 broader latent state: 45%
- H2 proxy/dataset structure: 20%

## Next experiment (exactly one)

**R002 — output-level baseline.** Fit the identical pipeline (L2 logistic, same C
grid, val-only selection, question-clustered bootstrap) to the D007 features already
cached in the R001 pass — `think_logit`, `think_logprob`, `think_margin`,
`next_token_entropy`, `top1_logprob` — with no hidden activations. Same rows, same
splits. No GPU needed.

The D007 embargo lifts with the committed R001 entry, by its own terms.

Decision rule, on OOD AUROC of the best output-level baseline vs R001's 0.964:

- within ~0.03 -> H0 is doing most of the work; next step is residualising
  activations against these features, not more probing.
- 0.05–0.15 below -> a real activation residual exists; go to the depth-vs-output
  timing comparison (does depth 40 beat the final-layer output signal?).
- more than ~0.15 below -> H0 is weak here; prioritise D008 within-question paired
  analysis and the continuous-target check.

Report the baseline for each feature alone as well as the combined set; a single
feature matching 0.96 is a much stronger H0 result than a five-feature combination.

## Current blocker

None. R002 needs no GPU: everything runs from
`artifacts/runs/r001_qwen32b/probe_scores.csv` and the cached output features.
