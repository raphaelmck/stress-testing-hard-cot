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

Informal research beliefs, not statistical probabilities. R002 moved these: a single
untrained scalar recovers most of the distance from chance, but a paired-bootstrap
advantage for activations survives it.

- H0 output-proximal: 45% (up from 35%)
- H1 broader latent state: 40%
- H2 proxy/dataset structure: 15%

R002 also produced a structural fact that constrains both: the output features are at
chance on the distance-based **train** labels (AUROC ~0.51) yet reach 0.81 OOD, and
the activation probe transfers across that same label-construction gap at 0.90.

## R002 headline

Primary comparator = val-selected depth 40, OOD 0.904 (D010; NOT depth 56 / 0.964).

| baseline | OOD AUROC [95% CI] |
|---|---|
| activation, depth 40 | 0.904 [0.834, 0.966] |
| `think_logprob` (untrained scalar) | 0.807 [0.696, 0.905] |
| `think_margin` | 0.803 [0.694, 0.902] |
| multivariate output-only (handicapped, see R002) | 0.727 [0.607, 0.843] |

Paired question-clustered delta, depth 40 − `think_logprob`, on ood_test:
**+0.096 [+0.000, +0.201]**, P(Δ>0) = 0.975 → the 0.05–0.15 band.
`top1_is_think` is identically zero on all 4,216 rows; mean p(`</think>`) ~2e-12.

## Next experiment (exactly one)

**R003 — residual/conditional test.** Does the depth-40 activation advantage survive
removing what the output distribution already knows? Regress the output-level
features out of the activation representation (or equivalently fit the probe on the
residual), refit under the identical D005 protocol, and evaluate with the same
paired question-clustered bootstrap against both the raw depth-40 probe and
`think_logprob`. No GPU needed.

Decision rule on the residualised probe's OOD AUROC:

- drops to within ~0.03 of `think_logprob` (≈0.81) -> the advantage was output
  information after all; H0 is close to sufficient and the project's answer is a
  sharp near-negative result.
- stays within ~0.03 of 0.904 -> the activation signal is largely independent of
  immediate `</think>` propensity; H1 becomes the leading explanation and D008
  (within-question paired control) becomes the next target.
- lands between -> report the partition honestly; prefer D008 over more probing.

Do not run D008 yet. Do not fit new activation probes at other depths.

## Current blocker

None. R003 needs no GPU: it runs from the cached R001 activations
(`artifacts/runs/r001_qwen32b/activations/`) and the D007 output features.
