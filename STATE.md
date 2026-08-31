# Current state

Last updated: 2026-08-31

## Current conclusion

Nothing has been reproduced yet. Repo scaffolding and a first automated data inventory exist;
no activations extracted, no probes trained. Phase A step 1 (data audit) is partially done.

## Best evidence

Automated inventory of `cot-proxy-tasks/datasets/1/` (commit `4482324`), verified 2026-08-31:

| split | prompts | rollout records | labels (no/yes) | label type |
|---|---|---|---|---|
| train | 796 | 46,160 | 23,080 / 23,080 | proxy (`distance_from_end`) |
| val | 30 | 72 | 36 / 36 | behavioral (`yes_count`/`total_resamples`) |
| test | 22 | 86 | 43 / 43 | behavioral |
| ood_test | 32 | 58 | 29 / 29 | behavioral |

- Train records carry `distance_from_end`, `prefix_words`, `total_words` and no resampling fields.
- Val/test/ood_test records carry `yes_count`, `no_count`, `total_resamples`,
  `mean_yes_position`, `token_length` and no distance fields.
- This confirms the brief's warning: train and eval labels are constructed differently.

## Known caveats (established before any result)

- **The evaluation sets are tiny.** OOD AUROC is measured on 29 positives vs 29 negatives.
  Every reported AUROC needs a bootstrap CI; differences below ~0.05 are probably not real.
  Report CIs from the first run onward, not retroactively.
- Val (n=72) is behaviorally labelled but is also the regularization-selection set. Selecting
  `C` on 72 behavioral examples while training on proxy labels is a real train/select mismatch.
- No result has been produced, so all beliefs below are priors, not posteriors.

## Current beliefs

Informal research beliefs, not statistical probabilities. No evidence collected yet.

- H0 output-proximal: 40%
- H1 broader latent state: 40%
- H2 proxy/dataset structure: 20%

## Open questions

- Does `</think>` logit/probability alone approach hidden-state probe OOD AUROC?
- At what depth does the activation signal first appear relative to output-level evidence?
- Does probe score track continuous `yes_count/total_resamples` or only the binary label?
- How much of the train->eval generalization is carried by prefix length / position cues?

## Next experiment

ONE experiment only:

**R001 — sparse-layer reproduction.** Balanced 4k subset of `train`, L2 logistic probe on
last-real-prefix-token residual stream at 5 layers spanning Qwen3-32B, `C` chosen on `val`,
evaluated on `test` and `ood_test` with bootstrap CIs.

Prerequisite engineering ticket (not an experiment, produces no RESULTS entry):
a validated activation extractor with a padding-index assertion and a 10-example smoke test.

Why: the brief forbids mechanistic interpretation before the phenomenon is trustworthy,
and everything downstream depends on the extractor being correct.

Decision rule:
- OOD AUROC >= 0.85 -> reproduction accepted, proceed to Phase B output-level baselines.
- 0.75 <= OOD AUROC < 0.85 -> record it, check layer choice and input formatting once, then proceed.
- OOD AUROC < 0.75 -> assume a pipeline bug. Debug chat formatting, activation token index,
  padding side, split usage, and model revision before any scientific interpretation.

## Blockers

- No GPU / inference backend for `Qwen/Qwen3-32B` is configured or verified in this repo yet.
  Resolve this before the extractor ticket; it gates everything.
