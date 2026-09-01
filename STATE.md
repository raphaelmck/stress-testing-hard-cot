# Current state

Last updated: 2026-08-31

## Current conclusion

**Dataset audit complete; hypotheses and evaluation rules frozen.** No activations extracted,
no probes fitted. The phenomenon has not yet been reproduced.

## Best evidence

From `src/inspect_task1.py` over all 46,376 released records (details in
`notes/manual_dataset_audit.md`, tables in `artifacts/tables/`):

- Splits are clean at the question level: **zero `question_id` overlap** between any pair.
- Train labels are a pure word-distance proxy (positives 25–55 from end; negatives a ladder
  300 -> 11,300). Eval labels are behavioural (50 resampled continuations).
- Raw length is a train-only shortcut: AUROC 0.606 train, 0.466 val, **0.510 test, 0.584 ood**.
  A strong OOD activation result cannot be dismissed as "positives are longer".
- Empirical purity thresholds differ by split: val/test 40–50/50, **ood_test 45–50/50**.
  `ood_test` is the cleanest split; `val`, the selection set, is the noisiest.
- `yes_count + no_count < total_resamples` in 139/216 eval records, so `yes_count/50` is
  **not** a calibrated termination probability.
- 16 of 32 `ood_test` questions carry both labels — a within-question paired control,
  preregistered in D008 and embargoed until reproduction is done.

## Known caveats

- `ood_test` is 58 rows from 32 questions. All CIs are question-clustered bootstraps (D006);
  differences below ~0.05 AUROC are not interpretable.
- `val` (n=72) is both the noisiest-label split and the model-selection split. val AUROC below
  ood AUROC is expected from label noise alone and is not evidence of anything.
- Beliefs below remain priors — no activation evidence exists yet.

## Current beliefs

Informal research beliefs, not statistical probabilities.

- H0 output-proximal: 40%
- H1 broader latent state: 40%
- H2 proxy/dataset structure: 20%

The length result weakens the crudest form of H2 (raw length does not transfer) but leaves
intact the real version: a probe trained on the proxy distribution may still learn generic
"late-reasoning" features.

## Open questions

- Does `</think>` logit/probability alone approach hidden-state probe OOD AUROC? (embargoed
  until reproduction is frozen — D007)
- At what depth does the activation signal appear relative to output-level evidence?
- Does probe score separate the within-question opposite-label pairs? (D008)
- How should the unclassified resample mass be handled in a continuous-target analysis?

## Next experiment

ONE experiment only:

**R001 — sparse-layer reproduction.** 4,000 balanced train rows (2,000 yes / 2,000 no; 500+500
per family across BIG-Bench / GPQA-chem / RACE / DailyDilemmas — verified each family holds
5,770 per label; seed 42), all val/test/ood_test rows. Last-real-prefix-token residual stream
at depths 8, 24, 40, 56, 64. Independent L2 logistic probe per layer, C in {0.01, 0.1, 1, 10}
chosen on val AUROC only.

Prerequisite ticket (infrastructure, produces no RESULTS entry): `src/extract_task1_activations.py`
and `src/fit_task1_probe.py`, with the 10-example smoke test passing before any full run.

Decision rule on max OOD AUROC over the five predeclared layers:
- **>= 0.90** -> reproduction excellent. Stop optimizing the probe entirely; go to Phase B.
- **0.85–0.90** -> accepted; proceed to output-level baselines (lift the D007 embargo).
- **0.75–0.85** -> record it, check training-set size and exact preprocessing once, then proceed.
- **< 0.75** -> assume implementation mismatch. Debug chat formatting, activation token index,
  padding side, split usage, and model revision before any scientific interpretation.

## Blockers

- **GPU.** Qwen3-32B is 65.5 GB in bf16; the local 48 GB Mac cannot host it and quantizing
  would alter the activations under study (D004). Target is Colab **A100 80 GB or H100** —
  the 40 GB A100 variant is insufficient, so check `nvidia-smi` before starting.
  Local work is limited to dataset/tokenizer/probe code and the smoke test.
- Not blocked: env is locked (`uv sync`), and the upstream prompt helper is verified to
  produce a correct single-`<think>` prompt under it (D004).
