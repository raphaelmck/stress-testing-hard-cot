# Current state

Last updated: 2026-09-01

## Completed

- Task 1 dataset audit complete (`notes/manual_dataset_audit.md`).
- H0/H1/H2 frozen in `PROJECT_BRIEF.md` before any activation result existed.
- R001 training sample frozen: 4,000 rows, 500 yes + 500 no per training family
  (`big_bench`, `gpqa_chem`, `race`, `daily_dilemmas`), seed 42,
  sample sha256 begins `d9eb713bcd366b6a`. Spans 779 questions, max 41 rows/question.
- `src/task1_data.py` implemented and validated.
- `src/extract_task1_activations.py` implemented and validated end to end:
  - exact upstream prompt construction (released `build_thinking_prompt`);
  - no truncation; aborts if any example exceeds the context limit;
  - right-padding invariance verified (bit-identical at all five depths);
  - preregistered depths 8/24/40/56/64 (blocks 7/23/39/55/63);
  - base-transformer forward, so no full-sequence vocabulary logits;
  - final-depth hook + final norm agrees exactly with the model's own
    `last_hidden_state` (`max_abs_diff=0`), confirming index and norm;
  - D007 output features cached in the same pass, inaccessible to the probe script;
  - resumable sharded output; `question_id` preserved on every row;
  - CUDA residency enforced -- CPU/disk offload aborts;
  - `--stress` mode for the pre-launch memory tests.
- `src/fit_task1_probe.py` implemented and validated on synthetic data in the real
  shard format: L2 logistic regression, C in {0.01, 0.1, 1, 10}, validation-only
  selection, raw per-example scores retained, question-clustered OOD bootstrap
  that discards single-label replicates rather than coercing them.
- Commits: audit `5413fa1`, extraction `d4b72a4`, probe `7367186`.

## Key audit findings that constrain interpretation

- Zero `question_id` overlap between any pair of splits.
- Raw length is a train-only shortcut: AUROC 0.606 train, 0.510 test, 0.584 ood.
  A strong OOD activation result is not reducible to "positives are longer".
- Empirical purity differs by split: val/test 40–50/50, ood_test 45–50/50.
  `val` is both the noisiest-label split and the selection split, so val AUROC
  below ood AUROC is expected from label noise alone.
- `yes_count + no_count < total_resamples` in 139/216 eval rows, so `yes_count/50`
  is not a calibrated termination probability.
- 16 of 32 ood_test questions carry both labels (D008 paired control, embargoed).

## Current beliefs

Informal research beliefs, not statistical probabilities. No activation evidence yet.

- H0 output-proximal: 40%
- H1 broader latent state: 40%
- H2 proxy/dataset structure: 20%

## R001 next experiment

Run the preregistered Qwen3-32B activation-probe reproduction. Full runbook:
`notes/gpu_runbook.md`.

Before full extraction, on the real GPU:

1. Real Qwen3-32B 10-example smoke test (`--smoke`).
2. Representative ~1.5–2.5k-token throughput/memory test (`--stress`, test B).
3. Longest selected training prefix (~16.5k tokens) at batch size 1 (`--stress`, test C).
4. Confirm all parameters/buffers remain CUDA-resident.
5. Confirm sufficient memory headroom (target: several GB free at peak, not tuned to the edge).

Then launch the frozen 4,000-train + all val/test/ood_test extraction, and run the
frozen probe script immediately after.

Do not inspect D007 output features or run D008 paired analyses until the activation
reproduction metrics are frozen.

## R001 decision rule

On max OOD AUROC across the five predeclared depths:

- **>= ~0.85** -> reproduction adequate; proceed to explanation tests (lift D007 embargo).
- **~0.90+** -> stop optimizing the probe entirely.
- **0.75–0.85** -> investigate sample size / preprocessing / reproduction mismatch.
- **< 0.75** -> treat as likely pipeline mismatch before any scientific interpretation.

## Current blocker

Access to a CUDA GPU capable of keeping Qwen3-32B bf16 fully resident (~65.5 GB
weights; needs an 80 GB card -- A100 80GB, H100, or larger). The 40 GB A100 variant
will abort by design rather than offload.

**Stop coding.** The infrastructure is sufficient; further engineering before real
data has diminishing returns.
