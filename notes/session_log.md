# Session log

Narrative history, for orientation only. The authoritative record is
`PROJECT_BRIEF.md` -> `DECISIONS.md` -> `RESULTS.md` -> `STATE.md`; where this file
and those disagree, they win. Commit messages carry the detail.

## 2026-08-31 — repo scaffolding

Started from a bare directory containing only `PROJECT_BRIEF.md` and a 1.1 GB clone
of `Centrattic/cot-proxy-tasks`. Built the four-file knowledge structure, `AGENTS.md`
(with `CLAUDE.md` as a one-line `@AGENTS.md` import rather than a symlink, D002), and
the run-artifact convention. Commit `c959d06`.

## 2026-08-31 — dataset audit

Wrote `src/inspect_task1.py` (stdlib only) and inventoried all 46,376 released
records. Findings that changed the plan, in `notes/manual_dataset_audit.md`:

- splits are question-disjoint; no leakage;
- train labels are a word-distance proxy whose arithmetic checks exactly;
- raw length is a train-only shortcut (0.606 train, 0.510 test, 0.584 ood), so the
  published OOD result is not reducible to "positives are longer";
- the "45/50" purity threshold applies to `ood_test` only; val/test go down to 40/50;
- `yes_count + no_count < total_resamples` in 139/216 eval rows, so `yes_count/50` is
  not a calibrated termination probability;
- 16 of 32 ood_test questions carry both labels -> a within-question paired control.

Froze H0/H1/H2 and the evaluation rules (D005–D008). Commit `5413fa1`.

A note on method: several audit findings arrived from outside this session. They were
checked against the files before being written into the frozen brief; all held, but
two claims needed correcting in the process. Keep doing that -- the ledger is only
worth what its weakest entry is worth.

## 2026-08-31 — environment

`uv` + Python 3.12 + committed `uv.lock`, resolving for macOS/arm64 and Linux/x86_64
so the same lockfile works on a cluster (D004). Verified the released
`build_thinking_prompt` produces exactly one `<think>` under this env -- the main
silent-mismatch risk, since upstream ships no dependency pins at all.

## 2026-09-01 — extraction and probe implementation

`src/task1_data.py` + `src/extract_task1_activations.py` (commit `d4b72a4`), then
`src/fit_task1_probe.py` (commit `7367186`), validated end to end against Qwen3-0.6B
and, for the probe, against synthetic data in the real shard format.

Two bugs caught during review rather than after the GPU run:

- the first version called `Qwen3ForCausalLM.forward`, whose `logits_to_keep=0`
  default materialises `[B, seq, 151936]` logits -- defeating the whole last-token
  design. Fixed by running the base transformer directly;
- the smoke test's padding check reported `max_abs_diff=0`, which a stale or shared
  hook tensor would also produce. Added a non-degeneracy assertion before it, so the
  check cannot pass vacuously.

## 2026-09-01 — compute

Colab evaluated and rejected; moving to the Mila cluster (D009). Runbook and SLURM
script added. Infrastructure work stops here: further engineering before real data
has diminishing returns.

## 2026-09-01 — R001, the reproduction

Ran on a Mila A100-80GB. Prompt gate re-verified on the cluster, smoke and memory
stress passed, then 4,216 rows extracted in 53.7 min. Max OOD AUROC 0.964 at depth
56; the val-selected depth 40 gives 0.904. Reproduction bar cleared, so probe
optimisation stopped and the D007 embargo lifted by its own terms.

Also corrected a hash recorded here in error: `d9eb713bcd366b6a` was the 10-row
smoke worklist, not the frozen sample (train-only `9ae14f9e27a5f66d`, full worklist
`c261306dde08c8b9`). No result depended on it.

## 2026-09-01 — R002, the obvious explanation

`think_logprob` used directly reaches 0.807 OOD against the probe's 0.904, paired
delta +0.096. Two facts mattered more than the gap: the argmax next token is never
`</think>` anywhere in the data, and every output feature is at chance on the
distance-based training labels. So the probe cannot plausibly have learned current
stop propensity from what it was trained on.

## 2026-09-01 — R003, the surprise that redirected the project

The preregistered within-question OOD control (D008). The probe separates 15/16
questions, p = 0.0005 -- and so does raw `token_length`, at exactly the same 0.938,
positive in 16/16. Recorded D011 promoting length to the primary nuisance variable,
closing `ood_test`, and redirecting the next experiment away from residualisation.

At this point I believed the result might collapse into a length shortcut.

## 2026-09-01 — R004, which falsified that

The same analysis on `test`, where there is power: length is at chance within
question (0.494) while the probe is 0.874, and the probe stays at 0.914 on the 52
pairs where the positive is *shorter*. So R003's tie was a property of ood_test's 16
mostly-1v1 questions, not a general fact.

## 2026-09-01 — R005, why the two splits differ

Read the three released v8 builders. Global length filter, per-prompt class
balancing by label quality, then length balancing in global 500-token buckets; no
within-prompt length matching in the code paths inspected. Recorded D012: the
identification limit of the OOD split is itself a result, and the project gets one
more experiment.

## 2026-09-01 — R006, the nuisance control

Score-level, fit on val without labels. Test AUROC 0.909 -> 0.831; almost all of the
cost is `think_logprob`, not length.

## 2026-09-01/02 — R007, the causal test

A CPU preflight showed a 2-val-SD edit is 2.6% of the residual norm, so D013 was
written and the intervention run once. Two OOM aborts first, both self-inflicted:
the full-sequence-logits bug this log records being caught before the R001 run
recurred in the steering script, and the Stage-1 forward was missing `no_grad`.

The experiment itself: baseline reproduces the released labels (0.895 vs 0.000), the
edit lands to 0.158% and propagates, and steering changes termination by ~1 point
while a matched-norm orthogonal direction changes it slightly more. Weak negative,
as D013 froze in advance.

## Where it ended

Seven runs, written up in `notes/writeup.md`. Experimentation is closed.
