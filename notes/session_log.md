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

## Next

R001 has not been run. Nothing in `RESULTS.md` yet.
