# Agent instructions

Canonical operational rules for any coding agent working in this repo (Claude Code, Codex,
Cursor, etc.). `CLAUDE.md` imports this file; do not maintain a second copy of these rules.

This is a time-limited empirical research project, not a software product.
**Optimize for information gained per unit researcher time.**

---

## 1. Read before acting

In this order, every session, before touching anything:

1. `PROJECT_BRIEF.md` — the stable scientific contract (question, hypotheses, protocol, metric).
2. `STATE.md` — where the project is right now and the ONE next experiment.
3. `RESULTS.md` — the latest entries (skim the whole ledger the first time).
4. `DECISIONS.md` — resolved questions; do not reopen these without new evidence.

Then inspect existing artifacts under `artifacts/runs/` before rerunning anything.

If a ticket conflicts with `PROJECT_BRIEF.md`, say so and stop. The brief wins.

## 2. Knowledge hierarchy

```
PROJECT_BRIEF  ->  DECISIONS  ->  RESULTS  ->  STATE  ->  raw artifacts
   defines         records         records      compressed    ground
   the question    reasoning       reality      current view  truth
```

| File | Mutability | Owner |
|---|---|---|
| `PROJECT_BRIEF.md` | rarely changes; human-only edits | research lead |
| `DECISIONS.md` | append-only | agent proposes, lead approves |
| `RESULTS.md` | **append-only** — never rewrite old entries | agent |
| `STATE.md` | rewritten in place, kept under ~100 lines | agent proposes, lead approves |
| `artifacts/runs/**` | write-once | agent |
| `notes/**` | free-form | either |

## 3. Do not

- broaden scope without evidence;
- rerun a completed experiment without stating why the existing artifact is insufficient;
- replace a simple baseline with a complicated method;
- infer conclusions from filenames, directory names, or partial logs;
- overwrite or delete raw outputs in `artifacts/`;
- hide, silently drop, or quietly retry failed experiments;
- tune anything on `ood_test`;
- edit past `RESULTS.md` entries to make the narrative cleaner;
- start a second experiment because the first one is running.

## 4. Per-experiment protocol

1. State the question in one sentence.
2. State what each competing hypothesis (H0 / H1 / H2) predicts for this experiment.
3. Run the **smallest** experiment that can distinguish them.
4. Write machine-readable outputs to `artifacts/runs/<run_id>/` (see §6).
5. Append a `RESULTS.md` entry with the actual numbers, the interpretation, and the
   evidence *against* that interpretation.
6. Propose an updated `STATE.md`.
7. Recommend **exactly one** next experiment, with a decision rule.

Never report an interpretation without also reporting what would falsify it.
Never call linear decodability evidence of causal use without an intervention.
Never call a probe vector "the termination direction".

## 5. Bug protocol

If you find a bug that could have affected results:

1. stop affected runs;
2. determine which past `RESULTS.md` entries are invalid;
3. append an `## R0NN INVALIDATED` block naming the cause and the superseding entry —
   do not edit the original;
4. do not silently regenerate the affected numbers as part of another task;
5. report it before continuing.

## 6. Run artifacts

One directory per run, never reused:

```
artifacts/runs/<run_id>/
    config.json     # every parameter needed to reproduce, incl. model revision, layers, N, seed, split
    metrics.json    # {"ood_auroc": ..., "val_auroc": ..., "n_eval": ..., ...} — numbers only
    metadata.json   # git commit, timestamp, host, exact command line, wall time
    stdout.log
```

`run_id` format: `<experiment>_<short-descriptor>_<NNN>`, e.g. `probe_l56_004`.
`scripts/new_run.sh <run_id>` creates the directory and writes `metadata.json`.

Large tensors (activation caches) go under `artifacts/cache/` and are **not** committed.

## 7. Environment facts

- Upstream data: `cot-proxy-tasks/` is a clone of `Centrattic/cot-proxy-tasks`
  pinned at `4482324`. It is gitignored (~1.1 GB) and **read-only** — never write into it.
- Task 1 lives at `cot-proxy-tasks/datasets/1/`:
  - `prompts/<split>/<question_id>.json` -> `{question_id, prompt_text}`
  - `qwen-3-32b/<split>/<question_id>_rollout_NNN_prefix_N.json`
  - `train` records carry proxy fields (`distance_from_end`, `prefix_words`, `total_words`).
  - `val` / `test` / `ood_test` records carry behavioral fields
    (`yes_count`, `no_count`, `total_resamples`, `mean_yes_position`, `token_length`).
  - These label constructions are **not** equivalent. See `PROJECT_BRIEF.md`.
- Model: `Qwen/Qwen3-32B`. Pin and record the HF revision in every `config.json`.
- Input construction: `prompt_text` + Qwen chat/thinking template + `cot_prefix`.
  Never feed `cot_prefix` alone except as a labelled ablation.
- Activation site: last **real** token of the prefix. Left-padding and right-padding index
  differently — assert the index against an unbatched forward pass.

## 8. Reporting back

End every ticket with:

- files changed;
- validation actually performed (not "should work");
- quantitative results, if any;
- scientific caveats;
- exactly one recommended next step.

If a result contradicts the current plan, prefer changing the plan over defending it.
