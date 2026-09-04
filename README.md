# Reasoning-termination probes — 20h research sprint

**Question:** what information makes reasoning termination linearly predictable across domains?

Subject model `Qwen/Qwen3-32B`; canonical data is Task 1 of `Centrattic/cot-proxy-tasks`.
Full scientific contract: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Knowledge hierarchy

```
PROJECT_BRIEF  ->  DECISIONS  ->  RESULTS  ->  STATE  ->  artifacts/
   the question    the reasoning   reality      current view   ground truth
```

| File | Read it for | Mutability |
|---|---|---|
| [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) | what the project is | stable, human-edited |
| [`DECISIONS.md`](DECISIONS.md) | why we changed course; what is closed | append-only |
| [`RESULTS.md`](RESULTS.md) | every quantitative finding | **append-only** |
| [`STATE.md`](STATE.md) | where we are, the ONE next experiment | rewritten, <~100 lines |
| [`AGENTS.md`](AGENTS.md) | operational rules for coding agents | as needed |

`CLAUDE.md` is a one-line import of `AGENTS.md`. Agent instructions have exactly one home.

## Layout

```
src/                    library code
scripts/                entry points; scripts/new_run.sh stamps a run directory
artifacts/runs/<id>/    config.json, metrics.json, metadata.json, stdout.log
artifacts/figures/      committed figures
artifacts/tables/       committed tables
notes/                  dataset audit, literature notes
cot-proxy-tasks/        upstream clone @ 4482324 — gitignored, read-only
```

## Setup

Environment is managed with [uv](https://docs.astral.sh/uv/); Python is pinned to 3.12
in `.python-version` and dependencies are locked in `uv.lock`.

```bash
uv sync                       # creates .venv from the lockfile
uv run python src/inspect_task1.py
```

The upstream data clone is not tracked by this repo. To recreate it:

```bash
git clone https://github.com/Centrattic/cot-proxy-tasks.git
git -C cot-proxy-tasks checkout 4482324b5e4a6277fa3bd544785cbd9875e11694
```

On the cluster, clone this repo into `$SCRATCH` and follow
[`notes/gpu_runbook.md`](notes/gpu_runbook.md) from step 1.

### Compute

`Qwen/Qwen3-32B` is 32.8B parameters, ~64 GB in bf16. It does **not** fit on the local
48 GB Mac, and quantizing to fit would alter the activations under study. Local work is
limited to analysis, probe fitting, and pipeline validation against a small Qwen3 model;
the real extraction runs on a rented Linux GPU box. `uv.lock` resolves for both
macOS/arm64 and Linux/x86_64, and on Linux the default PyPI torch wheel is CUDA-enabled,
so the same `uv sync` works in both places.

## Running R001 on a GPU

`Qwen/Qwen3-32B` needs an 80 GB card (65.5 GB in bf16). Compute is the **Mila
cluster** -- `--gres=gpu:h100:1` or `--gres=gpu:a100l:1`. Full step-by-step runbook
with the pre-launch verification gates: [`notes/gpu_runbook.md`](notes/gpu_runbook.md).
Batch script: `scripts/r001_extract.sbatch`.

## Working with agents

Give a bounded ticket, not "continue the project". Template in
[`notes/ticket_template.md`](notes/ticket_template.md). After every result, stop and decide
what uncertainty was actually reduced before launching the next agent.
