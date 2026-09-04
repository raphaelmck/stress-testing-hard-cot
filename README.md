# Stress-testing linear decodability of reasoning termination

**Question:** what does strong cross-domain linear decodability of reasoning termination
actually establish about the underlying model state?

Subject model `Qwen/Qwen3-32B`; data is Task 1 of `Centrattic/cot-proxy-tasks`.

**Start here: [`notes/writeup.md`](notes/writeup.md)** — the report, four claims over seven
runs. The scientific contract is [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Knowledge hierarchy

```
PROJECT_BRIEF  ->  DECISIONS  ->  RESULTS  ->  STATE  ->  artifacts/
   the question    the reasoning   reality      current view   ground truth
```

| File | Read it for | Mutability |
|---|---|---|
| [`notes/writeup.md`](notes/writeup.md) | the report | final |
| [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) | what the project is | stable, human-edited |
| [`DECISIONS.md`](DECISIONS.md) | why the course changed; what is closed | append-only |
| [`RESULTS.md`](RESULTS.md) | every quantitative finding, with the evidence against it | **append-only** |
| [`STATE.md`](STATE.md) | where the project ended | rewritten |

## Layout

```
src/                    library code, one module per experiment
scripts/                entry points; scripts/new_run.sh stamps a run directory
artifacts/runs/<id>/    config.json, metrics.json, metadata.json, stdout.log
artifacts/figures/      report_fig{1,2,3}.png are the report figures
artifacts/tables/       committed tables
notes/                  writeup, dataset audit, literature notes, GPU runbook
cot-proxy-tasks/        upstream clone @ 4482324 -- gitignored, read-only
```

## Setup

Environment is managed with [uv](https://docs.astral.sh/uv/); Python is pinned to 3.12 in
`.python-version` and dependencies are locked in `uv.lock`.

```bash
uv sync                       # creates .venv from the lockfile
uv run python src/inspect_task1.py
```

The upstream data clone is not tracked here. To recreate it:

```bash
git clone https://github.com/Centrattic/cot-proxy-tasks.git
git -C cot-proxy-tasks checkout 4482324b5e4a6277fa3bd544785cbd9875e11694
```

## Reproducing the runs

Two of the seven runs need a GPU: the activation extraction (R001) and the steering
intervention (R007). Both used one A100-80GB on the Mila cluster --- `Qwen/Qwen3-32B` is
65.5 GB in bf16 and the extractor aborts rather than offloading to CPU. Batch scripts are
`scripts/r001_extract.sbatch` and `scripts/r007_steer.sbatch`; the pre-launch verification
gates are in [`notes/gpu_runbook.md`](notes/gpu_runbook.md).

Everything else --- probe fitting, the output-level baselines, the within-question and
length controls, the builder audit, the nuisance control, and all figures --- runs from the
committed artifacts on a CPU:

```bash
uv run python src/fit_task1_probe.py --run-id r001_qwen32b   # needs the activation cache
uv run python src/audit_length_balance.py                    # committed data only
uv run python src/make_report_figures.py                     # rebuilds the report figures
```

The ~210 MB activation cache under `artifacts/runs/r001_qwen32b/activations/` is
gitignored; the per-row scores derived from it are committed.
