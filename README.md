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
configs/                run configs
artifacts/runs/<id>/    config.json, metrics.json, metadata.json, stdout.log
artifacts/figures/      committed figures
artifacts/tables/       committed tables
notes/                  dataset audit, literature notes
cot-proxy-tasks/        upstream clone @ 4482324 — gitignored, read-only
```

## Setup

The upstream data clone is not tracked by this repo. To recreate it:

```bash
git clone https://github.com/Centrattic/cot-proxy-tasks.git
git -C cot-proxy-tasks checkout 4482324b5e4a6277fa3bd544785cbd9875e11694
```

## Working with agents

Give a bounded ticket, not "continue the project". Template in
[`notes/ticket_template.md`](notes/ticket_template.md). After every result, stop and decide
what uncertainty was actually reduced before launching the next agent.
