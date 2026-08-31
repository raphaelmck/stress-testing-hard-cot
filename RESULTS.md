# Results ledger

Append-only. One entry per meaningful experimental run. Newest entries go at the bottom.

**Never edit or delete a past entry.** If a result turns out to be wrong, append an
`## R0NN INVALIDATED` block naming the cause and the entry that supersedes it.

Entries are numbered `R001`, `R002`, ... An entry exists only if an empirical scientific
result was produced. Infrastructure tickets do not get entries.

---

## Entry template

```markdown
## R0NN — short title

Commit: <sha>
Run IDs: <run_id>, ...
Date: YYYY-MM-DD

Question:
One sentence.

Hypothesis predictions:
- H0 predicts ...
- H1 predicts ...
- H2 predicts ...

Setup:
- model + revision
- N_train, split, balance
- layers, activation site
- selection procedure (what was chosen on which split)

Results:
| layer | val AUROC | test AUROC | OOD AUROC [95% CI] |
|---|---|---|---|

Interpretation:
...

Evidence against interpretation:
...

Decision:
...

Artifacts:
- artifacts/runs/<run_id>/
- artifacts/figures/...
```

---

*No results yet. First planned entry: R001 — sparse-layer reproduction (see `STATE.md`).*
