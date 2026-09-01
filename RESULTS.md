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


## R001 — sparse-layer activation-probe reproduction on Qwen3-32B

Commit: `416fe40` (extraction + probe fitting); entry written at the following commit
Run IDs: `r001_qwen32b` (gates: `r001_qwen32b_smoke`, `r001_qwen32b_stress`)
Date: 2026-09-01

Question:
Does a simple L2 logistic probe on Qwen3-32B hidden activations at the last real
prefix token reproduce the published high cross-domain AUROC for imminent `</think>`?

Hypothesis predictions:
- H0 (output-proximal) predicts a strong result here; the discriminating test is
  whether cached output-distribution features match it (embargoed until now, D007).
- H1 (broader latent state) predicts a strong result, and additionally that useful
  signal exists well before the final layers.
- H2 (proxy/dataset structure) predicts a strong result is possible even though
  train labels are distance-based and eval labels are resampling-based.
  **All three predict success here.** R001 is a reproduction gate, not a test
  between hypotheses.

Setup:
- `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137`, bf16, single
  A100-SXM4-80GB (cn-g018), transformers 5.16.1 / torch 2.13.0+cu130.
- Input = released `build_thinking_prompt(prompt_text, cot_prefix)`; no truncation;
  activation site = last real prefix token (`attention_mask.sum(1)-1`), right padding.
- 4,216 rows: 4,000 balanced train (seed 42, 500 yes + 500 no per family, 779
  questions), 72 val, 86 test, 58 ood_test. Worklist sha256 `c261306dde08c8b9`.
- Predeclared depths 8/24/40/56/64 (blocks 7/23/39/55/63), C in {0.01, 0.1, 1, 10}
  selected on **val only** (D005). `ood_test` was never used for any choice.
- CIs: 2,000 bootstrap replicates resampling **question_id clusters** (D006).

Results:

| depth | C | train | val AUROC [95% CI] | test AUROC [95% CI] | OOD AUROC [95% CI] |
|---|---|---|---|---|---|
| 8  | 0.01 | 0.993 | 0.625 [0.539, 0.760] | 0.706 [0.606, 0.797] | 0.699 [0.563, 0.820] |
| 24 | 0.01 | 1.000 | 0.725 [0.582, 0.864] | 0.795 [0.706, 0.883] | 0.832 [0.725, 0.924] |
| 40 | 10   | 1.000 | 0.897 [0.816, 0.985] | 0.909 [0.835, 0.966] | 0.904 [0.834, 0.966] |
| 56 | 10   | 1.000 | 0.896 [0.810, 0.965] | 0.965 [0.927, 0.991] | **0.964 [0.899, 1.000]** |
| 64 | 10   | 1.000 | 0.844 [0.747, 0.936] | 0.907 [0.859, 0.951] | 0.892 [0.806, 0.961] |

Max OOD AUROC across the five predeclared depths = **0.964 at depth 56**.
The val-selected depth is 40 by a hair (val 0.8966 vs 0.8958 at depth 56), whose OOD
AUROC is 0.904. Both are far above the 0.85 reproduction bar, and the two depths are
not separable given the CIs.

Pre-run gates (job 10621128, artifacts `r001_qwen32b_smoke` / `_stress`):
all 32.76B parameters CUDA-resident, 19.1 GB free after load; hooked final depth +
final norm == the model's own `last_hidden_state` (`max_abs_diff = 0`); no
full-sequence logits materialised; padded-vs-singleton worst cosine 0.999917; longest
example (16,652 tokens) ran alone at 68.8 GB peak with 14.4 GB free. Extraction wall
time 53.7 min.

Interpretation:
The Hard-CoT phenomenon reproduces cleanly and is not an artifact of a favourable
layer choice: three of five predeclared depths clear 0.89 OOD. The depth profile is
informative — decodability is near chance-plus at depth 8, climbs through the middle
of the network, peaks at depth 56, and *falls back* at the final depth 64. That the
peak is not the last layer is at least consistent with the probe reading something
other than the immediate output logit state, which would be expected to be maximal at
the end.

Evidence against that interpretation:
- **The decisive comparison has not been run.** H0 predicts exactly this table. The
  D007 output-distribution features were cached in the same pass and remain unanalysed
  as of this entry; until they are fit, nothing here separates H0 from H1.
- Depth 64 activations still reach 0.892, so "late layers are worse" is a small
  effect on wide, overlapping CIs — the 56-vs-64 gap (0.072) is within the noise this
  design can resolve (32 ood questions, 16 contributing both labels).
- The OOD CI reaches 1.000; 58 rows from 32 questions cannot distinguish 0.96 from
  0.93 or from 0.99. Do not report 0.964 as a precise quantity.
- Train AUROC saturates at 1.000 for four of five depths: the distance-based train
  labels are trivially separable, so the probe is fit on an easier problem than the
  one it is evaluated on. H2 remains live.
- val < test/ood at every depth. The audit predicted this from label purity
  (val/test 40–50/50 vs ood 45–50/50), but it also means the selection split is the
  noisiest one, so C and layer selection are made on the weakest signal.
- Activations are bf16 and batch-composition dependent (padded-vs-singleton cosine
  0.99992, max_abs_diff up to 8.0 at depth 64 where residual norms are ~1e2). Far
  below the signal, but not bit-reproducible across a different batching.
- Linear decodability is not causal use. Nothing here licenses any claim about the
  termination mechanism, and no vector in this run is "the termination direction".

Decision:
Max OOD AUROC 0.964 >= 0.90, so by the `STATE.md` rule the reproduction is adequate
and **probe optimisation stops here**. Committing this entry lifts the D007 embargo by
its own terms. Next: fit the identical probe pipeline to the cached output-level
features alone and compare OOD AUROC against 0.964 (see `STATE.md` for the rule).

Artifacts:
- `artifacts/runs/r001_qwen32b/` (config, metrics, metadata, sample_manifest,
  probe_scores.csv, stdout.log; activations under `activations/`, not committed)
- `artifacts/runs/r001_qwen32b_smoke/`, `artifacts/runs/r001_qwen32b_stress/`
- `artifacts/tables/reproduction_layer_auroc.csv`
- `artifacts/figures/reproduction_layer_auroc.png`
