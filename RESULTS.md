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

---

## R002 — output-level (H0) baselines from the cached D007 features

Commit: entry written at the following commit; run produced under `7a0123f` + `src/fit_output_baselines.py`
Run IDs: `r002_output_baseline` (features cached during `r001_qwen32b`; no new forward pass)
Date: 2026-09-01

Question:
How much of R001's cross-domain activation-probe performance is reducible to the
model's immediate next-token propensity to emit `</think>` at the same position?

Hypothesis predictions:
- H0 predicts an output-level score approaches the activation probe's OOD AUROC.
- H1 predicts the activation probe retains a substantial advantage over it.
- H2 is largely orthogonal here, but predicts nothing forces the train-label
  construction and the eval-label construction to agree.

Setup:
- Comparator is the **val-selected** activation depth, **40, OOD 0.904** (D005).
  R001's descriptive maximum (depth 56, 0.964) is *not* used: it was identified by
  looking at OOD, and comparing it against a val-selected baseline would give the
  activation side a post-hoc advantage.
- Features are the D007 values cached at the same last-real-prefix token during the
  R001 pass: `think_logit`, `think_logprob`, `think_margin`, `next_token_entropy`,
  `top1_logprob`, and `top1_is_think = (top1_token_id == 151668)`.
  `top1_token_id` is never used as a number — token ids have no ordinal meaning.
- Scalars are used directly as the score for `label=yes`; orientation is fixed on
  val, never on OOD (all six came out "as-is").
- Multivariate baseline: standardizer + L2 logistic on all six, fit on **train only**,
  C in {0.01, 0.1, 1, 10} selected on **val only** — exactly the R001 procedure.
- CIs: 2,000 question-clustered bootstrap replicates (D006). Deltas use a **paired**
  bootstrap: the same resampled clusters score both models.

Results:

| baseline | train | val | test [95% CI] | OOD [95% CI] |
|---|---|---|---|---|
| activation probe, depth 40 (R001) | 1.000 | 0.897 | 0.909 [0.835, 0.966] | **0.904 [0.834, 0.966]** |
| `think_logprob` (primary scalar) | 0.507 | 0.771 | 0.695 [0.603, 0.806] | **0.807 [0.696, 0.905]** |
| `think_margin` | 0.508 | 0.762 | 0.684 [0.591, 0.790] | 0.803 [0.694, 0.902] |
| `think_logit` | 0.498 | 0.529 | 0.601 [0.484, 0.724] | 0.577 [0.441, 0.703] |
| `next_token_entropy` | 0.503 | 0.505 | 0.464 [0.279, 0.639] | 0.578 [0.442, 0.722] |
| `top1_logprob` | 0.495 | 0.522 | 0.608 [0.428, 0.784] | 0.448 [0.311, 0.582] |
| `top1_is_think` | 0.500 | 0.500 | 0.500 | 0.500 (degenerate, see below) |
| multivariate output-only (C=0.01) | 0.512 | 0.707 | 0.561 [0.443, 0.692] | 0.727 [0.607, 0.843] |

Paired question-clustered deltas, activation(depth 40) − output:

| comparison | test | ood_test |
|---|---|---|
| vs `think_logprob` | +0.214 [+0.091, +0.311] | **+0.096 [+0.000, +0.201]**, P(Δ>0)=0.975 |
| vs multivariate output-only | +0.348 [+0.211, +0.470] | +0.177 [+0.064, +0.291], P(Δ>0)=0.9995 |

Two facts about the features themselves, both consequential:

1. `top1_is_think` is **identically zero on all 4,216 rows** — the argmax next token
   is never `</think>` at any prefix in this dataset. Mean p(`</think>`) at the next
   position is ~2e-12 (mean logprob ≈ −27 to −36 by split).
2. Every output feature is at chance on **train** (AUROC 0.495–0.508), while the
   same features reach 0.77–0.81 on val/ood. The train labels are distance-based
   ("25–55 words from the end" vs "300+"), and being 40 words from the end evidently
   says almost nothing about whether the *very next token* is `</think>`.

Interpretation:
The output-proximal explanation is a large part of the story but does not close it.
A single untrained scalar — the log-probability the model assigns to `</think>` at
the next position — reaches OOD AUROC 0.807 against the probe's 0.904, i.e. it
recovers most of the distance from chance. Under the preregistered bands, the paired
delta of +0.096 [+0.000, +0.201] sits in the **0.05–0.15 "meaningful unexplained
activation signal"** band, with the CI's lower edge at zero: H0 is substantially
supported, and not sufficient. Notably the surviving signal is *not* explained by the
probe having learned output propensity, because the labels it was trained on are
uncorrelated with that propensity (train AUROC 0.507).

Evidence against that interpretation:
- The delta's 95% CI touches 0.000 and P(Δ>0) is 0.975 — one or two questions'
  worth of resampling separates "activation advantage" from "no advantage". With 32
  OOD clusters this experiment cannot resolve a 0.10 difference confidently.
- The multivariate output baseline is **handicapped and should not be read as the
  strong H0 baseline**: it is fit on train labels its features cannot predict
  (train AUROC 0.512), so C=0.01 shrinks it toward noise and it lands *below* the
  untrained scalar (0.727 vs 0.807). The honest best output baseline here is the
  scalar. A properly trained output baseline would need labels of the evaluation
  kind, which D005 forbids fitting on — this asymmetry favours the activation side
  and is not resolved by this run.
- `think_logprob`'s advantage over `think_logit` (0.807 vs 0.577) means the useful
  quantity is normalised by the full partition function, so "the `</think>` logit"
  as such is not the carrier; the softmax denominator matters.
- The absolute probabilities are ~1e-12. Calling this an "output-level precursor" is
  a statement about the ordering of a far tail, not about the model being close to
  terminating. What the ordering of that tail reflects is itself unexplained.
- test and ood_test disagree in level for the output scalars (0.695 vs 0.807) while
  the activation probe is flat (0.909 vs 0.904). The splits differ in label purity,
  so some of this is label noise, but it means the output baseline's OOD number is
  the more favourable of the two available estimates.
- Depth 56 vs 64 in R001 is still not evidence against H0: representations can
  reorganise between layers while the LM head preserves the relevant information.
  This run does not test that.

Decision:
Under the R002 bands: activation advantage ≈ 0.096 → **test conditional/residual
information next**, not D008, and not another activation probe. R003 = residualise
the depth-40 activations against the output-level features (regress the output score
out of the probe input, or fit the probe on the residual and on
[activations ⟂ output features]) and re-evaluate OOD with the same paired bootstrap.
The question R003 answers: does the activation advantage survive removing what the
output distribution already knows?

Artifacts:
- `artifacts/runs/r002_output_baseline/` (config, metrics, metadata, baseline_scores.csv)
- `artifacts/tables/output_baseline_auroc.csv`
- `src/fit_output_baselines.py`
