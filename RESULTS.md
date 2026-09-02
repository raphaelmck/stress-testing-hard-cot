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

---

## R003 — preregistered within-question control on ood_test (D008)

Commit: entry written at the following commit; run produced under `0642469` + `src/paired_question_analysis.py`
Run IDs: `r003_paired_ood` (no refitting, no GPU)
Date: 2026-09-01

Question:
Do the depth-40 activation scores separate imminent-termination prefixes from
late-termination prefixes *within the same question*, where prompt, domain and
usually the already-obtained answer are held fixed?

Hypothesis predictions:
- H0 predicts activation and `think_logprob` separate to a similar degree and move
  together across questions.
- H1 predicts activation separates within question and beats `think_logprob`.
- H2 predicts that whatever separates the classes within question is generic
  "depth into the trace" structure rather than a termination-specific state.

Setup:
- 58 ood_test rows, 32 questions, **16 carrying both labels** (as preregistered).
  Most contribute exactly one YES and one NO row, so per-question concordance is
  coarse (0, 0.5 or 1).
- Nothing was refit. Scores are the frozen R001 depth-40 probe outputs (D010: the
  val-selected depth), `think_logprob` from the cached D007 features, and the
  released `token_length` as the sanity baseline.
- A_q = fraction of YES×NO pairs within q ranked correctly, ties 0.5; macro-averaged
  over the 16 questions. CIs and deltas from a 2,000-replicate bootstrap resampling
  **questions**, paired across scores.

Results:

| score | macro paired concordance [95% CI] | questions with Δ_q > 0 |
|---|---|---|
| depth-40 activation probe | 0.938 [0.812, 1.000] | 15/16 |
| `think_logprob` | 0.844 [0.656, 1.000] | 13/16 |
| **`token_length` (sanity baseline)** | **0.938 [0.844, 1.000]** | **16/16** |

Paired question-bootstrap deltas:

| comparison | Δ | 95% CI | P(Δ>0) |
|---|---|---|---|
| activation − `think_logprob` | +0.094 | [+0.000, +0.250] | 0.882 |
| **activation − `token_length`** | **+0.000** | **[−0.188, +0.125]** | 0.441 |

Sign/correlation diagnostics over the 16 questions:
- activation Δ_q > 0 on 15/16, two-sided binomial p = 0.0005;
- Spearman(Δ_activation, Δ_think) = +0.385, p = 0.141;
- questions where the output cue is wrong or uninformative (Δ_think ≤ 0): **3**
  (`josephus_7`, `crt_three`, `dice_even_product`); activation is still positive on
  2 of those 3, mean activation concordance there 0.667.

Interpretation:
The activation probe does separate YES from NO within question — 15/16 questions,
sign-test p = 0.0005 — so the headline OOD result is **not** merely between-question
topic structure, and it is not reducible to immediate `</think>` propensity within a
question either (activation is positive on 2 of the 3 questions where the output cue
points the wrong way, and beats it by +0.094 overall, though that delta is not
resolved: P(Δ>0) = 0.88).

But the sanity baseline is the finding. **Raw `token_length` matches the activation
probe exactly (0.938 vs 0.938, paired Δ = 0.000 [−0.188, +0.125]) and is positive on
16/16 questions** — within a question, the YES prefix is simply the longer one, by
100–1,250 tokens. The within-question control that was designed to remove topic
confounds does not remove, and probably concentrates, a trivial depth-into-the-trace
cue. H2 is substantially strengthened: this experiment provides no evidence that the
probe carries information beyond "how far into this trace are we", because the
cheapest possible feature does equally well here.

Note this does not contradict the audit's pooled length result (ood AUROC 0.584).
Pooled across questions, absolute length is uninformative because questions differ in
scale; *within* a question it is nearly decisive. The two facts together say the
length cue is conditional-on-question, which is exactly the regime this control
created.

Evidence against that interpretation:
- 16 questions, most contributing a single YES/NO pair, so each A_q is essentially a
  coin-flip-resolution 0/1. The CIs reach 1.000 and the activation−length delta
  spans [−0.19, +0.13]: this design cannot distinguish "equal" from "modestly
  better" or "modestly worse". Treat the tie as *no evidence of an advantage*, not
  as evidence of equality.
- A length cue is not automatically an artifact. Later prefixes really are closer to
  termination, so length is a legitimate — if trivial — predictor of the label. The
  finding constrains what the probe can be claimed to know, not whether the label is
  meaningful.
- The activation representation is taken at the last prefix token of a sequence whose
  length varies by 1,000+ tokens; positional information is available to the probe
  by construction. That this is *how* it wins is untested here.
- The three Δ_think ≤ 0 questions are three questions. 2/3 is not a result.
- `token_length` is the released field, not a re-measurement, and pairs it with rows
  whose labels came from resampling; no part of this run re-derives either.
- Everything here is ood_test, 58 rows. It has now been looked at three times
  (R001 evaluation, R002 deltas, R003). Further inspection of this split should stop.

Decision:
The planned residualization is now the wrong next experiment against the wrong
nuisance variable. **R004 = the length control**, replicated on `val` and `test`
(30 and 22 questions, more multi-row questions, and neither is the final OOD split)
so the comparison has power: within-question concordance for the activation probe,
`think_logprob` and `token_length`, plus the probe's within-question concordance
restricted to length-matched or length-residualized comparisons. The question R004
answers: does the activation probe know anything about imminent termination that
prefix length does not already say? See `STATE.md` for the decision rule.

Artifacts:
- `artifacts/runs/r003_paired_ood/` (config, metrics, metadata, per_question.csv)
- `artifacts/figures/r003_paired_questions.png`
- `src/paired_question_analysis.py`

---

## R003 — CORRECTION (wording only; no number changes)

Date: 2026-09-01

Two statements in the R003 entry above are too strong. The numbers stand; the
original entry is left intact per the append-only rule.

1. R003 controls **question / prompt / topic**, not the reasoning trajectory. The
   paired YES and NO prefixes within a question generally come from *different
   rollouts*, so it is not a same-trace matched-pair test.
2. "Positional information is available to the probe by construction" is wrong for
   this model. Qwen3 uses rotary position embeddings applied inside attention, not
   an absolute positional vector added into the residual stream. Whether the depth-40
   residual carries usable absolute-position or progress information is an empirical
   question, not a structural given — and that makes the length result *more*
   interesting, since anything the probe reads about progress it had to build.

---

## R004 — the length control on `test` (primary) and `val` (supplementary)

Commit: entry written at the following commit; run produced under `fdf2cc1` + `src/length_control.py`
Run IDs: `r004_length_control` (no refitting for A–C; no GPU)
Date: 2026-09-01

Question:
When prefix length orders a within-question YES/NO pair the *wrong* way, does the
frozen depth-40 activation probe still order it correctly?

Hypothesis predictions:
- H0 predicts `think_logprob` tracks the activation probe on the discordant pairs.
- H1 predicts the probe stays accurate where length fails.
- H2 (in its cheap form, "the probe reads depth into the trace") predicts the probe
  collapses toward 0.5 exactly where length is discordant or matched.

Setup:
- **Primary confirmatory split: `test`.** `val` is supplementary because depth 40 was
  selected on val (D010) and cannot carry a depth-40-vs-length claim. `ood_test` is
  closed (D011) and the script refuses it.
- Nothing refit for A–C: frozen R001 depth-40 scores, released `token_length`,
  `think_logprob` from the cached D007 pass.
- Pair-level analysis over all within-question YES×NO pairs; concordance macro-averaged
  within question then across questions; 2,000-replicate question bootstrap, paired
  across scores. Match thresholds |Δlen| ≤ 100 and ≤ 250 were frozen before the run.

Results — A, aggregate within-question concordance:

| split | questions | pairs | activation | `think_logprob` | `token_length` |
|---|---|---|---|---|---|
| **test** (primary) | 17 | 137 | **0.874** [0.727, 0.977] | 0.794 [0.664, 0.908] | **0.494** [0.321, 0.667] |
| val (supplementary) | 11 | 97 | 0.907 [0.801, 0.997] | 0.741 [0.485, 0.968] | 0.653 [0.425, 0.864] |

Paired deltas (test): activation − `think_logprob` [−0.070, +0.223], P(Δ>0) = 0.864;
activation − `token_length` **[+0.137, +0.605], P(Δ>0) = 0.998**.

Results — B, the primary diagnostic (length-discordant pairs, YES shorter than NO):

| split | discordant pairs | questions | activation pooled / macro [CI] | questions > 0.5 | `think_logprob` pooled / macro |
|---|---|---|---|---|---|
| **test** | 52 of 137 (0 tied) | 14 | **0.923 / 0.914 [0.814, 1.000]** | **12/14** | 0.635 / 0.723 |
| val | 30 of 97 | 6 | 0.933 / 0.944 [0.833, 1.000] | 6/6 | 0.833 / 0.722 |

Results — C, near-length-matched (thresholds frozen in advance):

| split | |Δlen| ≤ 100 | |Δlen| ≤ 250 |
|---|---|---|
| test | 23 pairs / 10 questions — activation **0.983**, think 0.567 | 49 pairs / 12 questions — activation **0.986**, think 0.707 |
| val | 20 pairs / 5 questions — activation 1.000, think 0.975 | 44 pairs / 8 questions — activation 0.938, think 0.782 |

Results — D, secondary: with the nuisance line fit on `val` only
(score ≈ −0.0111 × token_length + 25.41), the residualised probe scores **0.918**
[0.794, 0.995] macro on test and 0.864 on the discordant pairs. This removes the
fitted *marginal* association between one scalar and length; it does not remove
length information from the representation.

Interpretation:
This is interpretation band 1, decisively on the primary split. **Prefix length does
not explain the probe.** Within question on `test`, length is at chance (0.494) while
the probe is at 0.874; on the 52 pairs where length points the wrong way the probe is
still right 92% of the time and beats chance on 12 of 14 questions; on the 23 pairs
whose lengths are within 100 tokens it is at 0.983. `think_logprob` degrades exactly
where the probe does not (0.635 pooled on discordant pairs, 0.567 on near-matched),
so the surviving signal is not immediate `</think>` propensity either.

The R003 result therefore does not generalise: length's apparent parity with the probe
on `ood_test` was a property of that split's 16 mostly-one-versus-one questions, not a
general fact. The cheap form of H2 — "the probe reads how far into the trace we are" —
is not supported on the split with power.

Evidence against that interpretation:
- `test` is an in-distribution split. It shares question families with `train`, so
  R004 shows the probe beats length *within domain*. The cross-domain claim still
  rests on `ood_test`, where this comparison came out tied, and that split is closed.
  A confounded-on-ood / clean-on-test pattern is not fully resolved by "ood is small".
- 137 pairs come from 17 questions and are not independent; the macro average and the
  question bootstrap handle the weighting but cannot manufacture degrees of freedom.
  The activation − `think_logprob` delta on test still spans zero ([−0.070, +0.223]).
- Length being at chance within question on `test` is itself unexplained. It is not
  what R003's ood pattern predicted, and no mechanism here says why the two splits
  differ. Until that is explained, treat "length is irrelevant" as split-specific.
- `token_length` is one operationalisation of progress. A within-question chance
  result for absolute length does not rule out a *question-relative* progress
  variable, which is the version of H2 that would actually explain a probe that
  generalises across unseen questions. R004 does not test that.
- D is a two-parameter marginal correction, not a representational one, and its
  nuisance line was fit on val where the length–score relation may differ.
- Depth 40 is the val-selected depth; no other depth was examined here, and should
  not be without a new DECISIONS entry.

Decision:
Band 1 → length is not sufficient, H1 and the refined form of H2 both remain open.
**R005 = separate "termination-specific state" from "question-relative reasoning
progress".** The cheapest version: build a question-relative progress variable from
the released fields (e.g. prefix length as a fraction of the rollout's own total, or
rank of the prefix within its question), test it the same way on `test`, and check
whether the probe still wins on pairs where *that* variable is discordant. Only if it
does should activation-level residualization (the originally planned R005) be run.

Artifacts:
- `artifacts/runs/r004_length_control/` (config, metrics, metadata, pairs.csv)
- `src/length_control.py`

---

## R005 — benchmark-construction audit: global vs question-conditional length balance

Commit: entry written at the following commit; run produced under `da17c21` + `src/audit_length_balance.py`
Run IDs: `r005_length_balance_audit` (nothing fit, no GPU, no new predictor)
Date: 2026-09-01

Question:
The released evaluation builders balance token length. R003 found length ordering
ood_test labels almost perfectly within question while being weak pooled. Can the
balancing procedure actually leave that association, and does it in every split?

Setup:
- Builder logic read directly from the pinned upstream clone (`4482324`):
  `src/tasks/reasoning_termination/run_build_eval_v8.py`, `…_math_val_v8.py`,
  `…_ood_val_v8.py`. All three share the same structure.
- Split statistics computed from the released rollout records' own
  `token_length` and `label` fields only. No individual ood_test example inspected
  (D011). Activation numbers are the frozen R003/R004 values, not recomputed.

What the builders do (verified in the source, not inferred):
1. **Global length filter** first: keep `500 <= token_count < 3000` (`LENGTH_MIN`,
   `LENGTH_MAX`), before any balancing.
2. **Per-prompt class balancing**: for each prompt take `min(n_yes, n_no)` of each,
   sorted by `mean_yes_position` (yes) and `no_count` (no) — i.e. by *label
   quality*, never by token length. Single-class prompts contribute one row.
3. **Length balancing in global 500-token buckets** (steps 5a–5e): trim unpaired
   singles, remove whole pairs, add minority singles, stratified per-bucket trim of
   the majority class, then a global rebalance. `_bucket_stats` iterates the flat
   selected list; **no step groups buckets by prompt**.
4. **No within-prompt length matching exists** in any of the three builders — no
   step compares a yes item's `token_count` against a no item's from the same prompt.
5. The skew was known: step 5d's own comment is "stratified bucket trim — remove
   majority-class excess per bucket to eliminate the systematic length skew
   (no→short, yes→long)". The correction is applied globally.

Results:

| split | rows | questions (both labels) | pairs | pooled length AUROC | within-question length concordance | frozen depth-40 probe, same metric | mean / median within-question Δlen (YES−NO) |
|---|---|---|---|---|---|---|---|
| val | 72 | 30 (11) | 97 | 0.450 | 0.653 | 0.907 (R004) | +226 / +168 |
| test | 86 | 22 (17) | 137 | 0.499 | **0.494** | 0.874 (R004) | +207 / +163 |
| ood_test | 58 | 32 (16) | 31 | 0.587 | **0.938** | 0.938 (R003) | **+432 / +426** |

Global 500-token buckets are close to balanced in every split (e.g. ood_test
[500–1000) 10 yes / 12 no, [1000–1500) 11 / 12, [1500–2000) 5 / 3, [2000–2500) 3 / 2;
global mean length yes 1259 vs no 1136). The balancing the builder enforces did
what it says.

Deterministic toy example (arithmetic only, not evidence): one long question
contributing 1 YES at 3000 and 3 NOs at 2900, one short question contributing 3 YES
at 1000 and 1 NO at 900 gives **pooled length AUROC 0.438 with within-question
concordance 1.000** in both questions. Unequal per-question class counts across
different length scales are sufficient; no adversarial construction is needed.

Interpretation:
Global bucket balance and a question-conditional length–label association are
compatible, and in `ood_test` both hold at once: buckets are balanced, pooled AUROC
is 0.587, and length still orders the labels within question at 0.938. The mechanism
is legible in the numbers — the mean within-question gap in ood_test is **+432
tokens, smaller than the 500-token bucket width**, so a procedure that balances in
500-token buckets cannot see it by construction.

The supportable claim is therefore narrow: *global token-length balance is not
sufficient to rule out a question-conditional length shortcut, and in the released
OOD math set absolute length is weak pooled across questions but almost perfectly
orders termination labels within questions.* Consequently a high pooled OOD AUROC
alone does not establish that a probe's signal transfers across domains independently
of reasoning progress — the split is not constructed to separate those.

Evidence against that interpretation, and what this does **not** show:
- **The conditional confound is not a general property of the released splits.** On
  `test` the within-question length concordance is 0.494 — chance — and on `val`
  0.653. Only ood_test shows it. Any statement of the form "the benchmark is
  length-confounded" is wrong as written; the correct scope is this split.
- ood_test's estimate rests on **31 pairs from 16 questions**, most contributing one
  YES and one NO. `test` has 137 pairs from 17 questions. The split showing the
  confound is the split with the least data to establish it.
- This says nothing about whether the probe uses the confound. R004 shows it does
  not need to: on `test` the probe scores 0.874 within question where length is at
  chance, and 0.914 on the 52 pairs where length points the wrong way.
- Not concluded, and contradicted by R004: that the activation probe is a length
  detector; that the published probe result is invalid; that length explains R001.
- The audit reads the builder source, not the authors' full pipeline or any
  unreleased step. "Conditional length confound left by global balancing" is the
  accurate description; nothing here establishes a bug.
- Pooled length AUROC differs across splits (0.450 / 0.499 / 0.587) and ood_test's
  0.587 matches the earlier audit's 0.584 on a different length measure, so the
  pooled quantity is stable; the conditional one is what varies.

Decision:
The identification limit is now itself a result: **the released OOD split cannot
cleanly separate a termination-specific state from reasoning progress**, while the
in-domain `test` split can and favours the probe carrying more than length or
`</think>` propensity. One targeted experiment remains worth the budget —
R006, a representation-level control on `test`: does the depth-40 score retain
held-out predictive signal after removing its marginal linear associations with both
`think_logprob` and `token_length`? Framed narrowly, not as "removing termination
propensity". After that, stop experimenting and write up.

Artifacts:
- `artifacts/runs/r005_length_balance_audit/` (config, metrics, metadata)
- `artifacts/figures/r005_length_balance.png`
- `src/audit_length_balance.py`

---

## R005 — CORRECTION (wording only; no number changes)

Date: 2026-09-01

The R005 entry says ood_test's mean within-question gap is "+432 tokens, smaller
than the 500-token bucket width, so a procedure that balances in 500-token buckets
cannot see it by construction". That is stronger than the evidence supports — the
bucket width is a plausible explanation, not a demonstrated mechanical cause.

Replace with: *global 500-token bucket balancing does not constrain within-prompt or
within-bucket label–length differences, so a +432-token conditional skew can survive
despite good global balance.*

---

## R006 — score-level nuisance control on `test` (final experiment)

Commit: entry written at the following commit; run produced under `e204966` + `src/nuisance_control.py`
Run IDs: `r006_nuisance_control` (nothing refit, no GPU)
Date: 2026-09-01

Question:
Does the frozen depth-40 probe score retain held-out predictive information after
removing its fitted linear association with absolute prefix length and with current
`</think>` log-probability?

This is a **score-level** nuisance control. It modifies one scalar. It does not
project anything out of the 5,120-dimensional activation.

Setup:
- Nuisance fit on `val` only, using **no labels**: OLS of the frozen score `s` on
  `z_val(token_length)` and `z_val(think_logprob)`. Coefficients and standardisation
  constants frozen, then applied to `test`:
  `s_resid = s − β_L·z_val(L) − β_T·z_val(T)`. No reorientation or calibration on
  test labels. `ood_test` closed (D011).
- Val fit: `s = 15.000 − 4.245·z(L) + 8.425·z(T)`, **R² = 0.334**
  (length-only R² = 0.098, think-only R² = 0.274).
- Pooled AUROC CIs are question-clustered (D006); concordance CIs and all deltas use
  paired question bootstraps, 2,000 replicates.

Results on `test`:

| score | pooled AUROC [95% CI] | within-question concordance | Δ vs raw (AUROC) | Δ vs raw (concordance) |
|---|---|---|---|---|
| raw depth-40 | 0.909 [0.835, 0.966] | 0.874 | — | — |
| **joint residual (primary)** | **0.831 [0.716, 0.928]** | 0.818 | +0.077 [−0.006, +0.173] | +0.056 [−0.132, +0.240] |
| length-only residual | 0.882 [0.779, 0.954] | **0.918** | +0.027 [−0.025, +0.086] | −0.044 [−0.191, +0.059] |
| think-only residual | 0.841 [0.717, 0.934] | 0.868 | +0.068 [−0.008, +0.173] | +0.006 [−0.096, +0.113] |

Against the band frozen before the run (raw reference 0.909): the joint residual
drops **0.077**, i.e. the middle band — *the nuisances explain part but not all of
the signal*.

Interpretation:
The frozen probe score is partly but not mostly a linear function of these two
nuisances: they explain R² = 0.33 of it on val, and removing both costs 0.077 AUROC
on held-out `test`. What survives — 0.831 pooled, 0.818 within question — remains far
above the `think_logprob` baseline on the same split (0.695, R002) and far above
chance. Almost all of the cost comes from `think_logprob`, not length: removing
length alone costs 0.027 and actually *raises* within-question concordance to 0.918,
which is consistent with R004's finding that absolute length is not what the probe
uses within a question.

Evidence against that interpretation:
- Every delta CI includes zero (joint: [−0.006, +0.173]). The direction is
  consistent across all three variants but the magnitude is not resolved by 86 rows
  from 22 questions.
- The control is linear and marginal. It says nothing about nonlinear encodings of
  length, reasoning progress, or output propensity, and a residual that survives a
  linear control can still be a nonlinear function of the same nuisances.
- The nuisance coefficients were fit on `val` (72 rows), which is small and is the
  split the depth was selected on. A different fit split could give a different β.
- One depth, one split, one probe. `test` is in-domain; nothing here extends the
  claim across domains, and the split that would (ood_test) is closed and confounded.
- **This does not establish a termination-specific latent state.** It establishes
  robustness to two fitted linear nuisance associations, and nothing more.

Decision:
Per D012, experimentation stops here. R001–R006 are the result set; the writeup
reports them together with the identification limit as the central caveat.

Artifacts:
- `artifacts/runs/r006_nuisance_control/` (config, metrics, metadata)
- `src/nuisance_control.py`
