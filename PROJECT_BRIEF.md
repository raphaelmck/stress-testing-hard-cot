# PROJECT_BRIEF.md

## Project

**What information makes reasoning termination linearly predictable across domains?**

This is a short empirical research project for a Neel Nanda MATS application. The hard research-time budget is approximately 20 hours, so prioritize rapid falsifiable experiments, simple baselines, careful controls, and interpretable results over methodological complexity.

## Starting observation

Recent Hard-CoT interpretability work reports that simple linear probes on Qwen3-32B hidden activations can predict whether the model will terminate its chain of thought soon with >0.90 OOD AUROC.

Task 1 of the released `Centrattic/cot-proxy-tasks` repository is the canonical dataset and evaluation setting.

The subject model is:

`Qwen/Qwen3-32B`

The target variable is whether the model will emit `</think>` soon after a supplied chain-of-thought prefix.

A separate recent causal analysis of Qwen3 reasoning termination suggests that the actual stopping mechanism is high-dimensional and localized substantially to late-model computation; a single residual-stream direction does not appear to constitute the termination mechanism.

This creates the central puzzle:

> Why is imminent termination so linearly decodable across domains if the causal termination mechanism is not itself a simple one-dimensional feature?

## Primary research question

**What information makes Qwen3-32B's imminent reasoning termination linearly predictable
across domains?**

Deliberately not "find the termination direction" or "find the termination circuit".

What information is a successful OOD termination probe actually decoding?

In particular, distinguish among:

1. an immediate output-level precursor to `</think>`;
2. a broader domain-general latent state such as reasoning sufficiency / answer commitment / termination readiness;
3. dataset or proxy-label structure correlated with being near the end of reasoning.

## Hypotheses

**FROZEN 2026-08-31**, before any activation was extracted. H0/H1/H2 below are not to be
edited, reworded, or added to after seeing probe results; a new hypothesis gets a new label
and a `DECISIONS.md` entry recording when and why it was introduced.


### H0 — output-proximal explanation

Most of the linear probe's predictive power is reducible to information already present in the model's immediate output distribution, especially its propensity to emit `</think>`.

Operational prediction:

* `</think>` logit/probability and simple output-distribution features approach the OOD performance of hidden-state probes;
* removing or controlling for this information removes most of the apparent activation advantage.

### H1 — broader latent-state explanation

Hidden activations contain domain-general information about a reasoning state associated with imminent termination that is not reducible to immediate `</think>` propensity.

Operational predictions may include:

* hidden-state probes substantially outperform output-level baselines;
* useful signal exists before the very final layers;
* probe performance remains after controlling for obvious output-level signals.

H1 does **not** imply that the true termination mechanism is one-dimensional.

### H2 — proxy/dataset explanation

Training and evaluation share domain-general correlates of "being near a conclusion" that make termination predictable without corresponding cleanly to either a causal stop mechanism or a unified latent termination state.

This is particularly plausible because the released training and evaluation labels are constructed differently.

## Important dataset detail

Do not erase this distinction.

The released training set uses cheap distance-based proxy labels:

* positive examples are prefixes approximately 25, 35, 45, or 55 **words** from the end of an existing CoT;
* negative examples are prefixes 300+ words from the end.

Empirically verified in the released files (`src/inspect_task1.py`, 2026-08-31): train
positives are `distance_from_end` in {25, 35, 45, 55} and negatives form a ladder from 300 to
11,300 words in steps of 200 — "300+" is correct but hides that negatives span the full depth
of a trace. `total_words - prefix_words == distance_from_end` holds for all 46,160 rows.

The stronger evaluation datasets use behavioral labels obtained by resampling continuations:
50 per prefix, with positives requiring `</think>` soon and negatives requiring it late or
never.

**Do not describe all released evaluation splits as using a 45/50 threshold.** The observed
empirical thresholds differ by split:

| split | positives | negatives |
|---|---|---|
| val | `yes_count` 40–50 | `no_count` 40–50 |
| test | `yes_count` 40–50 | `no_count` 40–50 |
| ood_test | `yes_count` 45–50 | `no_count` 46–50 |

The current v8 builder scripts in the upstream repository do use 45/50, but the released
`val`/`test` splits were exported from distinct source datasets rather than regenerated under
one rule, so the repository carries genuine provenance/version complexity. Document the
observed thresholds; do not "fix" the released data.

Also note: `yes_count + no_count < total_resamples` in 139/216 evaluation records, by 1–10
continuations. The two counts are counts of continuations meeting *distinct* criteria
(terminate soon / terminate late-or-never) with an unclassified band between them — they are
not a binary partition of the 50 samples. `yes_count/50` is therefore **not** a calibrated
termination probability, and any continuous-target analysis must account for this.

Therefore, do not casually describe training labels as equivalent to the evaluation target.

## Canonical input construction

A sample should be evaluated as the model actually encountered it:

`original user prompt + Qwen chat/thinking formatting + released cot_prefix`

Do **not** feed `cot_prefix` alone unless explicitly running that as an ablation.

When caching hidden activations, use the activation corresponding to the final real token of the supplied prefix. Be careful with padding.

## Primary metric

AUROC.

Use threshold-dependent metrics only as secondary diagnostics.

## Initial reproduction target

First reproduce the qualitative Hard-CoT phenomenon with an intentionally simple linear probe.

Use approximately:

* balanced training subset, initially ~4k samples;
* logistic regression with L2 regularization;
* a tiny validation-selected regularization grid;
* five strategically selected depths spanning the network;
* full available evaluation sets.

Initial reproduction succeeds if a straightforward implementation gives approximately:

`OOD AUROC >= 0.85`

A result around or above 0.90 is excellent.

If OOD AUROC is below ~0.75, assume the pipeline is wrong until proven otherwise. Debug formatting, activation location, labels, split use, model version, and token indexing before interpreting the result scientifically.

Exact reproduction of the authors' full hyperparameter sweep is not required.

## Planned research sequence

The experiment sequence should be adaptive rather than a fixed checklist.

### Phase A — establish the phenomenon

1. Audit Task 1 data and split construction.
2. Manually inspect positive/negative examples.
3. Reproduce strong OOD termination prediction using hidden-state probes.
4. Record layer dependence.

Do not begin mechanistic interpretation until reproduction is trustworthy.

### Phase B — test simple explanations

Prioritize simple baselines such as:

* `</think>` logit;
* `</think>` probability;
* relevant logit margins;
* next-token entropy;
* simple top-logit/output-distribution features;
* prefix length;
* simple text features where useful.

Compare these directly with activation probes on the same evaluation examples.

### Phase C — distinguish explanations

Choose subsequent experiments based on Phase B.

High-priority possibilities include:

* control/residualize hidden representations against output-level termination signals and retrain probes;
* test when in depth the activation signal appears relative to strong output-level evidence;
* determine whether probe score tracks termination probability continuously rather than merely the binary label;
* compare probe predictions with measures of answer commitment/sufficiency if these can be constructed cleanly;
* causal intervention only if the preceding results provide a clear reason to do it.

Do not perform steering merely because a probe direction exists.

## Scientific success criterion

The project succeeds if it produces evidence that distinguishes plausible explanations for the published high OOD probe performance.

A negative result is valid.

Examples of good outcomes:

* `</think>` propensity nearly explains the full activation result;
* activation probes contain a substantial residual OOD signal after output-level controls;
* signal is mainly explained by training/evaluation proxy structure;
* the strongest predictive representation appears earlier than output-level termination evidence;
* a seemingly strong probe result disappears under a meaningful control.

The project does **not** require a positive mechanistic discovery.

## What not to do

Unless clearly motivated by an existing result, do not spend time on:

* SAEs;
* BSFs;
* MLP probes;
* large hyperparameter sweeps;
* exhaustive layer sweeps before a sparse layer check;
* multi-model replication;
* full circuit reconstruction;
* large-scale new dataset generation;
* steering without a concrete causal question;
* optimizing a benchmark score for its own sake.

Simple methods are preferred when they answer the scientific question.

## Research standards

Every result should distinguish:

* observation;
* interpretation;
* alternative explanations;
* evidence against the preferred interpretation.

Never describe linear decodability as evidence of causal use without an intervention supporting that claim.

Never describe a single probe vector as "the termination direction" unless evidence justifies this unusually strong interpretation.

Do not tune decisions on the held-out OOD test set.

Record failures and abandoned hypotheses rather than deleting them.

## Agent behavior

Before changing experimental direction:

1. read this file;
2. read `STATE.md`;
3. read the latest entries in `RESULTS.md` and `DECISIONS.md`;
4. inspect existing artifacts rather than rerunning experiments unnecessarily.

When finishing a meaningful task:

1. save machine-readable outputs;
2. update `RESULTS.md` with the actual quantitative result;
3. update `STATE.md` with the current project state;
4. add a `DECISIONS.md` entry if the result changes what experiment should come next;
5. record the exact command/config needed for reproduction.

Do not silently change the scientific question or evaluation protocol.

If evidence contradicts the current hypothesis, report that explicitly and recommend the cheapest decisive next experiment.

## Current priority

The first milestone is:

> Reproduce strong OOD termination prediction with a simple hidden-state logistic probe on a small training subset and a few strategically selected Qwen3-32B layers.

Only after this milestone is trustworthy should the project move to output-level baselines and explanatory experiments.
