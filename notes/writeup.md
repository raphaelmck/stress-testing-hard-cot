# What does linear decodability of reasoning termination actually establish?

A reproduction and falsification study of Hard-CoT Task 1 on Qwen3-32B.

---

## 1. Overview

Simple linear probes on Qwen3-32B hidden states predict whether the model is about to
close its chain of thought with `</think>`, and they do so across domains. We reproduce
that result and then spend the rest of the project trying to break it.

The reproduction holds. The obvious deflationary explanations do not account for it. But
two further results limit what the headline number can be said to establish, and they
point in opposite directions from each other:

- On held-out **in-domain** data, the probe carries information that neither the model's
  immediate propensity to emit `</think>` nor the raw length of the prefix explains.
- On the released **cross-domain** split, absolute prefix length is nearly perfectly
  predictive *within* a question while being weak when pooled across questions. The
  released builders balance length globally, in 500-token buckets, which does not
  constrain that conditional association. So the split's high AUROC does not by itself
  demonstrate transfer independent of question-conditional reasoning progress.
- A predeclared steering intervention along the frozen probe direction did not produce
  direction-specific control of termination. An equal-norm orthogonal perturbation
  produced a comparable behavioural change.

The summary claim: **strong, reliable decodability is established; a transferable causal
termination representation is not.** Each of those two halves is load-bearing.

Everything below comes from seven preregistered runs on a frozen protocol. Hypotheses,
evaluation depths, the regularisation grid, and the split roles were fixed before the
first activation was extracted; every subsequent design choice that could have tilted a
comparison was written down before the run that used it.

---

## 2. Motivation and hypotheses

Recent Hard-CoT interpretability work reports that linear probes on Qwen3-32B activations
predict imminent termination with >0.90 OOD AUROC. Separately, causal analysis of Qwen3
suggests the actual stopping mechanism is high-dimensional and concentrated in late
computation — not a single residual-stream direction. That tension is the motivating
puzzle: why is termination so linearly decodable if the causal mechanism is not itself
one-dimensional?

Three explanations were frozen before any data was seen:

- **H0, output-proximal.** The probe is largely reading information already present in
  the model's next-token distribution, especially its propensity to emit `</think>`.
- **H1, broader latent state.** Activations carry domain-general information about a
  reasoning state associated with imminent termination that is not reducible to that
  propensity.
- **H2, proxy/dataset structure.** Training and evaluation share correlates of "being
  near a conclusion" that make termination predictable without corresponding to a unified
  latent termination state.

A methodological commitment came with them: linear decodability is never to be described
as causal use without an intervention, and no probe vector is to be called "the
termination direction."

---

## 3. Data and protocol

Task 1 of the released `cot-proxy-tasks` repository, pinned at commit `4482324`. The
subject model is `Qwen/Qwen3-32B` at revision `9216db5`, run in bf16 on a single
A100-80GB.

One dataset property drives much of what follows. The **training** labels are cheap
distance proxies: positives are prefixes 25–55 words from the end of an existing CoT,
negatives 300+ words. The **evaluation** labels are behavioural: 50 continuations are
resampled per prefix, and a positive requires `</think>` to appear at token index 20–60.
These are not the same target, and the project never treats them as equivalent.

Inputs are built with the released `build_thinking_prompt` helper — the original user
prompt, the Qwen chat/thinking template, then the released prefix — and the activation is
taken at the **last real token of the prefix**. No truncation. The prompt construction was
re-verified on the cluster before extraction, because the chat template is jointly
determined by the transformers version and the model revision.

Evaluation hygiene, frozen in advance: train on `train`; select the regularisation
constant and the layer on `val` only; `test` is the in-domain check; `ood_test` is the
final cross-domain check and is never tuned on. Depths **8, 24, 40, 56, 64** and the grid
`C ∈ {0.01, 0.1, 1, 10}` were predeclared. Because `ood_test` is 58 rows from only 32
questions — 16 of which contribute both a positive and a negative — every confidence
interval bootstraps **question clusters**, not rows, and every method comparison uses a
**paired** bootstrap on identical resampled clusters.

---

## 4. Claim 1 — Strong termination decodability reproduces

Extracting activations for 4,216 rows (4,000 balanced training rows spanning 779
questions, plus all of val/test/ood_test) and fitting L2 logistic probes gives:

| depth | val AUROC | test AUROC | OOD AUROC [95% CI] |
|---|---|---|---|
| 8 | 0.625 | 0.706 | 0.699 [0.563, 0.820] |
| 24 | 0.725 | 0.795 | 0.832 [0.725, 0.924] |
| **40** | **0.897** | **0.909** | **0.904 [0.834, 0.966]** |
| 56 | 0.896 | 0.965 | 0.964 [0.899, 1.000] |
| 64 | 0.844 | 0.907 | 0.892 [0.806, 0.961] |

The **primary, non-leaky result is depth 40**: it is the depth validation selects, and its
OOD AUROC is **0.904 [0.834, 0.966]**. Depth 56 reaches 0.964, but that is the *maximum
over the five predeclared depths*, identified by looking at OOD, and it is reported here
only descriptively. Every subsequent comparison in this project uses depth 40, so that a
val-selected probe is never being compared against a val-selected baseline on unequal
terms.

Three of five predeclared depths clear 0.89 OOD, so the phenomenon is not an artifact of a
lucky layer. Signal rises through the network, peaks at depth 56, and falls back at the
final depth. We record that shape but do not lean on it: the 56-vs-64 gap is 0.072 against
intervals this wide, and representations can reorganise between layers while the LM head
preserves the relevant information.

Two facts constrain how much this table can carry on its own. Train AUROC saturates at
1.000 at four of five depths — the distance-based training labels are trivially separable,
so the probe is fit on an easier problem than the one it is scored on. And `val` is both
the noisiest-label split and the selection split, which is why val sits below test and OOD
throughout.

---

## 5. Claim 2 — Simple output and length explanations are insufficient

### Immediate stopping propensity

During the same forward pass we cached last-token output statistics — the `</think>`
logit, its log-probability, the logit margin, next-token entropy, top-1 identity and
log-probability — and embargoed them until the reproduction was committed. Fitting them
under the identical protocol:

| baseline | test AUROC | OOD AUROC |
|---|---|---|
| activation probe, depth 40 | 0.909 | 0.904 |
| `think_logprob`, used directly as a score | 0.695 | 0.807 |
| `think_margin` | 0.684 | 0.803 |
| `think_logit` | 0.601 | 0.577 |
| multivariate output-only (L2 logistic) | 0.561 | 0.727 |

A single untrained scalar — the log-probability the model assigns to `</think>` at the
next position — recovers much of the distance from chance. The paired question-clustered
delta against the depth-40 probe is **+0.214 [+0.091, +0.311]** on test and
**+0.096 [+0.000, +0.201]** on OOD. So the output-proximal story is a real part of the
picture and is not the whole of it.

Two structural facts sharpen this. First, the argmax next token is **never** `</think>` in
any of the 4,216 rows, and mean p(`</think>`) at the next position is ~2e-12: whatever
`think_logprob` contributes is the ordering of a far distributional tail, not the model
being about to stop. Second, **every output feature is at chance on the training
distribution** (AUROC 0.495–0.508) while reaching 0.77–0.81 on evaluation splits. Being 40
words from the end says almost nothing about the very next token. The probe therefore
cannot have learned "current `</think>` propensity" from its training labels — it was
trained on labels uncorrelated with that quantity. This also handicaps the multivariate
output baseline, which is fit on train where its features carry no signal and consequently
lands *below* the untrained scalar; the honest strong output baseline here is
`think_logprob` alone.

### Absolute prefix length

Length is the other cheap explanation. On held-out `test`, comparing every within-question
YES/NO pair (17 questions, 137 pairs), and macro-averaging within question then across
questions:

| score | within-question concordance [95% CI] |
|---|---|
| depth-40 activation probe | **0.874 [0.727, 0.977]** |
| `think_logprob` | 0.794 [0.664, 0.908] |
| `token_length` | **0.494 [0.321, 0.667]** |

Length is at chance. The paired delta, activation minus length, is
**+0.137 to +0.605**, P(Δ>0) = 0.998. The decisive subset is the **52 pairs where the
positive prefix is *shorter* than the negative**: there the probe is still correct
**0.923** of the time (macro 0.914 [0.814, 1.000]), beating chance on 12 of 14 questions,
while `think_logprob` falls to 0.635. On near-matched pairs — thresholds |Δlen| ≤ 100 and
≤ 250 frozen before the run — the probe scores **0.983** and **0.986**, against 0.567 and
0.707 for `think_logprob`.

### Linear nuisance control

Finally, a **score-level** control: regress the frozen depth-40 score on standardised
`token_length` and `think_logprob`, fitting on `val` only and using no labels, then apply
the frozen coefficients to `test`.

| score on test | pooled AUROC [95% CI] | within-question |
|---|---|---|
| raw depth-40 | 0.909 [0.835, 0.966] | 0.874 |
| joint residual | **0.831 [0.716, 0.928]** | 0.818 |
| length-only residual | 0.882 [0.779, 0.954] | 0.918 |
| think-only residual | 0.841 [0.717, 0.934] | 0.868 |

The nuisances explain R² = 0.334 of the score on val, and removing both costs 0.077 AUROC
— paired delta +0.077 **[−0.006, +0.173]**, an interval that includes zero. Nearly all the
cost is `think_logprob`; removing length alone costs 0.027 and *raises* within-question
concordance to 0.918.

**What this supports:** absolute length and current stop propensity explain some of the
structure and do not account for the held-out probe result. **What it does not support:**
that the surviving signal is a termination-specific representation. This is a linear,
marginal control on one scalar. A residual that survives it can still be a nonlinear
function of the same nuisances, and nothing was projected out of the 5,120-dimensional
activation.

---

## 6. Claim 3 — The OOD evaluation does not cleanly identify progress-independent transfer

The preregistered within-question control on `ood_test` — 16 questions carry both labels,
holding prompt, domain and often the already-obtained answer fixed — produced the result
that redirected the project:

| score | macro within-question concordance | questions with positive Δ |
|---|---|---|
| depth-40 activation probe | 0.938 [0.812, 1.000] | 15/16 |
| `think_logprob` | 0.844 [0.656, 1.000] | 13/16 |
| **`token_length`** | **0.938 [0.844, 1.000]** | **16/16** |

The probe separates within question — 15 of 16, two-sided binomial p = 0.0005 — so the OOD
result is not merely between-question topic structure. But raw length matches it exactly
(paired delta 0.000 [−0.188, +0.125]) and is positive in *every* question, with a mean
within-question gap of **+432 tokens**.

Reading the three released v8 builders directly explains how that coexists with a
length-balanced benchmark. They apply a global length filter to [500, 3000), balance class
counts *per prompt* by label quality (`mean_yes_position`, `no_count` — never by length),
and then balance token length in **global 500-token buckets**. No step in any of the three
compares a positive item's length against a negative item's from the same prompt. The
skew was known — one step is commented "remove majority-class excess per bucket to
eliminate the systematic length skew (no→short, yes→long)" — and the correction is applied
globally.

**Global 500-token bucket balancing does not constrain within-prompt or within-bucket
label–length differences, so a substantial conditional skew can survive despite good
global balance.** The released bucket tables confirm the global balance holds. A
deterministic toy example makes the arithmetic concrete: one long question contributing 1
positive and 3 negatives, one short question contributing 3 positives and 1 negative,
gives pooled length AUROC 0.438 with within-question concordance 1.000. Unequal
per-question class counts across different length scales suffice.

**This is split-specific and must be stated as such:**

| split | pooled length AUROC | within-question length concordance | depth-40 probe |
|---|---|---|---|
| val | 0.450 | 0.653 | 0.907 |
| test | 0.499 | **0.494** | 0.874 |
| ood_test | 0.587 | **0.938** | 0.938 |

The conditional association appears in `ood_test` and not in `test`. And `ood_test` — 31
pairs from 16 mostly one-versus-one questions — is the split with the least data to
establish it.

The supportable conclusion: **high pooled OOD AUROC on this released split does not by
itself establish transfer independently of question-conditional reasoning progress.** Not
supported, and contradicted by Claim 2: that the benchmark is broken, that the probe is a
length detector, or that length explains the published result generally.

---

## 7. Claim 4 — The decoded direction is not a specific causal control direction under our intervention

Everything above is correlational. The final experiment intervenes.

The frozen depth-40 probe was recovered in raw activation coordinates
(`β[j] = coef_[j] / scaler.scale_[j]`, ‖β‖ = 7.849) after verifying that a deterministic
refit reproduces the committed scores to 2.0e-7 relative. The edit
`δh = (Δs/‖β‖²)·β` moves the frozen score by exactly `Δs`. Magnitudes were chosen from the
**validation** score distribution (σ = 17.18) before any behaviour was observed; a 2-SD
edit is **2.6%** of the depth-40 residual norm. The intervention is applied to the final
real prefix token at prefill and to every newly generated token thereafter. Seven
conditions — β at −2/−1/0/+1/+2 SD, plus one fixed seeded matched-norm **orthogonal**
direction at ±2 — across all 86 test prefixes, 4 continuations each, temperature 0.7,
60-token cap, with seeds shared across conditions so every condition sees identical random
draws. 2,408 generations.

**Validity first.** Unsteered, the protocol reproduces the released labels: positives
terminate within 60 tokens at **0.895** (0.866 inside the released 20–60 window),
negatives at **0.000**. The edit lands at the requested score shift to **0.158%**, the
depth-64 state moves ~2% of its norm, and generations remain coherent.

**Behaviour:**

| β −2 | β −1 | β 0 | β +1 | β +2 | ⊥ −2 | ⊥ +2 |
|---|---|---|---|---|---|---|
| 0.439 | 0.436 | 0.448 | 0.459 | 0.453 | 0.442 | 0.459 |

Primary contrast P(+2β) − P(−2β) = **+0.0145 [+0.0019, +0.0306]**. Matched orthogonal
contrast = **+0.0174 [+0.0000, +0.0404]** — at least as large.

There is no meaningful monotonic dose response; an orthogonal perturbation of equal norm
produces a comparable swing; 85–88% of steered continuations are token-identical to
baseline under common random numbers; and on released-negative prefixes, where the model
terminates within 60 tokens 0% of the time, +2σ steering produced termination in **0 of
172** generations.

**Despite strong linear decodability, sustained steering along the frozen depth-40 probe
direction produced no direction-specific control of termination at the tested magnitudes.
An equal-norm orthogonal perturbation produced a comparable behavioural change.**

Limitations that bound this precisely: one direction, depth 40 only, one intervention
schedule, edits up to 2.6% of the residual norm, and the immediate `think_logprob` channel
did not move monotonically either. A different causal representation, layer, or
intervention can exist. This result does not show that the probe is non-causal in general,
does not prove that decodability fails to imply causality as a principle, and does not
establish that termination is high-dimensional.

---

## 8. What is and is not established

| claim | status |
|---|---|
| Termination is strongly linearly decodable, including cross-domain | **Established** (OOD 0.904 at the val-selected depth) |
| Reducible to immediate `</think>` propensity | **No** — chance on the training distribution; probe leads by +0.214 on test |
| More than between-question topic structure | **Yes** — 15/16 within-question on OOD, p = 0.0005 |
| More than raw prefix length, in-domain | **Yes** — 0.874 vs 0.494; 0.914 where length is discordant |
| Survives a linear control for length and stop propensity | **Yes, at a cost** — 0.909 → 0.831, interval includes zero |
| Transfer independent of reasoning progress, cross-domain | **Not identifiable on the released OOD split** |
| The decoded direction causally controls stopping | **No evidence** — and the null is itself weak |

---

## 9. Limitations and the next experiment

The largest limitations are structural rather than incidental. `ood_test` is 58 rows from
32 questions and was deliberately inspected only three times; several intervals reach
1.000 and no difference below ~0.05 AUROC is resolvable there. `test` is in-domain, so the
strongest positive results — the length control and the nuisance control — establish
in-domain behaviour, while the cross-domain claim rests on the split whose confound this
project documents. Activations were cached in bf16 and depend mildly on batch composition.
Every probe result is at one val-selected depth. And the steering null is bounded as
described above.

Two candidate next experiments, in order:

1. **A progress control that does not use future information.** The natural competitor to
   H1 is now "question-relative reasoning progress," not raw length. Constructing it
   without leaking the rollout's eventual total length is the hard part and is why it was
   not run here.
2. **A representation-level nuisance projection and a larger or multi-layer intervention.**
   The control in Claim 2 acts on a scalar; projecting nuisance subspaces out of the
   activation, and steering with larger or distributed edits, are the two ways to convert
   the current weak null into something stronger in either direction.

---

## 10. Reproduction

All numbers come from committed run artifacts under `artifacts/runs/`, each with a
`config.json` recording the model revision, sample hash, depths, and seed. The frozen
sample is deterministic from seed 42 (`sha256` prefix `c261306dde08c8b9` for the full
4,216-row worklist). The ledger in `RESULTS.md` is append-only and records, for every run,
the interpretation *and* the evidence against it; `DECISIONS.md` records every design
choice that could have tilted a comparison, with the date it was frozen relative to the
run that used it.

Figures: `artifacts/figures/reproduction_layer_auroc.png` (Claim 1),
`r005_length_balance.png` (Claim 3), `r007_steering_dose.png` (Claim 4).
