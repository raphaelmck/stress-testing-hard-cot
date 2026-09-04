# What Does a Reasoning-Termination Probe Actually Know?

*Stress-testing Hard-CoT Task 1 on Qwen3-32B*

---

## Executive summary

Simple linear probes predict whether Qwen3-32B will soon terminate its chain of thought,
including on a held-out domain. I reproduced this result — **OOD AUROC 0.904 at the
validation-selected layer** — and then asked what, if anything, that performance tells us
about the model's underlying termination mechanism.

**Two simple explanations account for only part of the signal.** The model's current
log-probability of emitting `</think>` reaches 0.807 OOD AUROC on its own but
substantially underperforms the activation probe (paired advantage +0.214 on test), and it
has essentially no marginal discriminative signal for the proxy labels the probe was trained
on (AUROC ~ 0.5). Raw prefix length also fails on the better-powered held-out test set:
within a question the
activation probe reaches 0.874 concordance versus 0.494 for length, and it stays at 0.914
on the pairs where the positive prefix is actually *shorter* than the negative.

**The cross-domain result is harder to interpret.** In the released OOD math split, pooled
length is only weakly predictive (AUROC 0.587), while within the same question it orders
positive above negative prefixes at 0.938 concordance — exactly matching the activation
probe. Inspecting the released builders showed global 500-token length balancing but no
explicit within-prompt length-matching step in the code paths I read. So the high OOD
AUROC does not by itself establish that the decoded signal transfers independently of
question-relative reasoning progress.

**Finally, I intervened causally** along the frozen validation-selected probe direction.
The generation protocol reproduced the behavioural labels, the intervention propagated
downstream and preserved coherent output, but ±1–2 SD steering produced no
direction-specific or monotonic control of termination; an equal-norm orthogonal direction
produced a comparable effect. On benchmark-negative prefixes, +2 SD steering caused **0 of
172** terminations.

My conclusion is deliberately narrower than "the probe found a termination mechanism":
**strong cross-domain decodability is reproducible, but these experiments do not
identify a transferable causal termination representation.** The value is in mapping where
increasingly strong interpretations of an impressive probe result do and do not survive
falsification.

### Results at a glance

| Within-question comparison | Held-out `test` | Cross-domain `ood_test` |
|---|---|---|
| Activation probe (depth 40) | **0.874** | **0.938** |
| Prefix length | **0.494** | **0.938** |
| Interpretation | Probe clearly beats length | Evaluation cannot separate probe from progress/length |

**The apparent tension here is the main finding, not a contradiction: the strongest
positive control is in-domain, while the strongest identification limitation is
cross-domain.**

Three takeaways:

- The reported cross-domain decoding performance reproduces cleanly at a validation-selected
  layer.
- Immediate stopping propensity explains part of the signal; raw length does not explain
  the held-out test result.
- Neither the released OOD evaluation nor a controlled steering test supports the stronger
  claim that a transferable causal termination representation has been identified.

---

## 1. The puzzle, and what would have changed my mind

Hard-CoT [1] reports that linear probes on Qwen3-32B activations predict imminent
`</think>` with >0.90 OOD AUROC. Separately, recent causal work on smaller Qwen3 models
argues that stopping is mediated by distributed late-layer verification computation rather
than a single residual-stream direction [2]. If something like that holds here, why is termination
so linearly decodable?

After the initial dataset audit, and before examining any activation-probe result, I wrote
down three competing explanations:

- **Output-proximal:** the probe mostly reads information already in the next-token
  distribution, especially `</think>` propensity.
- **Latent state:** activations carry domain-general information about a reasoning state
  associated with imminent termination, not reducible to that propensity.
- **Proxy/dataset structure:** training and evaluation share correlates of "being near a
  conclusion" that make termination predictable without a unified latent state.

The initial hypotheses, split roles, probe layers and regularisation grid were fixed before
examining activation results. Subsequent controls were chosen **adaptively** in response to
earlier findings; for each, the important analysis choices and interpretation criteria were
recorded before inspecting that experiment's outcome. That sequence — and the two places it
forced me to reverse a conclusion — is the substance of what follows.

I also committed to two rules: linear decodability would never be described as causal use
without an intervention, and no probe vector would be called "the termination direction."

---

## 2. Setup

Task 1 of the released `cot-proxy-tasks` benchmark, subject model `Qwen/Qwen3-32B`.

One dataset property drives much of the analysis. **Training labels are cheap distance
proxies** — positives are prefixes 25–55 words from the end of an existing CoT, negatives
300+ words. **Evaluation labels are behavioural** — 50 continuations are resampled per
prefix, and a positive requires `</think>` at token index 20–60. These are not the same
target, and I never treat them as equivalent.

Probes are L2 logistic regressions on the activation at the last real token of the prefix,
at five predeclared depths (8, 24, 40, 56, 64). Training uses `train` only; the
regularisation constant and the layer are selected on `val` only; `test` is the in-domain
check; `ood_test` is the final cross-domain check and is never tuned on. Because `ood_test`
is 58 rows from 32 questions, every interval bootstraps **question clusters**, and every
method comparison uses a **paired** bootstrap on identical resampled clusters. Exact model
revision, prompt construction, and reproduction details are in the appendix.

---

## 3. The probe result reproduces

Fitting the frozen protocol on 4,216 rows (4,000 balanced training rows from 779 questions
plus all evaluation splits):

| depth | val | test | OOD [95% CI] |
|---|---|---|---|
| 8 | 0.625 | 0.706 | 0.699 [0.563, 0.820] |
| 24 | 0.725 | 0.795 | 0.832 [0.725, 0.924] |
| **40** | **0.897** | **0.909** | **0.904 [0.834, 0.966]** |
| 56 | 0.896 | 0.965 | 0.964 [0.899, 1.000] |
| 64 | 0.844 | 0.907 | 0.892 [0.806, 0.961] |

*Figure 1 (`report_fig1.png`): termination-probe AUROC at the five predeclared depths.
Depth 40 is validation-selected; depth 56 is the descriptive maximum over those depths. The
train line shows the saturation caveat discussed below.*

**The primary result is depth 40 — the depth validation selects — at OOD 0.904.** Depth 56
reaches 0.964, but that is the maximum over the five predeclared depths, identified by
looking at OOD, and I report it descriptively only. Every later comparison uses depth 40,
so a val-selected probe is never compared against a val-selected baseline on unequal terms.

Three of five predeclared depths clear 0.89 OOD, so the effect is not confined to a single
selected layer. Two facts temper the table: train AUROC saturates at 1.000 at four depths,
because the distance-based training labels are trivially separable, and `val` is also the
selection split and has substantially lower measured AUROC.

---

## 4. Simple output and length cues explain only part of the probe

During the same forward pass I cached last-token output statistics and embargoed them until
the reproduction was committed. Used directly as scores:

| baseline | test | OOD |
|---|---|---|
| activation probe, depth 40 | 0.909 | 0.904 |
| `think_logprob` | 0.695 | 0.807 |
| `think_margin` | 0.684 | 0.803 |
| `think_logit` | 0.601 | 0.577 |

The paired question-clustered advantage of the probe over `think_logprob` is **+0.214
[+0.091, +0.311]** on test and +0.096 [+0.000, +0.201] on OOD.

Two structural facts matter more than the table. The argmax next token is **never**
`</think>` in any of the 4,216 rows and mean p(`</think>`) is ~2e-12, so this feature is the
ordering of a far distributional tail, not the model being about to stop. And all cached
output features have approximately chance discriminative performance on the *training*
distribution (AUROC 0.495–0.508). Some — most notably `think_logprob` and `think_margin` —
become substantially predictive on the behavioural evaluation splits despite that lack of
training signal. The training labels therefore provide essentially no marginal signal about
current `</think>` propensity, making a simple account in which the probe merely learns that
quantity unlikely.

---

## 5. What surprised me: an OOD conditional-length association

I expected the main question to be whether the probe merely decoded immediate stopping
propensity. The result above weakened that. The bigger surprise came from a control I had
registered in advance: within the 16 `ood_test` questions that carry both a positive and a
negative prefix — holding the question, prompt and domain fixed, though the two prefixes
generally come from different reasoning rollouts —

| score | macro within-question concordance | questions positive |
|---|---|---|
| activation probe, depth 40 | 0.938 [0.812, 1.000] | 15/16 |
| `think_logprob` | 0.844 [0.656, 1.000] | 13/16 |
| **prefix length** | **0.938 [0.844, 1.000]** | **16/16** |

The probe separates within question (15/16, p = 0.0005), so the OOD result is not merely
topic structure. But raw length matched it exactly, positive in every question, with a mean
within-question gap of **+432 tokens**. I initially thought this collapsed the whole result
into a length shortcut.

That surprise sent me to the released builders rather than to another model. In the code
paths I inspected, all three apply a global length filter, balance class counts *per
prompt* by label quality (never by length), then balance token length in **global 500-token
buckets**; I found no explicit within-prompt length-matching step. The skew was known — one
step is commented "remove majority-class excess per bucket to eliminate the systematic
length skew (no→short, yes→long)" — and the correction is global.

**Global 500-token bucket balancing does not constrain within-prompt or within-bucket
label–length differences, so a substantial conditional skew can survive despite good global
balance.** A toy construction shows that this pooled-versus-conditional discrepancy can
arise even when pooled length appears balanced; it is intuition, not evidence that this
mechanism caused the observed OOD skew.

*Figure 2 (`report_fig2.png`): grey is prefix length pooled across questions (AUROC);
orange and blue are within-question concordance for length and for the probe. Both metrics
have 0.5 as chance. The conditional length association appears on OOD and not on test.*

| split | pooled length AUROC | within-question length | probe |
|---|---|---|---|
| val | 0.450 | 0.653 | 0.907 |
| test | 0.499 | **0.494** | 0.874 |
| ood_test | 0.587 | **0.938** | 0.938 |

**This is split-specific.** The conditional association appears in `ood_test` and not in
`test` — and `ood_test`, 31 pairs from 16 mostly one-versus-one questions, is the split with
the least data to establish it. The supportable statement: *the released OOD split does not
let this analysis distinguish domain-general termination information from question-conditional
reasoning progress as cleanly as its pooled length balance suggests.*

---

## 6. The critical falsification: the probe is not a length detector

The obvious next inference — "the probe is reading length" — is the one I tested next, on
`test` rather than `val`, because depth 40 was selected on `val`. Across 17 questions and
137 within-question pairs:

- probe **0.874 [0.727, 0.977]**; `think_logprob` 0.794; **length 0.494 [0.321, 0.667]**;
  paired advantage of probe over length **+0.137 to +0.605**, P(Δ>0) = 0.998.
- On the **52 pairs where the positive prefix is shorter**, the probe is still right
  **0.923** of the time (macro 0.914 [0.814, 1.000]), beating chance on 12 of 14 questions,
  while `think_logprob` falls to 0.635.
- On near-matched pairs (thresholds |Δlen| ≤ 100 and ≤ 250, frozen before the run) the probe
  scores **0.983** and **0.986**.

So the OOD tie with length does not generalise, and the naive raw-length explanation is
inconsistent with the better-powered held-out comparison.

---

## 7. Steering the decoded direction

Everything above is correlational. The last experiment intervenes, with the design frozen
in advance.

I recovered the frozen depth-40 probe in raw activation coordinates and verified that the
edit `δh = (Δs/‖β‖²)·β` moves the probe score by exactly `Δs`. Magnitudes came from the
**validation** score distribution before any behaviour was observed; a 2-SD edit is 2.6% of
the depth-40 residual norm. The edit is applied to the final prefix token and to every newly
generated token thereafter. Seven conditions — β at −2/−1/0/+1/+2 SD plus one fixed
matched-norm **orthogonal** direction at ±2 — over all 86 test prefixes, 4 continuations
each, temperature 0.7, 60-token cap, with random draws shared across conditions. 2,408
generations.

**Validity first.** Unsteered, the protocol reproduces the released labels: positives
terminate within 60 tokens at **0.895**, negatives at **0.000**. The realized probe-score shift is within
**0.158%** of the requested shift, the final-layer state moves ~2% of its norm, and output stays
coherent.

| β −2 | β −1 | β 0 | β +1 | β +2 | ⊥ −2 | ⊥ +2 |
|---|---|---|---|---|---|---|
| 0.439 | 0.436 | 0.448 | 0.459 | 0.453 | 0.442 | 0.459 |

- Primary contrast P(+2β) − P(−2β) = **+0.0145 [+0.0019, +0.0306]**
- Matched orthogonal contrast = **+0.0174 [+0.0000, +0.0404]**
- On benchmark-negative prefixes, +2β caused termination in **0 of 172** generations

**The small β-direction effect is not specific: the tested equal-norm orthogonal direction
produces at least as large a change.** There is no monotonic dose response, and 85–88% of
steered continuations are token-identical to baseline.

*Figure 3 (`report_fig3.png`): the orthogonal control is a single fixed matched-norm
direction. The axis starts at zero because the effects are ~1 point around a ~45% baseline.*

Stated precisely: *despite strong linear decodability, sustained steering along the frozen
depth-40 probe direction produced no direction-specific control of termination at the tested
magnitudes.* This is one direction, one layer, one schedule, at edits up to 2.6% of the
residual norm. It does not show the probe is non-causal in general, does not prove that
decodability fails to imply causality as a principle, and does not establish that
termination is high-dimensional.

---

## 8. What is and is not established

| claim | status |
|---|---|
| Termination is strongly linearly decodable, including cross-domain | **Established** (OOD 0.904, val-selected depth) |
| Explained by immediate `</think>` propensity alone | **Not supported** — at chance on the training distribution; probe leads by +0.214 on test |
| More than between-question topic structure | **Yes** — 15/16 within-question on OOD, p = 0.0005 |
| More than raw prefix length, in-domain | **Yes** — 0.874 vs 0.494; 0.914 where length is discordant |
| Survives a linear nuisance control | **Yes, at a cost** — 0.909 → 0.831; the paired-drop CI includes zero |
| Transfer independent of reasoning progress, cross-domain | **Not identifiable on the released OOD split** |
| The decoded direction causally controls stopping | **No evidence** — and this null is itself weak |

These experiments do not establish that the decoded signal is a transferable causal
termination representation. They also do not establish that no such representation exists.

---

## 9. If I had one more day

**I would rebuild the OOD evaluation so that positive and negative prefixes are matched
within question on an online-available progress proxy, then rerun the frozen probe.** That
directly tests the largest remaining uncertainty: whether the activation signal transfers
cross-domain independently of reasoning progress. The proxy must not use the rollout's
eventual total length, which is why I did not run it here.

Only if the signal survives that would I invest in stronger mechanistic interventions —
projecting nuisance subspaces out of the representation rather than out of a scalar, and
larger or multi-layer edits.

---

## Appendix

**Model and protocol.** `Qwen/Qwen3-32B` at revision `9216db5`, bf16, single A100-80GB.
Upstream data pinned at commit `4482324`. Inputs use the released `build_thinking_prompt`
helper (original prompt + Qwen thinking template + released prefix); the prompt string was
re-verified on the cluster before extraction because the chat template depends jointly on
the transformers version and model revision. No truncation. Activations are taken at the
last real prefix token with right padding, verified against an unbatched forward pass. The
frozen 4,216-row sample is deterministic from seed 42.

**Uncertainty.** All intervals are 2,000-replicate bootstraps resampling `question_id`
clusters; single-label replicates are discarded rather than coerced. Condition and method
comparisons resample identical clusters for both scores.

**Nuisance control (secondary robustness check).** I fit on validation a linear nuisance
model predicting the frozen probe score from prefix length and current `</think>`
log-probability (R² = 0.334). Applying that frozen correction to `test` reduces AUROC from
0.909 to 0.831 [0.716, 0.928]. Length alone accounts for little of the drop (residual
0.882); most is associated with `think_logprob` (0.841). Because this operates only on the
scalar probe score and only linearly, I treat it as evidence that these nuisances matter,
not as removal of nuisance information from the representation.

| score | pooled AUROC [95% CI] | within-question |
|---|---|---|
| raw depth-40 | 0.909 [0.835, 0.966] | 0.874 |
| joint residual | 0.831 [0.716, 0.928] | 0.818 |
| length-only residual | 0.882 [0.779, 0.954] | 0.918 |
| think-only residual | 0.841 [0.717, 0.934] | 0.868 |

Paired delta raw − joint residual: +0.077 [−0.006, +0.173].

**Steering diagnostics.** The mandatory propagation checks were: realized vs requested
depth-40 score shift (within 0.158%), final-layer state change (~2% of norm), and logit change
above noise. The last passed at exactly one bf16 ULP (0.250 in every non-zero condition), so
the final-layer state change is the meaningful propagation evidence. The current
`think_logprob` moved non-monotonically under β (−0.115 to +0.129), so this run does not
provide a clean local-versus-future dissociation either. The 60-token cap cannot express the
released negative criterion (`</think>` after token 200 or absent).

**Run chronology.** R001 reproduction → R002 output baselines → R003 within-question OOD
control → R004 length control on test → R005 builder audit → R006 nuisance control → R007
steering. The append-only `RESULTS.md` ledger records, for each run, the interpretation *and*
the evidence against it; `DECISIONS.md` records each design choice with the date it was
frozen relative to the run that used it, including two entries where a result reversed the
planned next experiment.

**Tool use.** I used LLM coding agents heavily for implementation, experiment orchestration,
and bookkeeping, and used LLMs as research sounding boards. I made the final
experimental-design and interpretation decisions, reviewed the generated code and results,
and repeatedly changed or rejected proposed analyses when I thought the inference was
invalid. The repository contains the append-only decision/result ledger and all run
artifacts.

**Active project time: ~[FILL IN] hours**, excluding GPU queue and run time and generic
cluster setup, consistent with the application's time-counting rules.

**References.**

[1] D. Ivanova, R. Tyagi, J. Engels, N. Nanda. *Test your best methods on our hard CoT
interp tasks.* LessWrong / Alignment Forum, 2026.
`lesswrong.com/posts/tDJWZLQNN7poqCwKa/test-your-fancy-methods-on-our-hard-cot-interp-tasks`
— defines the partial-transcript / resampling termination task and the released splits used
here.

[2] C. Dutta. *The Termination Circuit (how reasoning models stop thinking).* LessWrong,
2026. `lesswrong.com/posts/ajhzc6ktEKyFeJFBS/the-termination-circuit-how-reasoning-models-stop-thinking`
— late-MLP verification gate, reported on Qwen3-1.7B; motivation for the puzzle here, not an
established claim about Qwen3-32B.

**Figures.** `artifacts/figures/report_fig{1,2,3}.png`, regenerated from the committed
tables and metrics by `src/make_report_figures.py`, which refits nothing and overwrites no
run artifact. The original per-experiment figures, including the within-question dumbbell
plot from the OOD control, remain alongside them.

**Repository:** `github.com/raphaelmck/nanda-w27-app`
