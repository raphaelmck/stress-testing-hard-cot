# Decisions

Append-only record of *why the project changed course*. Read this before proposing anything
that looks like a resolved question — these are closed unless the "Revisit if" clause fires.

---

## D001 — Repo is the single source of truth; four knowledge files, fixed roles

Date: 2026-08-31

Context:
A ~20 hour sprint run across multiple agent sessions and possibly multiple agent tools.
Chat history is not durable and does not transfer between tools.

Decision:
`PROJECT_BRIEF.md` (stable contract) / `DECISIONS.md` (reasoning) / `RESULTS.md` (append-only
reality) / `STATE.md` (compressed current view, rewritten, <~100 lines). Agents read all four
before acting and update `RESULTS.md` + propose `STATE.md` after every meaningful result.

Reason:
Separates *what the project is* from *what we currently believe*. Without this an agent
reconstructs the scientific goal from filenames and drifts.

Revisit if:
Never, within this sprint.

---

## D002 — Agent instructions live in `AGENTS.md`; `CLAUDE.md` is a one-line import

Date: 2026-08-31

Context:
Claude Code reads `CLAUDE.md`; most other coding agents read `AGENTS.md`. Options were
(a) duplicate the content, (b) symlink `CLAUDE.md -> AGENTS.md`, (c) a one-line `@AGENTS.md`
import in `CLAUDE.md`.

Decision:
(c). `AGENTS.md` is canonical. `CLAUDE.md` contains only `See @AGENTS.md`.

Reason:
Duplication drifts silently, which is the exact failure mode this repo exists to prevent.
A symlink is a single file on disk but degrades invisibly wherever symlinks are not preserved
(Windows checkouts without `core.symlinks`, zip/tarball copies, web IDEs, some sandboxes) —
it becomes a text file whose entire content is the string `AGENTS.md`, and the agent silently
gets no instructions. Claude Code expands `@path` imports into context, so the import costs one
indirection and fails loudly rather than silently.

Revisit if:
The instructions grow large enough that per-tool sections are needed. Then keep `AGENTS.md`
canonical and add tool-specific content *below* the import in `CLAUDE.md`, never instead of it.

---

## D003 — Upstream dataset is a pinned, gitignored, read-only clone

Date: 2026-08-31

Context:
`cot-proxy-tasks/` is a 1.1 GB clone of `Centrattic/cot-proxy-tasks` with its own `.git`.

Decision:
Do not vendor or submodule it. Gitignore the directory, pin the commit (`4482324`,
"tasks 1 and 2", 2026-05-04) here and in every run `config.json`, and treat it as read-only.

Reason:
Committing 46k JSON files or nesting a git repo buys nothing; the commit hash is the entire
reproducibility requirement. Writing into the clone would make provenance unrecoverable.

Revisit if:
Upstream publishes a revision that changes Task 1 labels — then pin the new hash in a new
DECISIONS entry and mark affected results as superseded, not invalidated.

---

## D004 — uv + Python 3.12 lockfile; local Mac is not an inference target

Date: 2026-08-31

Context:
The repo had no environment. Local machine is an M5 Pro Mac, 48 GB unified memory, no CUDA;
system Python is 3.14 with no relevant packages. Upstream `cot-proxy-tasks` ships **no**
dependency pins of any kind, so its environment cannot be matched by version.

Decision:
- `uv` with `pyproject.toml` + committed `uv.lock`; Python pinned to 3.12 in `.python-version`.
- One dependency set for both platforms; `[tool.uv] environments` locks resolution for
  macOS/arm64 and Linux/x86_64 so the same lockfile is valid on a rented GPU box.
- The local Mac is for analysis, probe fitting, and pipeline validation only. Qwen3-32B
  (~64 GB bf16) does not fit in 48 GB, and quantizing to fit would change the activations
  under study, so no real activation run happens locally.
- Model pinned to `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137`.

Reason:
3.12 rather than the system 3.14 because the GPU-side ecosystem (transformers, accelerate,
and standard CUDA images) is best-supported there. A committed lockfile is what makes a
`RESULTS.md` entry's commit hash actually mean something.

Verified:
Under the locked env (transformers 5.16.1, torch 2.13.0), the upstream helper
`cot-proxy-tasks/src/utils/chat_template.py::build_thinking_prompt` produces

    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    <|im_start|>user\n{prompt_text}<|im_end|>\n
    <|im_start|>assistant\n<think>\n{cot_prefix}

with exactly one `<think>` and no auto-inserted empty think block, and the final token of
the string is the final token of `cot_prefix`. This was the main version risk: some Qwen3
chat-template revisions emit `<think>\n\n</think>\n\n` under `add_generation_prompt=True`,
which would silently double the tag and move the activation site.

Revisit if:
transformers, the pinned model revision, or the upstream helper changes. Re-run the prompt
assertion before trusting any activation cached under the new versions.

---

## D005 — Evaluation hygiene, frozen before any activation is extracted

Date: 2026-08-31

Decision:

```text
Primary metric: AUROC.
Training:                      train split only.
Hyperparameter/layer choice:   val only.
ID check:                      test.
Final cross-domain check:      ood_test.
```

Never tune `C`, layer choice, preprocessing, or feature design on `ood_test`.

Predeclared now, not after seeing results:
- transformer depths **8, 24, 40, 56, 64** (zero-indexed `model.layers` 7, 23, 39, 55, 63);
- regularization grid **C in {0.01, 0.1, 1, 10}**, selected on val AUROC.

OOD performance may be reported for all five *predeclared* layers, but the experiment must
not be modified based on which OOD layer looks best. Do not repeatedly inspect OOD AUROC
during development.

Reason:
`ood_test` is 58 rows. It cannot survive repeated inspection, and it is the only clean-label
split (D-audit: positives 45–50/50 vs 40–50/50 in val/test). Predeclaring the layer set and
grid is what makes the reproduction a test rather than a search.

Revisit if:
Never within this sprint. A layer outside the predeclared five requires a new DECISIONS entry
written *before* the run.

---

## D006 — Uncertainty on ood_test is bootstrapped clustered by question_id

Date: 2026-08-31

Context:
`ood_test` is 58 records from only **32 questions**, and 16 of those 32 carry both a positive
and a negative prefix. Rows are not independent.

Decision:
All confidence intervals on `ood_test` (and `val`/`test`) come from a bootstrap that resamples
**question_id clusters**, not individual rows. Report the CI alongside every AUROC from the
first run onward.

Reason:
Row-level bootstrap would understate the interval by treating correlated prefixes from one
question as independent evidence. With 32 effective units, honest intervals are wide, and
a difference of <~0.05 AUROC between layers should not be interpreted.

Revisit if:
Never. Retrofitting CIs after seeing point estimates is exactly the failure this prevents.

---

## D007 — Cache stop-token statistics during the reproduction pass; embargo their analysis

Date: 2026-08-31

Context:
Testing H0 needs last-token output-distribution features. Obtaining them later would mean a
second full Qwen3-32B pass over every sample — a large fraction of a 20-hour budget.

Decision:
During the reproduction forward pass, also cache at the last real prefix token:
`</think>` logit, `</think>` log-probability, `</think>` logit margin vs the highest
non-`</think>` token, next-token entropy, and top-1 token id + log-probability.

**Embargo:** these values must not be used in probe selection, plotted against labels, or
inspected in any way until the activation reproduction is frozen and its `RESULTS.md` entry
is written.

Reason:
The scientific separation that matters is between *seeing the baseline result* and *choosing
the activation experiment* — not between physically executing two forward passes. Caching is
free; peeking is not.

Revisit if:
Never. The embargo lifts by its own terms once the reproduction entry is committed.

---

## D008 — Preregistered control: answer obtained =/= termination imminent

Date: 2026-08-31

Context:
The manual audit found evaluation negatives that have already solved the problem correctly
(`tetrahedron_volume` 18*sqrt(2), `lcm_three` 180, `euler_phi_30` 8, `partition_ordered` 64)
with 0/50 continuations terminating soon, while positives often contain answer + verification.
16 of 32 `ood_test` questions carry both labels, enabling a within-question paired analysis at
no extra compute.

Decision:
Register this now as a planned falsification/control, *before* any activation result exists.
Do **not** analyse these pairs until the reproduction is complete and frozen.

Reason:
This is the specific test that speaks to the Hard-CoT / causal-analysis reconciliation rather
than being a generic probe result. Its value depends entirely on having been declared before
the data was seen. Preserve the ordering: reproduce first, explain second.

Revisit if:
The reproduction fails (max OOD AUROC < 0.75), in which case there is no phenomenon to explain
and this control is moot until the pipeline is fixed.

---

## D009 — Compute is the Mila cluster (H100 / a100l 80 GB), not Colab

Date: 2026-09-01

Context:
Qwen3-32B is 65.5 GB in bf16 and needs an 80 GB card. Colab was evaluated first: it
does offer A100 80 GB and H100 tiers, and the official `google-colab-cli` would have
suited a script-shaped workload. But nothing reachable with the available Colab
compute credits could hold the model resident.

Decision:
Run R001 on the Mila cluster: `--gres=gpu:h100:1` or `--gres=gpu:a100l:1` (the 80 GB
A100 variant). Plain `a100` may be the 40 GB variant and must not be used. Runbook:
`notes/gpu_runbook.md`; batch script: `scripts/r001_extract.sbatch`.

Reason:
Quantizing to fit a smaller card would alter the activations under study and would
break the R001 debug rule -- a sub-0.75 OOD AUROC could not be distinguished from a
quantization artifact (D004). Renting an 80 GB card is the only option that keeps the
reproduction interpretable.

Consequences:
- HF cache goes on `$SCRATCH`, not `$HOME`; weights are prefetched from a login node
  so GPU time is not spent downloading 65 GB.
- Jobs are resumable by design, so pre-emption costs shards, not the run.
- The D004 prompt assertion must be re-run on the cluster before extraction, since
  the environment there is rebuilt from `uv.lock` rather than inherited.

Revisit if:
Mila queue times make the sprint infeasible, in which case a paid 80 GB instance
elsewhere is preferable to any quantized run.

---

## D010 — R002 comparator, feature handling, and paired inference

Date: 2026-09-01

Context:
R001 gave OOD AUROC 0.904 at the val-selected depth 40 and 0.964 at depth 56, where
depth 56 was identified as the maximum over the five predeclared depths *after*
looking at OOD. R002 compares an activation probe against output-level baselines
whose C is selected on val.

Decision:
1. The primary activation comparator for every explanation experiment is the
   **val-selected depth (40, OOD 0.904)**. Depth 56 / 0.964 is reported only as
   "descriptive maximum over the five predeclared depths". Using an OOD-selected
   depth against a val-selected baseline would give the activation side a post-hoc
   advantage.
2. `top1_token_id` is never a numeric feature. Token ids have no ordinal meaning;
   it enters only as `top1_is_think = (top1_token_id == </think>)`.
3. Method comparisons use a **paired** question-clustered bootstrap of
   Δ = AUROC(activation) − AUROC(output), resampling the same clusters for both
   score vectors — never two independent CIs, which are far too wide for a
   difference between scores evaluated on identical rows.
4. Scalar baselines are scored directly (untrained); orientation is fixed on val
   and never on ood_test.

Reason:
Each of these is a way the comparison could silently tilt toward the conclusion we
are trying to test. The paired bootstrap in particular is the difference between
"the intervals overlap, so who knows" and an actual estimate of the advantage.

Consequence discovered in R002:
The multivariate output baseline is fit on train, where every output feature is at
chance (AUROC ~0.51) because the train labels are distance-based. It therefore
lands *below* the untrained `think_logprob` scalar (0.727 vs 0.807) and is not the
strong H0 baseline it was intended to be. The best available output baseline is the
untrained scalar. Building a trained output baseline would require fitting on
evaluation-style labels, which D005 forbids; any future attempt needs its own
DECISIONS entry and a cross-fitting scheme clustered by question.

Revisit if:
A future experiment needs a trained output-level baseline. Then design the
cross-fitting first, in writing, before seeing its OOD number.

---

## D011 — the within-question control is confounded by prefix length; length becomes the primary nuisance variable

Date: 2026-09-01

Context:
D008 preregistered the within-question ood_test control as the clean test of whether
probe scores track imminent termination with topic held fixed. R003 ran it. The
activation probe separated 15/16 questions (sign test p = 0.0005) — but the released
`token_length`, included only as a sanity baseline, matched it exactly (macro paired
concordance 0.938 vs 0.938; paired delta 0.000 [-0.188, +0.125]) and was positive on
16/16 questions. Within a question, the YES prefix is simply the longer one, by
100-1,250 tokens.

Decision:
1. `token_length` is promoted from sanity baseline to **the primary nuisance
   variable** for every subsequent experiment. Any claim that the probe carries
   termination-specific information must be made against length, not only against
   output-level features.
2. The next experiment (R004) is the length control on `val` and `test`, not
   residualization against the D007 output features. val/test have 30 and 22
   questions with more multi-row questions, so the comparison has power that 16
   mostly-1v1 ood questions do not.
3. `ood_test` is now closed to further inspection until a preregistered final
   evaluation. It has been read three times (R001, R002, R003).

Reason:
D008 was designed to remove between-question topic structure and it does. It does
not remove depth-into-the-trace structure, and by holding the question fixed it
arguably concentrates it. Reporting R003's activation result without the length
baseline would have been the single most misleading thing this project could have
published.

Revisit if:
R004 shows the probe beats length within question on val/test. Then residualization
against the output features (the previously planned experiment) becomes the way to
separate H1 from H2, and it should be run against length as well.

---

## D012 — the OOD split's identification limit is a result; one experiment left

Date: 2026-09-01

Context:
R005 read the three released v8 builders in the pinned upstream clone. They filter
length globally to [500, 3000), balance classes per prompt by label quality, and
then balance token length in **global 500-token buckets** with no step that matches
YES/NO lengths within a prompt. In ood_test the mean within-question length gap is
+432 tokens -- smaller than the 500-token bucket width, so the balancing procedure
cannot see it. Buckets are balanced; within-question length concordance is 0.938.
On `test` the same conditional statistic is 0.494 (chance).

Decision:
1. Record as a result of the project that **the released ood_test split cannot
   cleanly separate a termination-specific state from reasoning progress**. This is
   an identification limit of the evaluation, not an uncertainty of ours, and it is
   scoped to that split -- `test` shows no such conditional association.
2. Claims stay narrow: "conditional length confound left by global balancing", never
   "the benchmark is broken", "the probe is a length detector", or "the published
   result is invalid". R004 contradicts the last two directly.
3. Exactly **one** more experiment (R006): a representation-level control on `test`
   asking whether the depth-40 score retains held-out signal after removing its
   marginal linear associations with `think_logprob` and `token_length`. Then stop
   and write up. No R007-R010.

Reason:
The remaining budget buys either one more control or a writeup, not both. The
distinction worth publishing is between what this project can establish (in-domain,
on `test`) and what the released evaluation does not let anyone establish
(cross-domain independence from reasoning progress).

Revisit if:
R006 comes back ambiguous. Then the writeup reports R001-R005 and names R006 as the
open question rather than running a sixth variant of it.
