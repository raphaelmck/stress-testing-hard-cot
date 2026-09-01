# Task 1 dataset audit

Source: `cot-proxy-tasks/datasets/1/`, upstream commit `4482324`.
Programmatic pass: `src/inspect_task1.py` -> `artifacts/tables/task1_{manifest.csv,summary.md}`.

**Status: complete. Do not spend further time characterising the dataset.**

## 1. Programmatic inventory (verified 2026-08-31)

| split | prompts | records | no / yes | unique qid | label construction |
|---|---|---|---|---|---|
| train | 796 | 46,160 | 23,080 / 23,080 | 796 | proxy: `distance_from_end` |
| val | 30 | 72 | 36 / 36 | 30 | behavioural: 50 resamples |
| test | 22 | 86 | 43 / 43 | 22 | behavioural |
| ood_test | 32 | 58 | 29 / 29 | 32 | behavioural |

All splits exactly balanced. **Zero `question_id` overlap between every pair of splits** —
the split is clean at the question level, no leakage.

## 2. Train labels are a distance proxy, and the arithmetic is exact

- positives: `distance_from_end` in {25, 35, 45, 55}, ~5.7k each;
- negatives: a ladder 300 -> 11,300 words in steps of 200 (7,705 at exactly 300, long tail);
- `total_words - prefix_words == distance_from_end` for **all 46,160 rows**, 0 violations;
- released `prefix_words` equals a naive whitespace split for all 46,160 rows.

## 3. Length is a training shortcut that does NOT transfer

Derived prefix word count, median by label:

| split | NO | YES |
|---|---|---|
| train | 407 | **609** |
| val | 569 | 531 |
| test | 687 | 637 |
| ood_test | 640 | 672 |

AUROC of raw length as a termination predictor:

| feature | train | val | test | ood_test |
|---|---|---|---|---|
| prefix word count | 0.606 | 0.466 | 0.510 | 0.584 |
| released `token_length` | - | 0.450 | 0.499 | 0.587 |

The training-length association is mechanical (a YES prefix is nearly the whole trace; a NO
prefix is truncated 300+ words earlier). It is essentially useless ID (0.50) and weak OOD
(0.58–0.59).

**Consequence:** a strong OOD activation result cannot be dismissed as "positive prefixes are
longer". But a probe trained on this proxy distribution could still learn generic
"late-reasoning" features even though raw length itself does not transfer — that is H2, and it
is not ruled out by these numbers.

## 4. Empirical purity thresholds differ by split — correction

Do **not** state that all released evaluation splits use 45/50:

| split | positives | negatives | records failing a >=90% agreement bar |
|---|---|---|---|
| val | `yes_count` 40–50 | `no_count` 40–50 | 32 / 72 |
| test | `yes_count` 40–50 | `no_count` 40–50 | 42 / 86 |
| ood_test | `yes_count` 45–50 | `no_count` 46–50 | 0 / 58 |

`ood_test` is the cleanest label set; `val` — which is also the model-selection set — is the
noisiest. If val AUROC comes in below ood AUROC, label noise is a sufficient explanation.

Upstream's current v8 builder scripts do use 45/50, but the released `val`/`test` splits were
exported from distinct source datasets rather than regenerated under one rule. This is
upstream provenance complexity. Document it; do not "fix" the released data.

## 5. `yes_count + no_count != total_resamples`

In **139 / 216** evaluation records the two counts fall short of `total_resamples` by 1–10.
They count continuations meeting *distinct* criteria (terminate soon / terminate late-or-never)
with an unclassified band between, not a binary partition of 50 samples.

**Consequence:** `yes_count/50` is not a calibrated termination probability. The planned
"does probe score track termination probability continuously?" experiment must handle the
unclassified mass explicitly rather than treating `yes_count/50` as p(terminate).

(This also corrects an earlier inference of mine: I read ood negatives as `no_count >= 48`
from `yes_count <= 2`. The counts do not sum, so the true range is `no_count` 46–50.)

## 6. Manual inspection — answer obtained =/= termination imminent

Several evaluation **negatives have already solved the problem correctly**, yet 0/50
continuations terminate soon:

| question_id | split | label | resamples | state of reasoning |
|---|---|---|---|---|
| `tetrahedron_volume` | ood_test | no | 0/50 yes | has derived 18*sqrt(2) |
| `lcm_three` | ood_test | no | 0/50 yes | has derived 180 |
| `euler_phi_30` | ood_test | no | 0/50 yes | has derived 8 |
| `partition_ordered` | ood_test | no | 0/50 yes | has reached 64 |

Conversely, positives frequently contain the answer *plus* verification. Most sharply,
`partition_ordered` carries **both labels on the same question**, with the model having
reached 64 in each case (negatives at `prefix_idx` 0, positives at `prefix_idx` 2).
`lcm_three` is a second such pair (positive 47/50 at rollout 20, negative 0/50 at rollout 22).

This is the audit's most useful product: a preregistered control for

> **answer obtained =/= termination imminent**

which speaks directly to the Hard-CoT / causal-analysis reconciliation rather than being a
generic probe result.

### The control set is larger than it first appears

Questions carrying *both* labels within a split:

| split | questions with both labels | of total |
|---|---|---|
| val | 11 | 30 |
| test | 17 | 22 |
| ood_test | **16** | 32 |

Half of the OOD questions are matched positive/negative pairs. That enables a **within-question
paired analysis** on `ood_test` at no extra compute — the prompt and most of the reasoning are
held fixed, so a probe score difference cannot be attributed to topic or domain.

ood_test questions with both labels:
`binary_no_consec`, `chessboard_squares`, `colored_balls`, `committee_women`, `crt_three`,
`dice_even_product`, `gcd_2024`, `josephus_7`, `lcm_three`, `matrix_power_entry`,
`modular_inverse_7_11`, `partition_ordered`, `permutation_mississippi`, `polygon_diagonals`,
`sum_divisors_72`, `triangle_vertices`.

**Do not analyse these yet.** They are registered here as a planned falsification/control
*before* any activation result exists. Preserve that ordering: reproduce first, explain second.

## 7. Sampling consequences for extraction

- train is heavily clustered: 796 questions, median 53 records/question, **max 452**. A naive
  random 4k subset would be dominated by a few prolific questions — stratify by family and
  sample deterministically.
- train prefixes have a long tail absent from eval: est. tokens median 697, q90 4,927,
  q99 10,491, **max 16,569** (measured at 1.36 tokens/word on 300 real prefixes), versus eval
  `token_length` median 915, max 2,903. Batch by length bucket.
- eval clustering is mild: ood_test is 58 records over 32 questions (median 1.5).
