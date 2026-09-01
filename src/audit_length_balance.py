#!/usr/bin/env python3
"""R005: benchmark-construction audit -- global vs question-conditional length balance.

No model is fit here and no new predictor is built. This reads the released
evaluation records and the released builder, and reports pre-specified aggregate
quantities only. Individual OOD examples are not inspected (D011).

The question: the released builders balance token length in GLOBAL 500-token
buckets after per-prompt class balancing. Does that leave a question-conditional
length-label association?

Outputs, per behavioural split:
  a  pooled token_length AUROC
  b  macro within-question length concordance (questions carrying both labels)
  c  eligible question count
  d  YES/NO pair count
  e  mean and median within-question token_length(YES) - token_length(NO)
  f  the released split's own global 500-token bucket table (the balance the
     builder actually enforces)

Plus a deterministic toy example showing the two can coexist, which is
explanatory only and is not evidence about this dataset.

Usage:
    python src/audit_length_balance.py --run-id r005_length_balance_audit
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score

import task1_data as T

SPLITS = ["val", "test", "ood_test"]
BUCKET = 500                      # the builder's bucket size


def load_lengths(split):
    rows = []
    for path in sorted((T.DATA_ROOT / "qwen-3-32b" / split).glob("*.json")):
        rec = json.loads(path.read_text())
        rows.append({"question_id": rec["question_id"],
                     "label": rec["label"],
                     "token_length": float(rec["token_length"])})
    return rows


def split_report(rows):
    y = np.array([1 if r["label"] == "yes" else 0 for r in rows])
    L = np.array([r["token_length"] for r in rows])

    by_q = defaultdict(lambda: {"yes": [], "no": []})
    for r in rows:
        by_q[r["question_id"]][r["label"]].append(r["token_length"])
    both = {q: d for q, d in by_q.items() if d["yes"] and d["no"]}

    per_q, diffs = {}, []
    for q, d in both.items():
        pos, neg = np.array(d["yes"]), np.array(d["no"])
        diff = pos[:, None] - neg[None, :]
        per_q[q] = float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
        diffs.extend(diff.ravel().tolist())

    buckets = defaultdict(lambda: {"yes": 0, "no": 0})
    for r in rows:
        buckets[int(r["token_length"]) // BUCKET][r["label"]] += 1

    return {
        "n_rows": len(rows),
        "n_questions": len(by_q),
        "n_questions_both_labels": len(both),
        "n_pairs": len(diffs),
        "pooled_length_auroc": float(roc_auc_score(y, L)),
        "macro_within_question_length_concordance":
            float(np.mean(list(per_q.values()))) if per_q else float("nan"),
        "within_question_length_diff_mean": float(np.mean(diffs)) if diffs else float("nan"),
        "within_question_length_diff_median": float(np.median(diffs)) if diffs else float("nan"),
        "global_buckets": {f"[{b*BUCKET}-{(b+1)*BUCKET})": dict(v)
                           for b, v in sorted(buckets.items())},
        "global_yes_mean_length": float(L[y == 1].mean()),
        "global_no_mean_length": float(L[y == 0].mean()),
    }


def toy_example():
    """Deterministic demonstration: pooled AUROC < 0.5, conditional concordance 1.0.

    Two questions on different length scales, contributing different numbers of
    each class. Within each question the YES prefix is the longer one; pooled
    across questions, a long question's NO rows are longer than a short
    question's YES rows, so the marginal ordering is destroyed.

    Explanatory only. This is arithmetic, not evidence about the released data.
    """
    rows = ([("Q_long", 1, 3000)] + [("Q_long", 0, 2900)] * 3
            + [("Q_short", 1, 1000)] * 3 + [("Q_short", 0, 900)])
    y = np.array([r[1] for r in rows])
    L = np.array([float(r[2]) for r in rows])
    per_q = {}
    for q in sorted({r[0] for r in rows}):
        pos = np.array([r[2] for r in rows if r[0] == q and r[1] == 1], dtype=float)
        neg = np.array([r[2] for r in rows if r[0] == q and r[1] == 0], dtype=float)
        d = pos[:, None] - neg[None, :]
        per_q[q] = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return {
        "rows": [{"question": q, "y": int(a), "token_length": l} for q, a, l in rows],
        "pooled_length_auroc": float(roc_auc_score(y, L)),
        "within_question_concordance": per_q,
        "macro_within_question_concordance": float(np.mean(list(per_q.values()))),
        "note": ("YES is the longer prefix in every question (conditional "
                 "concordance 1.0) while pooled AUROC is below chance, because "
                 "the long question contributes mostly NO rows and the short "
                 "question mostly YES rows. A global balancing procedure sees "
                 "the pooled view."),
    }


def make_figure(path, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in SPLITS if s in table]
    x = np.arange(len(splits))
    w = 0.26
    pooled = [table[s]["pooled_length_auroc"] for s in splits]
    within = [table[s]["macro_within_question_length_concordance"] for s in splits]
    act = [table[s].get("activation_within_question_concordance", np.nan) for s in splits]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhline(0.5, color="0.45", lw=1.0, ls=":", zorder=1)
    b1 = ax.bar(x - w, pooled, w, color="#8C8C8C", label="length, pooled AUROC")
    b2 = ax.bar(x, within, w, color="#E08A1E", label="length, within-question concordance")
    b3 = ax.bar(x + w, act, w, color="#2E6DB4", label="depth-40 probe, within-question")
    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            if np.isfinite(h):
                ax.text(r.get_x() + r.get_width() / 2, h + 0.015, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUROC / concordance")
    ax.set_title("R005: global length balance holds in every split; the "
                 "within-question\nlength-label association appears only in ood_test",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="r005_length_balance_audit")
    args = ap.parse_args()

    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    for split in SPLITS:
        table[split] = split_report(load_lengths(split))

    # frozen activation within-question concordance already measured elsewhere
    r003 = json.loads((T.REPO / "artifacts/runs/r003_paired_ood/metrics.json").read_text())
    r004 = json.loads((T.REPO / "artifacts/runs/r004_length_control/metrics.json").read_text())
    table["ood_test"]["activation_within_question_concordance"] = \
        r003["macro_paired_concordance"]["activation_depth40"]
    table["ood_test"]["activation_source"] = "R003"
    for split in ("test", "val"):
        table[split]["activation_within_question_concordance"] = \
            r004["splits"][split]["A_macro_concordance"]["activation_depth40"]
        table[split]["activation_source"] = "R004"

    for split in SPLITS:
        d = table[split]
        print(f"\n===== {split} =====")
        print(f"  rows={d['n_rows']}  questions={d['n_questions']}  "
              f"both-label questions={d['n_questions_both_labels']}  "
              f"pairs={d['n_pairs']}")
        print(f"  a pooled length AUROC                   = {d['pooled_length_auroc']:.3f}")
        print(f"  b macro within-question length concord. = "
              f"{d['macro_within_question_length_concordance']:.3f}")
        print(f"    (frozen depth-40 probe, same metric)  = "
              f"{d['activation_within_question_concordance']:.3f} [{d['activation_source']}]")
        print(f"  e within-question length diff YES-NO: mean="
              f"{d['within_question_length_diff_mean']:.0f} "
              f"median={d['within_question_length_diff_median']:.0f} tokens")
        print(f"  global mean length: yes={d['global_yes_mean_length']:.0f} "
              f"no={d['global_no_mean_length']:.0f}")
        print("  f global 500-token buckets (the balance the builder enforces):")
        for k, v in d["global_buckets"].items():
            print(f"      {k:14s} yes={v.get('yes', 0):3d}  no={v.get('no', 0):3d}")

    toys = {"unequal_counts": toy_example()}
    print("\n===== toy example (explanatory only, not evidence) =====")
    t = toys["unequal_counts"]
    print(f"  pooled length AUROC = {t['pooled_length_auroc']:.3f}; "
          f"macro within-question concordance = "
          f"{t['macro_within_question_concordance']:.3f}")

    builder = {
        "files": [
            "cot-proxy-tasks/src/tasks/reasoning_termination/run_build_eval_v8.py",
            "cot-proxy-tasks/src/tasks/reasoning_termination/run_build_math_val_v8.py",
            "cot-proxy-tasks/src/tasks/reasoning_termination/run_build_ood_val_v8.py",
        ],
        "upstream_commit": "4482324",
        "read_directly": True,
        "global_length_filter": {"LENGTH_MIN": 500, "LENGTH_MAX": 3000,
                                 "where": "step 1b, before any balancing"},
        "per_prompt_step": ("step 4 takes min(n_yes, n_no) items per prompt, sorted "
                            "by mean_yes_position (yes) and no_count (no) -- i.e. by "
                            "label quality, NOT by token length"),
        "length_balancing": ("step 5a-5e operates on 500-token buckets computed over "
                             "the WHOLE selected set (_bucket_stats iterates the flat "
                             "list, never grouping by prompt): 5a trims unpaired "
                             "singles, 5b removes whole pairs, 5c adds minority "
                             "singles, 5d stratified bucket trim, 5e global rebalance"),
        "within_prompt_length_matching": ("absent -- no step in any of the three "
                                          "builders compares a yes item's token_count "
                                          "with a no item's token_count from the same "
                                          "prompt"),
        "builder_own_comment": ("step 5d is commented 'stratified bucket trim -- remove "
                                "majority-class excess per bucket to eliminate the "
                                "systematic length skew (no->short, yes->long)', so "
                                "the skew was known and was addressed globally"),
    }

    report = {"run_id": args.run_id,
              "no_model_fit": True,
              "builder": builder,
              "splits": table,
              "toy_example": toys,
              "conclusion_supported": (
                  "Global token-length balance is insufficient to rule out a "
                  "question-conditional length shortcut. In the released ood_test "
                  "math set, absolute length is weak pooled across questions but "
                  "almost perfectly orders termination labels within questions."),
              "conclusions_NOT_supported": [
                  "the activation probe is merely a length detector",
                  "the published probe result is invalid",
                  "length explains R001 generally",
              ]}
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "splits": SPLITS, "bucket_size": BUCKET,
        "inputs": "released rollout json (token_length, label) + the v8 builders",
        "fits_nothing": True,
    }, indent=2))
    fig = T.REPO / "artifacts/figures/r005_length_balance.png"
    make_figure(fig, table)
    print(f"\nwrote {run_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
