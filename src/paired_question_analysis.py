#!/usr/bin/env python3
"""R003: preregistered within-question control on ood_test (D008).

16 of the 32 ood_test questions carry both a YES and a NO prefix. Within such a
question the prompt, the domain and usually the already-obtained answer are held
fixed, so a score that still separates YES from NO there is not separating topics.

This script REFITS NOTHING. It reads:
  - the frozen depth-40 activation probe scores from R001 (`probe_scores.csv`),
    depth 40 because that is the depth validation selected (D010), never 56;
  - `think_logprob` from the D007 features cached in the same R001 pass;
  - the released `token_length` as a sanity baseline.

Primary metric (preregistered): within-question pairwise concordance
    A_q = P(s(YES) > s(NO) | q) over all YES x NO pairs, ties counting 0.5,
macro-averaged over questions so every question weighs the same regardless of how
many prefixes it contributes.

Usage:
    python src/paired_question_analysis.py --run-id r003_paired_ood
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

import task1_data as T
from fit_task1_probe import N_BOOTSTRAP
from fit_output_baselines import load_output_features

SPLIT = "ood_test"
SCORES = ["activation_depth40", "think_logprob", "token_length"]


def concordance(pos: np.ndarray, neg: np.ndarray) -> float:
    """Fraction of YES x NO pairs ranked correctly; ties count 0.5."""
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def macro_concordance(qids, per_q):
    return float(np.mean([per_q[q] for q in qids]))


def question_bootstrap(qids, per_q_by_score, n_boot, seed):
    """Resample QUESTIONS with replacement; return CIs plus paired deltas.

    The same resampled question list scores every metric on every replicate, so
    the delta interval is paired rather than a comparison of two independent CIs.
    """
    rng = np.random.default_rng(seed)
    qids = np.asarray(qids)
    draws = [rng.choice(qids, size=len(qids), replace=True) for _ in range(n_boot)]
    out = {}
    for name, per_q in per_q_by_score.items():
        vals = np.array([macro_concordance(d, per_q) for d in draws])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name] = {"ci_lo": float(lo), "ci_hi": float(hi)}
    deltas = {}
    for name in per_q_by_score:
        if name == "activation_depth40":
            continue
        d = np.array([macro_concordance(dr, per_q_by_score["activation_depth40"])
                      - macro_concordance(dr, per_q_by_score[name]) for dr in draws])
        lo, hi = np.percentile(d, [2.5, 97.5])
        deltas[f"activation_depth40 - {name}"] = {
            "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_replicates_positive": float((d > 0).mean()),
            "n_replicates": int(len(d))}
    return out, deltas


def make_figure(path, rows, act_sd, think_sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["delta_think"])
    y = np.arange(len(rows))
    act = np.array([r["delta_activation"] / act_sd for r in rows])
    think = np.array([r["delta_think"] / think_sd for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvline(0, color="0.45", lw=1.0, zorder=1)
    for yi, a, t in zip(y, act, think):
        ax.plot([t, a], [yi, yi], color="0.82", lw=1.6, zorder=2)
    ax.scatter(think, y, s=46, color="#E08A1E", zorder=3,
               edgecolor="white", linewidth=0.8, label="think_logprob")
    ax.scatter(act, y, s=46, color="#2E6DB4", zorder=4,
               edgecolor="white", linewidth=0.8, label="depth-40 activation probe")
    ax.set_yticks(y)
    ax.set_yticklabels([r["question_id"] for r in rows], fontsize=7.5)
    ax.set_xlabel("within-question YES − NO mean score difference (SDs of that score)")
    ax.set_title("R003: within-question separation on ood_test\n"
                 "16 questions carrying both labels, ordered by think_logprob delta",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(axis="x", color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="r003_paired_ood")
    ap.add_argument("--activation-run-id", default="r001_qwen32b")
    ap.add_argument("--depth", type=int, default=40,
                    help="val-selected depth (D010); not the OOD-max depth")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir = T.REPO / "artifacts/runs" / args.activation_run_id
    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((src_dir / "config.json").read_text())

    # --- cached output features (D007) for the ood_test rows ---
    F, rows = load_output_features(src_dir, src_cfg["output_features"])
    col = {n: i for i, n in enumerate(src_cfg["output_features"])}
    think = {(r["split"], r["filename"]): float(F[i, col["think_logprob"]])
             for i, r in enumerate(rows)}

    # --- frozen depth-40 probe scores (R001), refit nothing ---
    with (src_dir / "probe_scores.csv").open() as fh:
        act = {(r["split"], r["filename"]): float(r["score"])
               for r in csv.DictReader(fh) if int(r["depth"]) == args.depth}
    if not act:
        raise SystemExit(f"no depth-{args.depth} scores in {src_dir}")

    # --- released token_length, straight from the read-only upstream files ---
    length = {}
    for path in sorted((T.DATA_ROOT / "qwen-3-32b" / SPLIT).glob("*.json")):
        rec = json.loads(path.read_text())
        length[(SPLIT, path.name)] = float(rec["token_length"])

    ood = [r for r in rows if r["split"] == SPLIT]
    by_q: dict[str, dict[str, list]] = {}
    for r in ood:
        key = (r["split"], r["filename"])
        d = by_q.setdefault(r["question_id"], {"yes": [], "no": []})
        d[r["label"]].append({
            "activation_depth40": act[key],
            "think_logprob": think[key],
            "token_length": length[key],
        })

    eligible = sorted(q for q, d in by_q.items() if d["yes"] and d["no"])
    print(f"{SPLIT}: {len(ood)} rows, {len(by_q)} questions, "
          f"{len(eligible)} carrying both labels")
    if not eligible:
        raise SystemExit("no question carries both labels")

    per_q_by_score = {s: {} for s in SCORES}
    per_question_rows = []
    for q in eligible:
        row = {"question_id": q,
               "n_yes": len(by_q[q]["yes"]), "n_no": len(by_q[q]["no"])}
        for s in SCORES:
            pos = np.array([e[s] for e in by_q[q]["yes"]])
            neg = np.array([e[s] for e in by_q[q]["no"]])
            per_q_by_score[s][q] = concordance(pos, neg)
            row[f"A_{s}"] = per_q_by_score[s][q]
            row[f"delta_{s}"] = float(pos.mean() - neg.mean())
        per_question_rows.append(row)

    # aliases used by the figure / diagnostics
    for row in per_question_rows:
        row["delta_activation"] = row["delta_activation_depth40"]
        row["delta_think"] = row["delta_think_logprob"]

    macro = {s: macro_concordance(eligible, per_q_by_score[s]) for s in SCORES}
    cis, deltas = question_bootstrap(eligible, per_q_by_score,
                                     args.n_bootstrap, args.seed)
    for s in SCORES:
        print(f"macro paired concordance  {s:20s} = {macro[s]:.3f} "
              f"[{cis[s]['ci_lo']:.3f}, {cis[s]['ci_hi']:.3f}]")
    for k, v in deltas.items():
        print(f"paired question-bootstrap  {k:44s} = "
              f"{macro['activation_depth40'] - macro[k.split(' - ')[1]]:+.3f} "
              f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  "
              f"P(>0)={v['frac_replicates_positive']:.3f}")

    # --- per-question sign diagnostics ---
    d_act = np.array([r["delta_activation"] for r in per_question_rows])
    d_think = np.array([r["delta_think"] for r in per_question_rows])
    d_len = np.array([r["delta_token_length"] for r in per_question_rows])
    n = len(eligible)
    sign = stats.binomtest(int((d_act > 0).sum()), n, 0.5, alternative="two-sided")
    rho = stats.spearmanr(d_act, d_think)

    discordant = [r for r in per_question_rows if r["delta_think"] <= 0]
    disc_act_pos = sum(1 for r in discordant if r["delta_activation"] > 0)
    disc_conc = ([r["A_activation_depth40"] for r in discordant] or [float("nan")])

    print(f"\ndelta > 0:  activation {int((d_act>0).sum())}/{n}   "
          f"think_logprob {int((d_think>0).sum())}/{n}   "
          f"token_length {int((d_len>0).sum())}/{n}")
    print(f"sign test on activation deltas: p={sign.pvalue:.4f} (two-sided binomial)")
    print(f"Spearman(delta_activation, delta_think) = {rho.statistic:+.3f} "
          f"(p={rho.pvalue:.3f})")
    print(f"questions where the output cue is wrong/uninformative "
          f"(delta_think <= 0): {len(discordant)}; activation still positive on "
          f"{disc_act_pos}/{len(discordant)}; "
          f"mean activation concordance there = {np.mean(disc_conc):.3f}")

    # --- outputs ---
    metrics = {
        "run_id": args.run_id,
        "split": SPLIT,
        "depth": args.depth,
        "depth_note": "val-selected depth (D010); the OOD-max depth 56 is not used",
        "refit": "none -- frozen R001 probe scores and cached D007 features",
        "n_rows_ood": len(ood),
        "n_questions_total": len(by_q),
        "n_questions_both_labels": n,
        "macro_paired_concordance": macro,
        "macro_paired_concordance_ci": cis,
        "paired_deltas": deltas,
        "bootstrap": {"n": args.n_bootstrap, "unit": "question"},
        "sign_diagnostics": {
            "n_delta_positive": {"activation_depth40": int((d_act > 0).sum()),
                                 "think_logprob": int((d_think > 0).sum()),
                                 "token_length": int((d_len > 0).sum())},
            "n_questions": n,
            "activation_sign_test_p_two_sided": float(sign.pvalue),
            "spearman_delta_act_vs_delta_think": {
                "rho": float(rho.statistic), "p": float(rho.pvalue)},
        },
        "output_cue_discordant": {
            "definition": "questions with delta_think_logprob <= 0",
            "n": len(discordant),
            "n_activation_delta_positive": disc_act_pos,
            "mean_activation_concordance": float(np.mean(disc_conc)),
            "question_ids": [r["question_id"] for r in discordant],
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "source_run": args.activation_run_id,
        "split": SPLIT, "depth": args.depth, "scores": SCORES,
        "primary_metric": "macro-averaged within-question YESxNO concordance, ties 0.5",
        "n_bootstrap": args.n_bootstrap, "bootstrap_unit": "question",
        "seed": args.seed,
        "source_sample_sha256": src_cfg["sample_sha256"],
    }, indent=2))

    cols = (["question_id", "n_yes", "n_no"]
            + [f"A_{s}" for s in SCORES] + [f"delta_{s}" for s in SCORES])
    with (run_dir / "per_question.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(per_question_rows, key=lambda r: r["delta_think"]))

    all_act = np.array([act[(SPLIT, r["filename"])] for r in ood])
    all_think = np.array([think[(SPLIT, r["filename"])] for r in ood])
    fig_path = T.REPO / "artifacts/figures/r003_paired_questions.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(fig_path, per_question_rows, all_act.std(), all_think.std())

    print(f"\nwrote {run_dir/'metrics.json'}, {run_dir/'per_question.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
