#!/usr/bin/env python3
"""R006 (final experiment): score-level nuisance control on `test`.

This is a **score-level** control, not a representation-level one. It removes a
fitted linear association from the frozen scalar probe score. Nothing is projected
out of the 5,120-dimensional activation, and no probe is refit.

    s      frozen depth-40 probe decision score (R001, val-selected depth, D010)
    L      token_length (released field)
    T      think_logprob (cached D007 feature)

Nuisance fit uses `val` ONLY, and uses no labels at all:

    s = b0 + bL * z_val(L) + bT * z_val(T) + eps            (OLS on val rows)

with the standardisation constants taken from val. On `test`:

    s_resid = s - bL * z_val(L) - bT * z_val(T)

No reorientation or calibration on test labels. `ood_test` is closed (D011).

Usage:
    python src/nuisance_control.py --run-id r006_nuisance_control
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score

import task1_data as T
from fit_task1_probe import N_BOOTSTRAP, cluster_bootstrap_auroc
from fit_output_baselines import load_output_features, paired_cluster_bootstrap_delta

FIT_SPLIT = "val"
EVAL_SPLIT = "test"
CLOSED = {"ood_test"}
RAW_TEST_AUROC_R001 = 0.9085992428339643      # frozen reference from R001


def concordance_per_question(rows, score):
    per_q = {}
    by_q = defaultdict(lambda: {"yes": [], "no": []})
    for r in rows:
        by_q[r["question_id"]][r["label"]].append(score[r["key"]])
    for q, d in by_q.items():
        if not (d["yes"] and d["no"]):
            continue
        pos, neg = np.array(d["yes"]), np.array(d["no"])
        diff = pos[:, None] - neg[None, :]
        per_q[q] = float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
    return per_q


def paired_question_bootstrap(per_q_a, per_q_b, n_boot, seed):
    """Macro-concordance CIs for two scores and their delta, on identical draws."""
    qids = sorted(set(per_q_a) & set(per_q_b))
    rng = np.random.default_rng(seed)
    a_vals, b_vals, d_vals = [], [], []
    for _ in range(n_boot):
        drawn = rng.choice(qids, size=len(qids), replace=True)
        a = float(np.mean([per_q_a[q] for q in drawn]))
        b = float(np.mean([per_q_b[q] for q in drawn]))
        a_vals.append(a); b_vals.append(b); d_vals.append(a - b)
    pct = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return {"a_ci": pct(a_vals), "b_ci": pct(b_vals), "delta_ci": pct(d_vals),
            "frac_delta_positive": float((np.asarray(d_vals) > 0).mean()),
            "n_questions": len(qids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="r006_nuisance_control")
    ap.add_argument("--activation-run-id", default="r001_qwen32b")
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir = T.REPO / "artifacts/runs" / args.activation_run_id
    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((src_dir / "config.json").read_text())

    F, all_rows = load_output_features(src_dir, src_cfg["output_features"])
    col = {n: i for i, n in enumerate(src_cfg["output_features"])}
    think = {(r["split"], r["filename"]): float(F[i, col["think_logprob"]])
             for i, r in enumerate(all_rows)}
    with (src_dir / "probe_scores.csv").open() as fh:
        act = {(r["split"], r["filename"]): float(r["score"])
               for r in csv.DictReader(fh) if int(r["depth"]) == args.depth}

    length = {}
    for split in (FIT_SPLIT, EVAL_SPLIT):
        if split in CLOSED:
            raise SystemExit(f"{split} is closed (D011)")
        for path in sorted((T.DATA_ROOT / "qwen-3-32b" / split).glob("*.json")):
            length[(split, path.name)] = float(json.loads(path.read_text())["token_length"])

    rows = {s: [dict(r, key=(r["split"], r["filename"]))
                for r in all_rows if r["split"] == s]
            for s in (FIT_SPLIT, EVAL_SPLIT)}

    # ---- nuisance fit on val only, no labels used ----
    kfit = [r["key"] for r in rows[FIT_SPLIT]]
    Lf = np.array([length[k] for k in kfit])
    Tf = np.array([think[k] for k in kfit])
    sf = np.array([act[k] for k in kfit])
    mu = {"L": float(Lf.mean()), "T": float(Tf.mean())}
    sd = {"L": float(Lf.std(ddof=0)), "T": float(Tf.std(ddof=0))}
    Zf = np.column_stack([np.ones_like(Lf), (Lf - mu["L"]) / sd["L"], (Tf - mu["T"]) / sd["T"]])
    beta, *_ = np.linalg.lstsq(Zf, sf, rcond=None)
    pred = Zf @ beta
    r2 = float(1 - ((sf - pred) ** 2).sum() / ((sf - sf.mean()) ** 2).sum())
    b0, bL, bT = (float(b) for b in beta)
    print(f"nuisance fit on {FIT_SPLIT} (n={len(kfit)}, labels unused): "
          f"s = {b0:.3f} + {bL:.3f}*z(L) + {bT:.3f}*z(T),  R^2 = {r2:.3f}")

    # single-nuisance fits (secondary)
    def fit_one(vals):
        Z = np.column_stack([np.ones_like(vals), (vals - vals.mean()) / vals.std(ddof=0)])
        b, *_ = np.linalg.lstsq(Z, sf, rcond=None)
        p = Z @ b
        return (float(b[0]), float(b[1]), float(vals.mean()), float(vals.std(ddof=0)),
                float(1 - ((sf - p) ** 2).sum() / ((sf - sf.mean()) ** 2).sum()))
    bL0, bL1, muL1, sdL1, r2L = fit_one(Lf)
    bT0, bT1, muT1, sdT1, r2T = fit_one(Tf)
    print(f"  length-only   R^2 = {r2L:.3f}   think-only R^2 = {r2T:.3f}")

    # ---- build scores on test ----
    ke = [r["key"] for r in rows[EVAL_SPLIT]]
    raw = {k: act[k] for k in ke}
    zL = {k: (length[k] - mu["L"]) / sd["L"] for k in ke}
    zT = {k: (think[k] - mu["T"]) / sd["T"] for k in ke}
    joint = {k: raw[k] - bL * zL[k] - bT * zT[k] for k in ke}
    len_only = {k: raw[k] - bL1 * ((length[k] - muL1) / sdL1) for k in ke}
    think_only = {k: raw[k] - bT1 * ((think[k] - muT1) / sdT1) for k in ke}

    y = np.array([1 if r["label"] == "yes" else 0 for r in rows[EVAL_SPLIT]])
    groups = np.array([r["question_id"] for r in rows[EVAL_SPLIT]])
    vec = lambda d: np.array([d[k] for k in ke])

    variants = {"raw": raw, "residual_joint": joint,
                "residual_length_only": len_only,
                "residual_think_only": think_only}
    out = {}
    for name, d in variants.items():
        sc = vec(d)
        auc = float(roc_auc_score(y, sc))
        lo, hi, _ = cluster_bootstrap_auroc(y, sc, groups, args.n_bootstrap, args.seed)
        per_q = concordance_per_question(rows[EVAL_SPLIT], d)
        macro = float(np.mean(list(per_q.values())))
        out[name] = {"pooled_auroc": auc, "auroc_ci": [lo, hi],
                     "macro_within_question_concordance": macro,
                     "per_question": per_q}
        print(f"{name:22s} pooled AUROC={auc:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"within-question={macro:.3f}")

    # ---- paired deltas, raw vs each residual ----
    per_q_raw = out["raw"]["per_question"]
    for name in ("residual_joint", "residual_length_only", "residual_think_only"):
        d_auc = paired_cluster_bootstrap_delta(y, vec(raw), vec(variants[name]),
                                               groups, args.n_bootstrap, args.seed)
        d_auc["delta_point"] = out["raw"]["pooled_auroc"] - out[name]["pooled_auroc"]
        d_con = paired_question_bootstrap(per_q_raw, out[name]["per_question"],
                                          args.n_bootstrap, args.seed)
        d_con["delta_point"] = (out["raw"]["macro_within_question_concordance"]
                                - out[name]["macro_within_question_concordance"])
        out[name]["delta_auroc_vs_raw"] = d_auc
        out[name]["delta_concordance_vs_raw"] = d_con
        print(f"  {name:22s} AUROC(raw)-AUROC(res) = {d_auc['delta_point']:+.3f} "
              f"[{d_auc['ci_lo']:+.3f}, {d_auc['ci_hi']:+.3f}]   "
              f"concordance delta = {d_con['delta_point']:+.3f} "
              f"[{d_con['delta_ci'][0]:+.3f}, {d_con['delta_ci'][1]:+.3f}]")

    drop = RAW_TEST_AUROC_R001 - out["residual_joint"]["pooled_auroc"]
    band = ("within ~0.03 of raw -- not explained by these fitted linear associations"
            if abs(drop) <= 0.03 else
            "0.03-0.10 drop -- nuisances explain part but not all of the signal"
            if drop <= 0.10 else
            "drop > 0.10 -- materially nuisance-dependent")
    print(f"\nfrozen band (declared before the run): joint residual vs R001 raw "
          f"{RAW_TEST_AUROC_R001:.3f} -> drop {drop:+.3f}: {band}")

    for v in out.values():
        v.pop("per_question", None)
    report = {
        "run_id": args.run_id,
        "kind": "score-level nuisance control (NOT representation-level)",
        "depth": args.depth,
        "fit_split": FIT_SPLIT, "eval_split": EVAL_SPLIT,
        "closed_splits": sorted(CLOSED),
        "labels_used_in_nuisance_fit": False,
        "nuisance_fit": {
            "joint": {"b0": b0, "beta_zL": bL, "beta_zT": bT, "r2": r2,
                      "val_mean": mu, "val_std": sd, "n": len(kfit)},
            "length_only": {"beta_z": bL1, "r2": r2L},
            "think_only": {"beta_z": bT1, "r2": r2T},
        },
        "test": out,
        "frozen_band": {"raw_reference_auroc": RAW_TEST_AUROC_R001,
                        "joint_residual_drop": float(drop), "verdict": band},
        "limitation": (
            "Survival of this control does NOT prove a termination-specific latent "
            "state and does NOT remove nonlinear representations of length, "
            "reasoning progress or output propensity from the activation. It shows "
            "only robustness to the fitted linear nuisance associations tested here, "
            "on one in-domain split, at one val-selected depth."),
        "bootstrap": {"n": args.n_bootstrap,
                      "auroc_unit": "question cluster", "concordance_unit": "question"},
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "source_run": args.activation_run_id,
        "depth": args.depth, "fit_split": FIT_SPLIT, "eval_split": EVAL_SPLIT,
        "nuisances": ["token_length", "think_logprob"],
        "refit_probe": False, "labels_used_in_nuisance_fit": False,
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "source_sample_sha256": src_cfg["sample_sha256"],
    }, indent=2))
    print(f"\nwrote {run_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
