#!/usr/bin/env python3
"""Stage 0 preflight for a possible steering experiment. NO GPU, NO behavioural outcome.

This is a feasibility diagnostic, not an experiment. It decides one thing: is an
edit that moves the frozen depth-40 probe score by a preregistered amount a sane
perturbation of the residual stream, or is it destructive? If destructive, the
project stops at R006 and D012 stands.

Steps:
  1. Re-run the deterministic frozen probe fit at depth 40 (same code path, same
     C, same train rows) and verify its decision scores reproduce the committed
     R001 `probe_scores.csv`. Abort on mismatch -- without that, beta is not the
     frozen probe's direction.
  2. Recover the direction in RAW activation coordinates. sklearn's coef_ lives in
     standardised coordinates, so with z_j = (h_j - mu_j)/sigma_j and s = w.z + b:
         beta_raw[j] = w[j] / scaler.scale_[j]
  3. Verify numerically on cached activations that
         delta_h = (ds / ||beta_raw||^2) * beta_raw
     changes the frozen decision score by exactly ds.
  4. Magnitude ruler is the frozen depth-40 score SD on **val** -- the split already
     used for model selection. Never test.
  5. For ds in {0.5, 1.0, 2.0} * sigma_s(val), report ||delta_h|| / ||h|| across val
     activations: median, p90, max, plus absolute norms.

Nothing here observes a steering outcome, and no magnitude is chosen from one.

Usage:
    python src/steering_preflight.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import task1_data as T
from fit_task1_probe import C_GRID, SELECTION_SPLIT, load_run
from sklearn.metrics import roc_auc_score

DEPTH = 40                       # val-selected depth (D010); nothing else
CANDIDATES = [0.5, 1.0, 2.0]     # in units of the val score SD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="r007_steering_preflight")
    ap.add_argument("--activation-run-id", default="r001_qwen32b")
    args = ap.parse_args()

    src_dir = T.REPO / "artifacts/runs" / args.activation_run_id
    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src_dir / "config.json").read_text())
    di = cfg["depths"].index(DEPTH)

    X, rows = load_run(src_dir)
    Xd = X[:, di, :]
    split = np.array([r["split"] for r in rows])
    y = np.array([1 if r["label"] == "yes" else 0 for r in rows])
    masks = {s: split == s for s in T.SPLITS}

    # ---- 1. reproduce the frozen fit, exactly as fit_task1_probe.py does it ----
    per_c = {}
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(l1_ratio=0.0, C=C, max_iter=5000))
        clf.fit(Xd[masks["train"]], y[masks["train"]])
        s_val = clf.decision_function(Xd[masks[SELECTION_SPLIT]])
        per_c[C] = (roc_auc_score(y[masks[SELECTION_SPLIT]], s_val), clf)
    best_C = max(per_c, key=lambda c: per_c[c][0])
    clf = per_c[best_C][1]
    scores = clf.decision_function(Xd)
    print(f"refit depth {DEPTH}: selected C={best_C} "
          f"(R001 recorded C={[d['C'] for d in json.loads((src_dir/'metrics.json').read_text())['per_depth'] if d['depth']==DEPTH][0]})")

    with (src_dir / "probe_scores.csv").open() as fh:
        frozen = {(r["split"], r["filename"]): float(r["score"])
                  for r in csv.DictReader(fh) if int(r["depth"]) == DEPTH}
    mine = np.array([scores[i] for i in range(len(rows))])
    theirs = np.array([frozen[(r["split"], r["filename"])] for r in rows])
    max_abs = float(np.abs(mine - theirs).max())
    rel = max_abs / float(np.abs(theirs).max())
    print(f"frozen-score reproduction: max_abs_diff={max_abs:.3e} "
          f"(relative to max |score| {np.abs(theirs).max():.2f} -> {rel:.3e})")
    if rel > 1e-6:
        raise SystemExit("ABORT: refit does not reproduce the frozen R001 scores; "
                         "beta would not be the frozen probe's direction")

    # ---- 2. raw-space direction ----
    scaler = clf.named_steps["standardscaler"]
    logreg = clf.named_steps["logisticregression"]
    w = logreg.coef_.ravel().astype(np.float64)
    sigma = scaler.scale_.astype(np.float64)
    beta_raw = w / sigma
    nb2 = float(beta_raw @ beta_raw)
    print(f"||w|| (standardised) = {np.linalg.norm(w):.4f}   "
          f"||beta_raw|| = {np.sqrt(nb2):.6f}   hidden={beta_raw.size}")

    # ---- 3. verify the edit moves the frozen score by exactly ds ----
    probe_val = Xd[masks[SELECTION_SPLIT]]
    ds_check = 1.0
    dh_check = (ds_check / nb2) * beta_raw
    moved = (clf.decision_function(probe_val + dh_check)
             - clf.decision_function(probe_val))
    err = float(np.abs(moved - ds_check).max())
    # Tolerance is relative to the score scale, not absolute: the cached
    # activations are fp16 promoted to fp32 and the scores run to ~70, so a
    # float32 round-off of ~1e-5 here is the same order as the reproduction
    # check above and is not a derivation error.
    err_rel = err / float(np.abs(theirs).max())
    print(f"edit check: requested ds={ds_check}, achieved "
          f"[{moved.min():.6f}, {moved.max():.6f}], max_abs_err={err:.2e} "
          f"(relative to score scale: {err_rel:.2e})")
    if err_rel > 1e-5:
        raise SystemExit("ABORT: the raw-space edit does not move the score as derived")

    # ---- 4/5. magnitude ruler from the VAL score distribution ----
    s_val = scores[masks[SELECTION_SPLIT]]
    sigma_s = float(s_val.std(ddof=0))
    hnorm = np.linalg.norm(probe_val, axis=1)
    print(f"\nval frozen score: mean={s_val.mean():.3f} SD={sigma_s:.3f} "
          f"range=[{s_val.min():.2f}, {s_val.max():.2f}]  (n={s_val.size})")
    print(f"val depth-{DEPTH} residual norm ||h||: median={np.median(hnorm):.1f} "
          f"p90={np.percentile(hnorm, 90):.1f} max={hnorm.max():.1f}")

    table = []
    print(f"\n{'ds (val SD)':>12}  {'ds':>9}  {'||dh||':>10}  "
          f"{'ratio med':>10}  {'p90':>8}  {'max':>8}")
    for k in CANDIDATES:
        ds = k * sigma_s
        dh = (ds / nb2) * beta_raw
        n_dh = float(np.linalg.norm(dh))
        ratio = n_dh / hnorm
        row = {"ds_in_val_sd": k, "ds": ds, "norm_delta_h": n_dh,
               "ratio_median": float(np.median(ratio)),
               "ratio_p90": float(np.percentile(ratio, 90)),
               "ratio_max": float(ratio.max()),
               "ratio_min": float(ratio.min())}
        table.append(row)
        print(f"{k:>12.1f}  {ds:>9.3f}  {n_dh:>10.2f}  "
              f"{row['ratio_median']:>9.1%}  {row['ratio_p90']:>7.1%}  "
              f"{row['ratio_max']:>7.1%}")

    verdict = ("attractive: a 1-val-SD edit is a few percent of the residual norm"
               if table[1]["ratio_p90"] < 0.05 else
               "plausible but substantial: ~5-15% of the residual norm"
               if table[1]["ratio_p90"] < 0.15 else
               "ugly: tens of percent of the residual norm"
               if table[1]["ratio_p90"] < 0.40 else
               "DO NOT RUN: a 1-val-SD edit is ~half the residual stream")
    print(f"\nverdict at 1 val SD (p90 ratio {table[1]['ratio_p90']:.1%}): {verdict}")

    out = {
        "run_id": args.run_id,
        "kind": "feasibility preflight -- no GPU, no behavioural outcome, no experiment",
        "depth": DEPTH,
        "frozen_fit_reproduced": {"max_abs_diff": max_abs, "relative": rel,
                                  "selected_C": best_C},
        "beta_raw": {"norm": float(np.sqrt(nb2)), "hidden": int(beta_raw.size),
                     "derivation": "beta_raw[j] = coef_[j] / scaler.scale_[j]"},
        "edit_identity_check": {"requested_ds": ds_check, "max_abs_err": err,
                                "max_rel_err": err_rel},
        "val_score": {"sd": sigma_s, "mean": float(s_val.mean()),
                      "min": float(s_val.min()), "max": float(s_val.max()),
                      "n": int(s_val.size)},
        "val_residual_norm": {"median": float(np.median(hnorm)),
                              "p90": float(np.percentile(hnorm, 90)),
                              "max": float(hnorm.max())},
        "candidates": table,
        "verdict": verdict,
        "note": ("magnitudes are expressed in SDs of the frozen depth-40 score on "
                 "val, the split already used for selection; test is not used here "
                 "and no steering outcome informed any number above"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    np.save(run_dir / "beta_raw_depth40.npy", beta_raw)
    print(f"\nwrote {run_dir/'metrics.json'} and beta_raw_depth40.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
