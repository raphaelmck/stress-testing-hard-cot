#!/usr/bin/env python3
"""R002: output-level (H0) baselines from the D007 features cached during R001.

The D007 embargo lifted when the R001 entry was committed (7a0123f). This script
loads ONLY the `output_features` array from the R001 shards -- no activations are
fit here; the activation side of the comparison is read back from the frozen
`probe_scores.csv` R001 wrote.

Frozen decisions this script implements:
  D005  train on `train` only; C selected on `val` AUROC only; `ood_test` is
        touched exactly once, at evaluation.
  D006  every CI is a bootstrap over question_id CLUSTERS.
  R002  the primary activation comparator is the **val-selected** depth (40,
        OOD 0.904), NOT the descriptive maximum over the five predeclared
        depths (56, OOD 0.964). Comparing a val-selected output baseline
        against an OOD-selected activation depth would hand the activation
        model a post-hoc advantage.
  R002  `top1_token_id` is NEVER used as a numeric feature -- token ids have no
        ordinal meaning. It enters only as `top1_is_think = (id == </think>)`.
  R002  the headline comparison is a PAIRED question-clustered bootstrap of
        delta = AUROC(activation) - AUROC(output), using the same resampled
        clusters for both score vectors, not two independent intervals.

Usage:
    python src/fit_output_baselines.py --run-id r002_output_baseline
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import task1_data as T
from fit_task1_probe import C_GRID, N_BOOTSTRAP, SELECTION_SPLIT, cluster_bootstrap_auroc

PRIMARY_SCALAR = "think_logprob"        # the literal next-token propensity for </think>
MULTIVARIATE = ["think_logit", "think_logprob", "think_margin",
                "next_token_entropy", "top1_logprob", "top1_is_think"]
SCALARS = ["think_logprob", "think_logit", "think_margin",
           "next_token_entropy", "top1_logprob", "top1_is_think"]


def load_output_features(run_dir: pathlib.Path, feature_names: list[str]):
    """Load only `output_features` + row metadata from every R001 shard."""
    shards = sorted((run_dir / "activations").glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards found in {run_dir/'activations'}")
    feats, rows = [], []
    for npz_path in shards:
        with np.load(npz_path) as z:
            if "output_features" not in z:
                raise SystemExit(f"{npz_path.name} has no output_features array")
            feats.append(z["output_features"])          # [n, n_features]
        with npz_path.with_suffix(".csv").open() as fh:
            rows.extend(list(csv.DictReader(fh)))
    F = np.concatenate(feats).astype(np.float64)
    if len(rows) != F.shape[0]:
        raise SystemExit(f"metadata/feature length mismatch: {len(rows)} vs {F.shape[0]}")
    if F.shape[1] != len(feature_names):
        raise SystemExit(f"expected {len(feature_names)} features, got {F.shape[1]}")
    if not np.isfinite(F).all():
        raise SystemExit("non-finite value in the cached output features")
    return F, rows


def load_activation_scores(run_dir: pathlib.Path, depth: int):
    """Per-example decision-function scores of the R001 probe at one depth."""
    with (run_dir / "probe_scores.csv").open() as fh:
        keyed = {(r["split"], r["filename"]): float(r["score"])
                 for r in csv.DictReader(fh) if int(r["depth"]) == depth}
    if not keyed:
        raise SystemExit(f"no probe scores at depth {depth} in {run_dir}")
    return keyed


def paired_cluster_bootstrap_delta(y, score_a, score_b, groups,
                                   n_boot=N_BOOTSTRAP, seed=0):
    """CI for AUROC(a) - AUROC(b) on the SAME resampled question clusters.

    Two independent intervals would ignore that both scores are evaluated on the
    same rows and would be far too wide for a difference. Single-label replicates
    are discarded for both scores together, never coerced.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas, discarded = [], 0
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yy = y[idx]
        if yy.min() == yy.max():
            discarded += 1
            continue
        deltas.append(roc_auc_score(yy, score_a[idx]) - roc_auc_score(yy, score_b[idx]))
    if len(deltas) < n_boot // 2:
        raise RuntimeError(f"only {len(deltas)}/{n_boot} paired replicates were valid")
    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_point": None,          # filled in by the caller (full-sample value)
            "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_replicates_positive": float((d > 0).mean()),
            "n_replicates": len(deltas), "n_discarded": discarded}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="r002_output_baseline")
    ap.add_argument("--activation-run-id", default="r001_qwen32b")
    ap.add_argument("--primary-depth", type=int, default=40,
                    help="val-selected depth from R001; NOT the OOD-max depth")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir = T.REPO / "artifacts/runs" / args.activation_run_id
    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    src_cfg = json.loads((src_dir / "config.json").read_text())
    cached_names = src_cfg["output_features"]
    F, rows = load_output_features(src_dir, cached_names)

    # ---- derive top1_is_think; never use the raw token id as a number ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src_cfg["model"], revision=src_cfg["revision"])
    think_ids = tok("</think>", add_special_tokens=False)["input_ids"]
    if len(think_ids) != 1:
        raise SystemExit(f"</think> is not a single token: {think_ids}")
    think_id = int(think_ids[0])
    print(f"</think> token id={think_id}")

    col = {n: i for i, n in enumerate(cached_names)}
    feat = {n: F[:, col[n]] for n in cached_names if n != "top1_token_id"}
    top1_ids = F[:, col["top1_token_id"]].astype(np.int64)
    feat["top1_is_think"] = (top1_ids == think_id).astype(np.float64)

    split = np.array([r["split"] for r in rows])
    y = np.array([1 if r["label"] == "yes" else 0 for r in rows])
    groups = np.array([r["question_id"] for r in rows])
    masks = {s: split == s for s in T.SPLITS}
    print(f"loaded {F.shape[0]} rows of cached output features "
          f"({', '.join(cached_names)})")
    print(f"  top1 is </think> in {int(feat['top1_is_think'].sum())}/{len(y)} rows")

    results = {}
    score_rows = []

    def evaluate(name, scores, extra=None):
        """AUROC on every split + clustered CIs, for one score vector."""
        row = {"baseline": name}
        if extra:
            row.update(extra)
        for s in T.SPLITS:
            m = masks[s]
            if not m.any():
                continue
            row[f"{s}_auroc"] = float(roc_auc_score(y[m], scores[m]))
            if s != "train":
                lo, hi, n_ok = cluster_bootstrap_auroc(
                    y[m], scores[m], groups[m], args.n_bootstrap, args.seed)
                row[f"{s}_ci_lo"], row[f"{s}_ci_hi"] = lo, hi
        for k in range(len(y)):
            score_rows.append({
                "baseline": name, "split": rows[k]["split"],
                "filename": rows[k]["filename"],
                "question_id": rows[k]["question_id"],
                "label": rows[k]["label"], "y": int(y[k]),
                "score": float(scores[k]),
            })
        return row

    # ---- 1/2. single scalars, used directly as the score for label=yes ----
    # Orientation is fixed on val when it is not obvious a priori; it is never
    # chosen on ood_test. The raw (unoriented) AUROC is reported alongside so
    # the orientation decision stays visible.
    scalar_rows = []
    for name in SCALARS:
        raw = feat[name]
        val_auc_raw = roc_auc_score(y[masks[SELECTION_SPLIT]], raw[masks[SELECTION_SPLIT]])
        sign = 1.0 if val_auc_raw >= 0.5 else -1.0
        row = evaluate(f"scalar:{name}", sign * raw,
                       {"orientation": ("as-is" if sign > 0 else "negated"),
                        "orientation_chosen_on": SELECTION_SPLIT,
                        "val_auroc_unoriented": float(val_auc_raw),
                        "primary": name == PRIMARY_SCALAR})
        scalar_rows.append(row)
        print(f"scalar {name:19s} orient={row['orientation']:7s} " + "  ".join(
            f"{s}={row.get(f'{s}_auroc', float('nan')):.3f}" for s in T.SPLITS))

    # ---- 3. multivariate output-only baseline ----
    Xo = np.column_stack([feat[n] for n in MULTIVARIATE])
    per_c = {}
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(l1_ratio=0.0, C=C, max_iter=5000))
        clf.fit(Xo[masks["train"]], y[masks["train"]])
        s_val = clf.decision_function(Xo[masks[SELECTION_SPLIT]])
        per_c[C] = (roc_auc_score(y[masks[SELECTION_SPLIT]], s_val), clf)
    best_C = max(per_c, key=lambda c: per_c[c][0])
    clf = per_c[best_C][1]
    multi_scores = clf.decision_function(Xo)
    multi_row = evaluate("multivariate:output_only", multi_scores,
                         {"C": best_C,
                          "features": MULTIVARIATE,
                          "val_auroc_by_C": {str(c): round(per_c[c][0], 4) for c in C_GRID}})
    print(f"multivariate output-only  C={best_C:<5g} " + "  ".join(
        f"{s}={multi_row.get(f'{s}_auroc', float('nan')):.3f}" for s in T.SPLITS))

    # ---- paired deltas vs the val-selected activation depth ----
    act_by_key = load_activation_scores(src_dir, args.primary_depth)
    act = np.array([act_by_key[(r["split"], r["filename"])] for r in rows])

    deltas = {}
    for label, out_scores in (("multivariate:output_only", multi_scores),
                              (f"scalar:{PRIMARY_SCALAR}", feat[PRIMARY_SCALAR])):
        for s in ("test", "ood_test"):
            m = masks[s]
            point = (roc_auc_score(y[m], act[m]) - roc_auc_score(y[m], out_scores[m]))
            d = paired_cluster_bootstrap_delta(y[m], act[m], out_scores[m], groups[m],
                                               args.n_bootstrap, args.seed)
            d["delta_point"] = float(point)
            deltas[f"{label}|{s}"] = d
            print(f"paired delta  depth{args.primary_depth} - {label:28s} "
                  f"{s:8s} = {point:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] "
                  f"P(delta>0)={d['frac_replicates_positive']:.2f}")

    # ---- outputs ----
    metrics = {
        "run_id": args.run_id,
        "activation_run_id": args.activation_run_id,
        "primary_activation_depth": args.primary_depth,
        "primary_activation_note": (
            "depth selected on val AUROC in R001; the descriptive OOD maximum "
            "(depth 56, 0.964) is deliberately NOT the comparator"),
        "primary_scalar": PRIMARY_SCALAR,
        "C_grid": C_GRID,
        "selection_split": SELECTION_SPLIT,
        "bootstrap": {"n": args.n_bootstrap, "clustered_by": "question_id",
                      "paired": "same resampled clusters for both score vectors"},
        "think_token_id": think_id,
        "scalars": scalar_rows,
        "multivariate": multi_row,
        "paired_deltas": deltas,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id,
        "source_run": args.activation_run_id,
        "source_sample_sha256": src_cfg["sample_sha256"],
        "model": src_cfg["model"], "revision": src_cfg["revision"],
        "features_cached": cached_names,
        "features_used_multivariate": MULTIVARIATE,
        "excluded": {"top1_token_id": "ids have no ordinal meaning; used only via top1_is_think"},
        "primary_scalar": PRIMARY_SCALAR,
        "primary_activation_depth": args.primary_depth,
        "C_grid": C_GRID, "selection_split": SELECTION_SPLIT,
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
    }, indent=2))

    with (run_dir / "baseline_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(score_rows[0]))
        w.writeheader()
        w.writerows(score_rows)

    tbl = T.REPO / "artifacts/tables/output_baseline_auroc.csv"
    tbl.parent.mkdir(parents=True, exist_ok=True)
    cols = (["baseline", "orientation", "C"]
            + [f"{s}_auroc" for s in T.SPLITS]
            + [f"{s}_ci_lo" for s in ("val", "test", "ood_test")]
            + [f"{s}_ci_hi" for s in ("val", "test", "ood_test")])
    with tbl.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in scalar_rows + [multi_row]:
            w.writerow(r)

    print(f"\nwrote {run_dir/'metrics.json'}, {run_dir/'baseline_scores.csv'}, {tbl}")
    print("Apply the R002 interpretation bands in STATE.md before choosing R003.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
