#!/usr/bin/env python3
"""R001: fit L2 logistic termination probes on cached Task 1 activations.

Frozen decisions this script implements (see DECISIONS.md):
  D005  train on `train` only; select C on `val` AUROC ONLY; `test` is the ID
        check; `ood_test` is the final cross-domain check. The five depths and
        the C grid were preregistered -- neither is chosen from results.
  D006  every confidence interval is a bootstrap over question_id CLUSTERS,
        never over individual rows. ood_test is 58 rows from 32 questions, and
        16 of those carry both labels, so rows are not independent.
  D007  the cached output features (</think> logit, margin, entropy, ...) are
        NOT loaded or touched here. Testing H0 is a separate, later experiment;
        this script must not be able to peek.

Usage:
    python src/fit_task1_probe.py --run-id r001_extract
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

C_GRID = [0.01, 0.1, 1.0, 10.0]        # preregistered, D005
N_BOOTSTRAP = 2000
SELECTION_SPLIT = "val"                 # D005: nothing is selected on ood_test


def load_run(run_dir: pathlib.Path):
    """Load activations + row metadata from every shard.

    Deliberately reads only the `activations` array from each npz. The
    `output_features` array is present in the same file and is left untouched
    under the D007 embargo.
    """
    act_dir = run_dir / "activations"
    shards = sorted(act_dir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards found in {act_dir}")

    acts, rows = [], []
    for npz_path in shards:
        with np.load(npz_path) as z:
            if "activations" not in z:
                raise SystemExit(f"{npz_path.name} has no activations array")
            acts.append(z["activations"])          # [n, depths, hidden]
        with npz_path.with_suffix(".csv").open() as fh:
            rows.extend(list(csv.DictReader(fh)))

    X = np.concatenate(acts).astype(np.float32)     # fp16 cache -> fp32 for fitting
    if len(rows) != X.shape[0]:
        raise SystemExit(f"metadata/activation length mismatch: {len(rows)} vs {X.shape[0]}")
    return X, rows


def cluster_bootstrap_auroc(y, scores, groups, n_boot=N_BOOTSTRAP, seed=0):
    """95% CI for AUROC, resampling question_id clusters with replacement (D006).

    Row-level resampling would treat several prefixes from one question as
    independent evidence and understate the interval. Draws that end up
    single-class are skipped, not counted.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, scores[idx]))
    if not vals:
        return (float("nan"), float("nan"), 0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi), len(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = T.REPO / "artifacts/runs" / args.run_id
    cfg = json.loads((run_dir / "config.json").read_text())
    depths = cfg["depths"]
    if not cfg.get("preregistered_depths_used", True):
        print("WARNING: this run used overridden depths and is NOT a valid R001 result")

    X, rows = load_run(run_dir)
    split = np.array([r["split"] for r in rows])
    y = np.array([1 if r["label"] == "yes" else 0 for r in rows])
    groups = np.array([r["question_id"] for r in rows])
    print(f"loaded {X.shape[0]} rows, activations {X.shape}, depths={depths}")
    for s in T.SPLITS:
        m = split == s
        if m.any():
            print(f"  {s:9s} n={m.sum():5d}  yes={y[m].sum():5d}  "
                  f"questions={len(np.unique(groups[m]))}")

    masks = {s: split == s for s in T.SPLITS}
    if not masks["train"].any():
        raise SystemExit("no train rows -- cannot fit probes")

    results, score_rows = [], []
    for di, depth in enumerate(depths):
        Xd = X[:, di, :]

        # --- select C on val AUROC only (D005) ---
        per_c = {}
        for C in C_GRID:
            clf = make_pipeline(
                StandardScaler(),
                # l1_ratio=0 is pure L2; sklearn 1.9 deprecated penalty="l2"
                LogisticRegression(l1_ratio=0.0, C=C, max_iter=5000))
            clf.fit(Xd[masks["train"]], y[masks["train"]])
            s_val = clf.decision_function(Xd[masks[SELECTION_SPLIT]])
            per_c[C] = (roc_auc_score(y[masks[SELECTION_SPLIT]], s_val), clf)
        best_C = max(per_c, key=lambda c: per_c[c][0])
        clf = per_c[best_C][1]

        row = {"depth": depth, "block_index": depth - 1, "C": best_C,
               "val_auroc_by_C": {str(c): round(per_c[c][0], 4) for c in C_GRID}}

        for s in T.SPLITS:
            m = masks[s]
            if not m.any():
                continue
            sc = clf.decision_function(Xd[m])
            auc = roc_auc_score(y[m], sc)
            row[f"{s}_auroc"] = float(auc)
            if s != "train":     # CIs where the cluster structure matters
                lo, hi, n_ok = cluster_bootstrap_auroc(
                    y[m], sc, groups[m], args.n_bootstrap, args.seed)
                row[f"{s}_ci_lo"], row[f"{s}_ci_hi"] = lo, hi
            for pos, k in enumerate(np.flatnonzero(m)):
                score_rows.append({
                    "depth": depth, "split": s,
                    "filename": rows[k]["filename"],
                    "question_id": rows[k]["question_id"],
                    "rollout_idx": rows[k]["rollout_idx"],
                    "prefix_idx": rows[k]["prefix_idx"],
                    "label": rows[k]["label"], "y": int(y[k]),
                    "score": float(sc[pos]),
                })
        results.append(row)
        print(f"depth {depth:2d}  C={best_C:<5g} " + "  ".join(
            f"{s}={row.get(f'{s}_auroc', float('nan')):.3f}" for s in T.SPLITS))

    # ---- outputs ----
    ood = [r["ood_test_auroc"] for r in results if "ood_test_auroc" in r]
    best = max(results, key=lambda r: r.get("ood_test_auroc", -1)) if ood else None
    metrics = {
        "run_id": args.run_id,
        "depths": depths,
        "C_grid": C_GRID,
        "selection_split": SELECTION_SPLIT,
        "bootstrap": {"n": args.n_bootstrap, "clustered_by": "question_id"},
        "per_depth": results,
        "max_ood_auroc": max(ood) if ood else None,
        "max_ood_auroc_depth": best["depth"] if best else None,
        "note": "D007 output features were not loaded by this script",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (run_dir / "probe_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(score_rows[0]))
        w.writeheader()
        w.writerows(score_rows)

    tbl = T.REPO / "artifacts/tables/reproduction_layer_auroc.csv"
    cols = ["depth", "block_index", "C"] + [
        f"{s}_auroc" for s in T.SPLITS] + [
        f"{s}_ci_{b}" for s in ("val", "test", "ood_test") for b in ("lo", "hi")]
    with tbl.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    make_figure(results, depths)
    print(f"\nwrote {run_dir/'metrics.json'}, {run_dir/'probe_scores.csv'}, {tbl}")
    if ood:
        print(f"max OOD AUROC = {max(ood):.3f} at depth {best['depth']} "
              f"[{best.get('ood_test_ci_lo', float('nan')):.3f}, "
              f"{best.get('ood_test_ci_hi', float('nan')):.3f}]")
        print("Apply the STATE.md decision rule before choosing the next experiment.")
    return 0


def make_figure(results, depths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, colour in [("train", "#999999"), ("val", "#4c72b0"),
                      ("test", "#dd8452"), ("ood_test", "#c44e52")]:
        ys = [r.get(f"{s}_auroc") for r in results]
        if not any(v is not None for v in ys):
            continue
        if f"{s}_ci_lo" in results[0]:
            err = np.array([[r[f"{s}_auroc"] - r[f"{s}_ci_lo"] for r in results],
                            [r[f"{s}_ci_hi"] - r[f"{s}_auroc"] for r in results]])
            ax.errorbar(depths, ys, yerr=err, marker="o", capsize=3,
                        label=s, color=colour)
        else:
            ax.plot(depths, ys, marker="o", label=s, color=colour, linestyle="--")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("transformer depth (preregistered)")
    ax.set_ylabel("AUROC")
    ax.set_title("R001: termination probe by depth\n"
                 "CIs bootstrap-clustered by question_id", fontsize=10)
    ax.set_xticks(depths)
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = T.REPO / "artifacts/figures/reproduction_layer_auroc.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
