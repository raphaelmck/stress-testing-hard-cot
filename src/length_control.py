#!/usr/bin/env python3
"""R004: does the frozen depth-40 probe know anything prefix length does not?

Primary confirmatory split is `test`. `val` is supplementary only: depth 40 was
selected on val (D005/D010), so val cannot carry a depth-40-vs-length claim.
`ood_test` is closed (D011) and this script refuses to touch it.

Nothing is refit for the primary analysis. Scores are the frozen R001 depth-40
probe outputs, the released `token_length`, and `think_logprob` from the cached
D007 features as secondary context.

Structure:
  A  within-question macro concordance for all three scores, per split
  B  PRIMARY: the length-discordant pairs -- pairs where token_length orders the
     YES/NO pair the wrong way. Does the activation score still get them right?
  C  near-length-matched pairs at |dlen| <= 100 and <= 250 tokens. Both thresholds
     were frozen before this script was run; no threshold search.
  D  secondary: probe score residualised on its fitted marginal association with
     length. The nuisance fit uses `val` only (no test labels, and no test rows).
     This removes a fitted marginal association with length from a scalar score.
     It does NOT "remove length information" from the representation.

Usage:
    python src/length_control.py --run-id r004_length_control
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import task1_data as T
from fit_task1_probe import N_BOOTSTRAP
from fit_output_baselines import load_output_features

PRIMARY_SPLIT = "test"
SUPPLEMENTARY_SPLIT = "val"
CLOSED_SPLITS = {"ood_test"}                  # D011
MATCH_THRESHOLDS = [100.0, 250.0]             # frozen before the run
SCORES = ["activation_depth40", "think_logprob", "token_length"]


def build_pairs(rows_by_q, score_names):
    """All within-question YES x NO pairs, with each score's ordering outcome."""
    pairs = []
    for q, d in rows_by_q.items():
        for pos in d["yes"]:
            for neg in d["no"]:
                p = {"question_id": q,
                     "dlen": pos["token_length"] - neg["token_length"],
                     "abs_dlen": abs(pos["token_length"] - neg["token_length"])}
                for s in score_names:
                    diff = pos[s] - neg[s]
                    p[s] = 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
                pairs.append(p)
    return pairs


def macro_by_question(pairs, key):
    """Mean over questions of the mean pair outcome within each question."""
    per_q = {}
    for p in pairs:
        per_q.setdefault(p["question_id"], []).append(p[key])
    return ({q: float(np.mean(v)) for q, v in per_q.items()},
            float(np.mean([np.mean(v) for v in per_q.values()])) if per_q else float("nan"))


def question_bootstrap(per_q_by_score, n_boot, seed):
    """Question-level bootstrap CIs, paired across scores on identical draws."""
    qids = sorted(set().union(*[set(d) for d in per_q_by_score.values()]))
    if len(qids) < 2:
        return {k: {"ci_lo": None, "ci_hi": None} for k in per_q_by_score}, {}
    rng = np.random.default_rng(seed)
    draws = [rng.choice(qids, size=len(qids), replace=True) for _ in range(n_boot)]
    cis = {}
    for name, per_q in per_q_by_score.items():
        vals = np.array([np.mean([per_q[q] for q in d if q in per_q]) for d in draws])
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        cis[name] = {"ci_lo": float(lo), "ci_hi": float(hi), "n_replicates": int(vals.size)}
    deltas = {}
    a = per_q_by_score.get("activation_depth40")
    for name, per_q in per_q_by_score.items():
        if name == "activation_depth40" or a is None:
            continue
        d = np.array([np.mean([a[q] for q in dr if q in a])
                      - np.mean([per_q[q] for q in dr if q in per_q]) for dr in draws])
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        deltas[f"activation_depth40 - {name}"] = {
            "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_replicates_positive": float((d > 0).mean()),
            "n_replicates": int(d.size)}
    return cis, deltas


def subset_report(pairs, label, n_boot, seed, min_pairs=10):
    """Concordance of each score on a subset of pairs, pooled and macro."""
    if not pairs:
        return {"label": label, "n_pairs": 0, "n_questions": 0,
                "underpowered": True, "note": "no pairs in this subset"}
    per_q_by_score, out = {}, {"label": label, "n_pairs": len(pairs),
                               "n_questions": len(({p["question_id"] for p in pairs}))}
    for s in ("activation_depth40", "think_logprob"):
        per_q, macro = macro_by_question(pairs, s)
        per_q_by_score[s] = per_q
        out[f"{s}_pooled"] = float(np.mean([p[s] for p in pairs]))
        out[f"{s}_macro"] = macro
        out[f"{s}_questions_above_half"] = int(sum(1 for v in per_q.values() if v > 0.5))
    cis, deltas = question_bootstrap(per_q_by_score, n_boot, seed)
    out["ci"] = cis
    out["paired_delta"] = deltas
    out["underpowered"] = len(pairs) < min_pairs or out["n_questions"] < 5
    if out["underpowered"]:
        out["note"] = ("too few pairs/questions to distinguish 'probe fails here' "
                       "from 'no data here'; absence of evidence is NOT evidence "
                       "for the length explanation")
    return out


def collect(split, act, think, length, rows):
    by_q = {}
    for r in rows:
        if r["split"] != split:
            continue
        key = (split, r["filename"])
        d = by_q.setdefault(r["question_id"], {"yes": [], "no": []})
        d[r["label"]].append({"filename": r["filename"],
                              "activation_depth40": act[key],
                              "think_logprob": think[key],
                              "token_length": length[key]})
    return {q: d for q, d in by_q.items() if d["yes"] and d["no"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="r004_length_control")
    ap.add_argument("--activation-run-id", default="r001_qwen32b")
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir = T.REPO / "artifacts/runs" / args.activation_run_id
    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((src_dir / "config.json").read_text())

    F, rows = load_output_features(src_dir, src_cfg["output_features"])
    col = {n: i for i, n in enumerate(src_cfg["output_features"])}
    think = {(r["split"], r["filename"]): float(F[i, col["think_logprob"]])
             for i, r in enumerate(rows)}
    with (src_dir / "probe_scores.csv").open() as fh:
        act = {(r["split"], r["filename"]): float(r["score"])
               for r in csv.DictReader(fh) if int(r["depth"]) == args.depth}

    length = {}
    for split in (PRIMARY_SPLIT, SUPPLEMENTARY_SPLIT):
        if split in CLOSED_SPLITS:
            raise SystemExit(f"{split} is closed by D011")
        for path in sorted((T.DATA_ROOT / "qwen-3-32b" / split).glob("*.json")):
            length[(split, path.name)] = float(json.loads(path.read_text())["token_length"])

    report = {"run_id": args.run_id, "depth": args.depth,
              "primary_split": PRIMARY_SPLIT,
              "supplementary_split": SUPPLEMENTARY_SPLIT,
              "closed_splits": sorted(CLOSED_SPLITS),
              "match_thresholds": MATCH_THRESHOLDS,
              "bootstrap": {"n": args.n_bootstrap, "unit": "question"},
              "splits": {}}
    pair_rows = []

    for split in (PRIMARY_SPLIT, SUPPLEMENTARY_SPLIT):
        by_q = collect(split, act, think, length, rows)
        pairs = build_pairs(by_q, SCORES)
        for p in pairs:
            pair_rows.append({"split": split, **p})
        print(f"\n===== {split}"
              f"{'  (PRIMARY)' if split == PRIMARY_SPLIT else '  (supplementary)'} "
              f"=====")
        print(f"{len(by_q)} questions with both labels, {len(pairs)} YES/NO pairs")

        # --- A. aggregate within-question concordance ---
        per_q_by_score, macro = {}, {}
        for s in SCORES:
            per_q_by_score[s], macro[s] = macro_by_question(pairs, s)
        cis, deltas = question_bootstrap(per_q_by_score, args.n_bootstrap, args.seed)
        for s in SCORES:
            print(f"  A macro concordance {s:20s} = {macro[s]:.3f} "
                  f"[{cis[s]['ci_lo']:.3f}, {cis[s]['ci_hi']:.3f}]")
        for k, v in deltas.items():
            print(f"  A paired delta {k:45s} "
                  f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] "
                  f"P(>0)={v['frac_replicates_positive']:.3f}")

        # --- B. PRIMARY: length-discordant pairs ---
        disc = [p for p in pairs if p["dlen"] < 0]
        tied = [p for p in pairs if p["dlen"] == 0]
        conc = [p for p in pairs if p["dlen"] > 0]
        b = subset_report(disc, "length-discordant (YES shorter)",
                          args.n_bootstrap, args.seed)
        print(f"  B length-concordant={len(conc)}  tied={len(tied)}  "
              f"discordant={len(disc)} pairs over {b['n_questions']} questions")
        if b["n_pairs"]:
            print(f"  B activation on discordant pairs: pooled="
                  f"{b['activation_depth40_pooled']:.3f} macro="
                  f"{b['activation_depth40_macro']:.3f}"
                  + (f" [{b['ci']['activation_depth40']['ci_lo']:.3f}, "
                     f"{b['ci']['activation_depth40']['ci_hi']:.3f}]"
                     if b["ci"].get("activation_depth40", {}).get("ci_lo") is not None
                     else "")
                  + f"  questions>0.5: {b['activation_depth40_questions_above_half']}"
                    f"/{b['n_questions']}")
            print(f"  B think_logprob on the same pairs: pooled="
                  f"{b['think_logprob_pooled']:.3f} macro={b['think_logprob_macro']:.3f}")
        if b["underpowered"]:
            print("  B *** UNDERPOWERED -- do not read a length explanation into this")

        # --- C. near-length-matched sensitivity ---
        cs = []
        for thr in MATCH_THRESHOLDS:
            sub = [p for p in pairs if p["abs_dlen"] <= thr]
            r = subset_report(sub, f"|dlen| <= {thr:.0f}", args.n_bootstrap, args.seed)
            cs.append(r)
            if r["n_pairs"]:
                print(f"  C |dlen|<={thr:.0f}: {r['n_pairs']} pairs / "
                      f"{r['n_questions']} questions  activation macro="
                      f"{r['activation_depth40_macro']:.3f}  think macro="
                      f"{r['think_logprob_macro']:.3f}"
                      + ("  *** UNDERPOWERED" if r["underpowered"] else ""))
            else:
                print(f"  C |dlen|<={thr:.0f}: no pairs")

        report["splits"][split] = {
            "n_questions_both_labels": len(by_q),
            "n_pairs": len(pairs),
            "A_macro_concordance": macro,
            "A_ci": cis,
            "A_paired_deltas": deltas,
            "B_length_pair_counts": {"concordant": len(conc), "tied": len(tied),
                                     "discordant": len(disc)},
            "B_discordant": b,
            "C_near_matched": cs,
        }

    # --- D. secondary: residualise the scalar score on length, fit on val only ---
    val_keys = [(SUPPLEMENTARY_SPLIT, f) for (s, f) in act if s == SUPPLEMENTARY_SPLIT]
    xv = np.array([length[k] for k in val_keys])
    yv = np.array([act[k] for k in val_keys])
    slope, intercept = np.polyfit(xv, yv, 1)
    resid = {k: act[k] - (slope * length[k] + intercept)
             for k in act if k[0] in (PRIMARY_SPLIT, SUPPLEMENTARY_SPLIT)}
    by_q = collect(PRIMARY_SPLIT, resid, think, length, rows)
    rpairs = build_pairs(by_q, ["activation_depth40", "think_logprob", "token_length"])
    per_q_res, macro_res = macro_by_question(rpairs, "activation_depth40")
    cis_res, _ = question_bootstrap({"residualised": per_q_res}, args.n_bootstrap, args.seed)
    disc_res = subset_report([p for p in rpairs if p["dlen"] < 0],
                             "residualised, length-discordant",
                             args.n_bootstrap, args.seed)
    print(f"\n===== D (secondary) =====")
    print(f"  nuisance fit on {SUPPLEMENTARY_SPLIT} only: "
          f"score ~ {slope:.5f} * token_length + {intercept:.3f}")
    print(f"  residualised probe, {PRIMARY_SPLIT} macro concordance = {macro_res:.3f} "
          f"[{cis_res['residualised']['ci_lo']:.3f}, {cis_res['residualised']['ci_hi']:.3f}]")
    if disc_res["n_pairs"]:
        print(f"  residualised on discordant pairs: macro="
              f"{disc_res['activation_depth40_macro']:.3f} "
              f"({disc_res['n_pairs']} pairs)")
    report["D_residualised"] = {
        "nuisance_fit_split": SUPPLEMENTARY_SPLIT,
        "slope": float(slope), "intercept": float(intercept),
        "interpretation_limit": ("removes the fitted MARGINAL association between "
                                 "the scalar probe score and token_length; it does "
                                 "NOT remove length information from the "
                                 "representation"),
        "primary_split_macro_concordance": macro_res,
        "ci": cis_res["residualised"],
        "discordant": disc_res,
    }

    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "source_run": args.activation_run_id,
        "depth": args.depth,
        "depth_note": "val-selected depth (D010)",
        "primary_split": PRIMARY_SPLIT,
        "supplementary_split": SUPPLEMENTARY_SPLIT,
        "closed_splits": sorted(CLOSED_SPLITS),
        "match_thresholds": MATCH_THRESHOLDS,
        "thresholds_frozen": "before the run; no threshold search",
        "refit": "none for A-C; D fits one 2-parameter line on val only",
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "source_sample_sha256": src_cfg["sample_sha256"],
    }, indent=2))
    with (run_dir / "pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0]))
        w.writeheader()
        w.writerows(pair_rows)
    print(f"\nwrote {run_dir/'metrics.json'}, {run_dir/'pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
