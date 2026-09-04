#!/usr/bin/env python3
"""Regenerate the three report figures from committed artifacts.

Read-only with respect to run directories: this reads the frozen tables and
metrics and writes `artifacts/figures/report_fig{1,2,3}.png`. It refits nothing
and overwrites no run artifact, so the originals produced by each experiment stay
exactly as committed.

Presentation conventions shared by all three figures: sentence-case titles with no
run identifiers, reader-facing split names (Validation / Test / OOD) rather than
`val` / `ood_test`, and prose rather than variable names in axes and legends.

Usage:
    python src/make_report_figures.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import task1_data as T

FIGS = T.REPO / "artifacts/figures"
BLUE, ORANGE, GREY = "#2E6DB4", "#E08A1E", "#8C8C8C"


def fig1():
    """Termination-probe performance across transformer depth."""
    with (T.REPO / "artifacts/tables/reproduction_layer_auroc.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    d = [int(r["depth"]) for r in rows]
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0.5, color="0.55", lw=1.0, ls=":", zorder=1)
    ax.plot(x, [float(r["train_auroc"]) for r in rows], "--o", color="0.6",
            ms=6, lw=1.6, label="Train", zorder=2)
    for key, colour, label in (("val", BLUE, "Validation"),
                               ("test", ORANGE, "Test"),
                               ("ood_test", "#B33A3A", "OOD")):
        v = np.array([float(r[f"{key}_auroc"]) for r in rows])
        lo = np.array([float(r[f"{key}_ci_lo"]) for r in rows])
        hi = np.array([float(r[f"{key}_ci_hi"]) for r in rows])
        ax.errorbar(x, v, yerr=[v - lo, hi - v], marker="o", ms=6, lw=2,
                    capsize=4, color=colour, label=label, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(d)
    ax.set_xlabel("Transformer depth")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Termination-probe performance across transformer depth\n"
                 "95% CIs clustered by question", fontsize=11)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(axis="y", color="0.92", lw=0.7); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "report_fig1.png", dpi=200)
    print("wrote report_fig1.png")


def fig2():
    """Global length balance hides an OOD within-question association."""
    m = json.loads((T.REPO / "artifacts/runs/r005_length_balance_audit"
                    / "metrics.json").read_text())["splits"]
    splits = ["val", "test", "ood_test"]
    names = ["Validation", "Test", "OOD"]
    x = np.arange(3); w = 0.26

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(0.5, color="0.55", lw=1.0, ls=":", zorder=1)
    series = [
        ([m[s]["pooled_length_auroc"] for s in splits], GREY,
         "Prefix length, pooled across questions"),
        ([m[s]["macro_within_question_length_concordance"] for s in splits], ORANGE,
         "Prefix length, within question"),
        ([m[s]["activation_within_question_concordance"] for s in splits], BLUE,
         "Activation probe, within question"),
    ]
    for i, (vals, colour, label) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colour, label=label)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                    f"{r.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8.5, color="0.25")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Predictive performance")
    # Headroom so the legend never sits over a value label.
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Global length balance hides an OOD within-question association",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left", ncol=1, framealpha=0.0,
              bbox_to_anchor=(0.0, 1.005))
    ax.grid(axis="y", color="0.92", lw=0.7); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "report_fig2.png", dpi=200)
    print("wrote report_fig2.png")


def fig3():
    """Probe-direction steering does not specifically control termination."""
    m = json.loads((T.REPO / "artifacts/runs/r007_steer_test"
                    / "metrics.json").read_text())["primary"]
    xs = [-2, -1, 0, 1, 2]
    b = [m[f"beta{k:+d}"] for k in xs]
    v = np.array([r[0] for r in b])
    lo = np.array([r[1] for r in b]); hi = np.array([r[2] for r in b])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], marker="o", ms=7, lw=2, capsize=4,
                color=BLUE, label="Probe direction", zorder=3)
    ov = np.array([m["ortho-2"][0], m["ortho+2"][0]])
    olo = np.array([m["ortho-2"][1], m["ortho+2"][1]])
    ohi = np.array([m["ortho-2"][2], m["ortho+2"][2]])
    ax.errorbar([-2, 2], ov, yerr=[ov - olo, ohi - ov], marker="s", ms=7, lw=0,
                elinewidth=2, capsize=4, color=ORANGE,
                label="Matched-norm orthogonal direction", zorder=4)
    ax.axhline(v[2], color="0.6", ls=":", lw=1, zorder=1)
    # Axis starts at zero: the whole point is that these are ~1-point changes
    # around a ~45% baseline, and a cropped axis would inflate them.
    ax.set_ylim(0, 0.58)
    ax.set_xticks(xs)
    ax.set_xlabel("Probe-score intervention (validation SDs)")
    ax.set_ylabel("Termination probability within 60 tokens")
    ax.set_title("Probe-direction steering does not specifically control termination",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(axis="y", color="0.92", lw=0.7); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "report_fig3.png", dpi=200)
    print("wrote report_fig3.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
