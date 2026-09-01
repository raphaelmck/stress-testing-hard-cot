#!/usr/bin/env python3
"""Programmatic audit of Task 1 of cot-proxy-tasks (reasoning termination).

Reads every rollout record under
    cot-proxy-tasks/datasets/1/qwen-3-32b/{train,val,test,ood_test}/
and writes

    artifacts/tables/task1_manifest.csv   one row per sample
    artifacts/tables/task1_summary.md     per-split statistics

The point of this script is to verify -- from the data itself, not from the
paper -- that the training and evaluation splits are labelled differently:

  * train is expected to use a WORD-DISTANCE PROXY
    (yes = 25/35/45/55 words from the end of an existing CoT, no = 300+),
  * val/test/ood_test are expected to use BEHAVIOURAL labels obtained by
    resampling 50 continuations per prefix.

The manifest deliberately stores no `cot_prefix` text: 46k prefixes would make
the table too large to commit, and every downstream check here needs length,
not content. Word/char counts are derived for all splits so that prefix length
is comparable across splits (the released `prefix_words` field exists only in
train).

Usage:
    python src/inspect_task1.py
    python src/inspect_task1.py --data-root <path> --out-dir <path>
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import sys

SPLITS = ["train", "val", "test", "ood_test"]

# Fields copied verbatim from the released records when present.
RELEASED_FIELDS = [
    "question_id",
    "rollout_idx",
    "prefix_idx",
    "label",
    "prefix_words",
    "total_words",
    "token_length",
    "distance_from_end",
    "yes_count",
    "no_count",
    "total_resamples",
    "mean_yes_position",
]

# Column order of the manifest.
COLUMNS = ["split", "filename"] + RELEASED_FIELDS + [
    "cot_prefix_n_words",   # derived: len(cot_prefix.split())
    "cot_prefix_n_chars",   # derived
]


def percentile(values, q):
    """Linear-interpolation percentile; q in [0, 100]. None for empty input."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def read_split(split_dir: pathlib.Path, split: str):
    """Yield one manifest row per JSON record in a split directory."""
    for path in sorted(split_dir.glob("*.json")):
        with path.open() as fh:
            rec = json.load(fh)

        row = {"split": split, "filename": path.name}
        for field in RELEASED_FIELDS:
            row[field] = rec.get(field)

        prefix = rec.get("cot_prefix")
        row["cot_prefix_n_words"] = len(prefix.split()) if prefix is not None else None
        row["cot_prefix_n_chars"] = len(prefix) if prefix is not None else None

        unknown = set(rec) - set(RELEASED_FIELDS) - {"cot_prefix"}
        if unknown:
            print(f"  note: {path.name} has unrecorded fields {sorted(unknown)}",
                  file=sys.stderr)
        yield row


def summarize_split(split: str, rows: list[dict]) -> str:
    """Per-split statistics block for the summary report."""
    out = [f"## {split}", ""]
    n = len(rows)
    labels = collections.Counter(r["label"] for r in rows)
    qids = collections.Counter(r["question_id"] for r in rows)
    per_q = sorted(qids.values())

    out += [
        f"- N = {n}",
        "- labels: " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())),
        f"- unique question_id = {len(qids)}",
        "- examples per question: "
        f"min={per_q[0] if per_q else '-'}, median={fmt(percentile(per_q, 50))}, "
        f"max={per_q[-1] if per_q else '-'}",
        "",
    ]

    # Which label-construction fields actually exist in this split.
    present = [f for f in RELEASED_FIELDS
               if any(r[f] is not None for r in rows)]
    out += ["- fields present: `" + "`, `".join(present) + "`", ""]

    # Prefix length by label -- derived, so comparable across every split.
    out += ["### Prefix length (derived word count), by label", "",
            "| label | n | q10 | median | q90 |", "|---|---|---|---|---|"]
    for label in sorted(labels):
        vals = [r["cot_prefix_n_words"] for r in rows
                if r["label"] == label and r["cot_prefix_n_words"] is not None]
        out.append(f"| {label} | {len(vals)} | {fmt(percentile(vals, 10))} | "
                   f"{fmt(percentile(vals, 50))} | {fmt(percentile(vals, 90))} |")
    out.append("")

    # Released token_length, where present.
    tls = [r["token_length"] for r in rows if r["token_length"] is not None]
    if tls:
        out += ["### token_length (released), by label", "",
                "| label | n | q10 | median | q90 |", "|---|---|---|---|---|"]
        for label in sorted(labels):
            vals = [r["token_length"] for r in rows
                    if r["label"] == label and r["token_length"] is not None]
            out.append(f"| {label} | {len(vals)} | {fmt(percentile(vals, 10))} | "
                       f"{fmt(percentile(vals, 50))} | {fmt(percentile(vals, 90))} |")
        out.append("")

    # Distance-proxy labelling.
    dists = [r for r in rows if r["distance_from_end"] is not None]
    if dists:
        out += ["### distance_from_end x label (proxy labelling)", "",
                "| distance_from_end | " +
                " | ".join(sorted(labels)) + " |",
                "|---|" + "---|" * len(labels)]
        table = collections.Counter(
            (r["distance_from_end"], r["label"]) for r in dists)
        for d in sorted({r["distance_from_end"] for r in dists}):
            cells = " | ".join(str(table[(d, lab)]) for lab in sorted(labels))
            out.append(f"| {d} | {cells} |")
        out += ["", f"- records with distance_from_end: {len(dists)} / {n}", ""]

    # Behavioural labelling.
    yc = [r for r in rows if r["yes_count"] is not None]
    if yc:
        totals = collections.Counter(r["total_resamples"] for r in yc)
        out += ["### yes_count / no_count (behavioural labelling)", "",
                "- total_resamples: " +
                ", ".join(f"{k}x{v}" for k, v in sorted(totals.items())),
                "",
                "| label | n | yes_count q10 | median | q90 | min | max |",
                "|---|---|---|---|---|---|---|"]
        for label in sorted(labels):
            vals = [r["yes_count"] for r in yc if r["label"] == label]
            if not vals:
                continue
            out.append(f"| {label} | {len(vals)} | {fmt(percentile(vals, 10))} | "
                       f"{fmt(percentile(vals, 50))} | {fmt(percentile(vals, 90))} | "
                       f"{min(vals)} | {max(vals)} |")
        out.append("")

        # Does every record clear the stated >=45/50 purity bar?
        bad = [r for r in yc
               if r["total_resamples"]
               and max(r["yes_count"], r["no_count"]) < 0.9 * r["total_resamples"]]
        out += [f"- records failing a >=90% resample-agreement bar: {len(bad)} / {len(yc)}",
                "- consistency of label vs majority resample vote: "
                f"{sum(1 for r in yc if (r['label'] == 'yes') == (r['yes_count'] > r['no_count']))}"
                f" / {len(yc)} agree",
                ""]

    return "\n".join(out)


def cross_split_report(by_split: dict[str, list[dict]]) -> str:
    """question_id overlap between splits -- a leakage check."""
    qids = {s: {r["question_id"] for r in rows} for s, rows in by_split.items()}
    out = ["## Cross-split question_id overlap", "",
           "| split | unique qids | " +
           " | ".join(f"shared w/ {s}" for s in SPLITS) + " |",
           "|---|---|" + "---|" * len(SPLITS)]
    for s in SPLITS:
        if s not in qids:
            continue
        cells = " | ".join(
            str(len(qids[s] & qids[o])) if o in qids else "-" for o in SPLITS)
        out.append(f"| {s} | {len(qids[s])} | {cells} |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=pathlib.Path,
                    default=repo / "cot-proxy-tasks/datasets/1/qwen-3-32b")
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=repo / "artifacts/tables")
    args = ap.parse_args()

    if not args.data_root.is_dir():
        print(f"data root not found: {args.data_root}", file=sys.stderr)
        print("see README.md for how to clone the upstream dataset", file=sys.stderr)
        return 1

    by_split: dict[str, list[dict]] = {}
    for split in SPLITS:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            print(f"missing split directory: {split_dir}", file=sys.stderr)
            continue
        rows = list(read_split(split_dir, split))
        by_split[split] = rows
        print(f"{split}: {len(rows)} records", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.out_dir / "task1_manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for split in SPLITS:
            for row in by_split.get(split, []):
                writer.writerow(row)

    report = [
        "# Task 1 programmatic audit",
        "",
        f"Generated by `src/inspect_task1.py` from `{args.data_root}`.",
        "Per-sample rows: `artifacts/tables/task1_manifest.csv`.",
        "",
        "`cot_prefix_n_words` / `cot_prefix_n_chars` are derived here for every split;",
        "`prefix_words`, `distance_from_end`, `yes_count`, `no_count` and",
        "`total_resamples` are copied verbatim from the released records.",
        "",
    ]
    for split in SPLITS:
        if split in by_split:
            report.append(summarize_split(split, by_split[split]))
    report.append(cross_split_report(by_split))

    summary = args.out_dir / "task1_summary.md"
    summary.write_text("\n".join(report))

    print(f"wrote {manifest}", file=sys.stderr)
    print(f"wrote {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
