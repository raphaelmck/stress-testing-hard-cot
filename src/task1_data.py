"""Task 1 data loading, family assignment, and the frozen R001 training sample.

No model, no torch. Everything here must be runnable on a laptop so that the
sample manifest can be frozen and inspected before any GPU time is spent.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "cot-proxy-tasks/datasets/1"
UPSTREAM_CHAT_TEMPLATE = REPO / "cot-proxy-tasks/src/utils/chat_template.py"

SPLITS = ["train", "val", "test", "ood_test"]

# Preregistered depths (DECISIONS D005). Depth d == output of zero-indexed
# block d-1 == hidden_states[d] in HF's convention.
DEPTHS = [8, 24, 40, 56, 64]
BLOCK_INDICES = [d - 1 for d in DEPTHS]

# R001 training sample (DECISIONS D005 / the R001 ticket).
SAMPLE_SEED = 42
PER_FAMILY_PER_LABEL = 500

MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"  # pinned in D004


def load_build_thinking_prompt():
    """Load the upstream helper by path.

    Imported by file location rather than as `src.utils.chat_template` because
    the upstream package is also called `src` and would collide with this one.
    Using the released helper is required: the prompt format must match the
    format the released labels were generated under.
    """
    spec = importlib.util.spec_from_file_location(
        "upstream_chat_template", UPSTREAM_CHAT_TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream helper: {UPSTREAM_CHAT_TEMPLATE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_thinking_prompt


def family(question_id: str) -> str:
    """Assign a train record to one of the four training families."""
    if question_id.startswith("bb_"):
        return "big_bench"
    if question_id.startswith("gpqa"):
        return "gpqa_chem"
    if question_id.startswith("race"):
        return "race"
    if question_id.startswith("daily") or question_id.startswith("dd"):
        return "daily_dilemmas"
    raise ValueError(f"unrecognised question family: {question_id}")


def load_records(split: str) -> list[dict]:
    """Load every rollout record for a split, joined with its prompt text.

    Sorted by filename so that ordering is deterministic across machines.
    """
    rollout_dir = DATA_ROOT / "qwen-3-32b" / split
    prompt_dir = DATA_ROOT / "prompts" / split
    out = []
    prompt_cache: dict[str, str] = {}
    for path in sorted(rollout_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        qid = rec["question_id"]
        if qid not in prompt_cache:
            prompt_cache[qid] = json.loads(
                (prompt_dir / f"{qid}.json").read_text())["prompt_text"]
        out.append({
            "split": split,
            "filename": path.name,
            "question_id": qid,
            "rollout_idx": rec["rollout_idx"],
            "prefix_idx": rec["prefix_idx"],
            "label": rec["label"],
            "cot_prefix": rec["cot_prefix"],
            "prompt_text": prompt_cache[qid],
        })
    return out


def sample_train_rows(records: list[dict],
                      seed: int = SAMPLE_SEED,
                      per_family_per_label: int = PER_FAMILY_PER_LABEL) -> list[dict]:
    """Deterministic balanced sample: per_family_per_label of each label, per family.

    Sampling is over a filename-sorted candidate list with a seeded RNG, so the
    same rows come back on any machine. Note this samples ROWS, not questions --
    the train split is heavily question-clustered (median 53 rows/question,
    max 452), so the resulting question concentration is reported in the frozen
    manifest rather than corrected for. That follows the R001 ticket exactly.
    """
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        buckets.setdefault((family(r["question_id"]), r["label"]), []).append(r)

    chosen: list[dict] = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda r: r["filename"])
        if len(pool) < per_family_per_label:
            raise ValueError(
                f"family/label {key} has only {len(pool)} rows, "
                f"need {per_family_per_label}")
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        picked = rng.sample(pool, per_family_per_label)
        for r in picked:
            r = dict(r)
            r["family"] = key[0]
            chosen.append(r)
    chosen.sort(key=lambda r: r["filename"])
    return chosen


def sample_hash(rows: list[dict]) -> str:
    """Stable hash of the frozen sample: identity + label of every row."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r["split"], r["filename"])):
        h.update(f"{r['split']}/{r['filename']}/{r['label']}\n".encode())
    return h.hexdigest()


MANIFEST_COLUMNS = [
    "split", "filename", "question_id", "rollout_idx", "prefix_idx",
    "label", "family", "n_words", "n_tokens",
]


def write_sample_manifest(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in MANIFEST_COLUMNS})


def read_sample_manifest(path: pathlib.Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))
