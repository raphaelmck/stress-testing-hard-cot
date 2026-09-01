#!/usr/bin/env python3
"""R001: cache Qwen3-32B last-real-prefix-token activations for Task 1.

Forward passes only -- no generation, no resampling, no truncation.

Frozen decisions this script implements (see DECISIONS.md):
  D004  model pinned to Qwen/Qwen3-32B @ 9216db5, prompt built by the released
        upstream helper, never reconstructed here
  D005  preregistered depths 8/24/40/56/64 (zero-indexed blocks 7/23/39/55/63)
  D006  question_id is preserved on every row so later CIs can cluster by question
  D007  last-token output statistics are cached in the SAME forward pass and are
        NOT to be analysed until the activation reproduction is frozen

Padding: RIGHT padding is used deliberately. We never generate, so the
decoder-only right-padding warning does not apply; with right padding the valid
tokens occupy positions 0..L-1 exactly as in the unpadded singleton, so default
position ids already match and no position-id surgery is needed. The last real
token is `attention_mask.sum(1) - 1`. (If anyone ever switches to left padding,
position ids must be built from the attention mask explicitly -- left padding
alone does NOT preserve activations.)

Usage:
    python src/extract_task1_activations.py --smoke --run-id r001_smoke
    python src/extract_task1_activations.py --run-id r001_extract
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import platform
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import task1_data as T

THINK_CLOSE = "</think>"

OUTPUT_FEATURES = [
    "think_logit",
    "think_logprob",
    "think_margin",       # think_logit - max logit over non-</think> tokens
    "next_token_entropy",
    "top1_token_id",
    "top1_logprob",
]

ROW_COLUMNS = [
    "split", "filename", "question_id", "rollout_idx", "prefix_idx",
    "label", "family", "n_tokens", "shard", "row_in_shard",
]


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------

def resolve_modules(model):
    """Return (base_transformer, decoder_layers, final_norm, lm_head).

    The base transformer is run directly during extraction so that the causal
    LM head is never evaluated over the sequence. Qwen3ForCausalLM.forward
    defaults to logits_to_keep=0, which materialises [B, seq_len, vocab]
    logits -- 151k vocab over a 16k-token prefix -- and would defeat the whole
    point of the last-token-only design.
    """
    base = getattr(model, "model", model)
    base = getattr(base, "language_model", base)  # some v5 wrappers nest here
    layers = getattr(base, "layers", None)
    norm = getattr(base, "norm", None)
    head = model.get_output_embeddings()
    if layers is None or norm is None or head is None:
        raise RuntimeError(
            "could not resolve decoder layers / final norm / lm_head on "
            f"{type(model).__name__}; inspect the model structure before running")
    return base, layers, norm, head


def assert_cuda_resident(model, report) -> None:
    """Abort if any parameter or buffer landed on CPU or disk.

    device_map="auto" silently offloads when the card is too small: a 40 GB A100
    would "work" but stream 25 GB of weights over PCIe on every batch. R001
    requires a fully resident model, so an undersized GPU must fail loudly here
    rather than produce correct results at an unusable speed.
    """
    dmap = getattr(model, "hf_device_map", None)
    if dmap:
        placements = sorted({str(v) for v in dmap.values()})
        report(f"  device_map placements: {placements}")
        bad = sorted({k for k, v in dmap.items() if str(v) in ("cpu", "disk")})
        if bad:
            raise SystemExit(
                f"ABORT: {len(bad)} module(s) offloaded to CPU/disk "
                f"(e.g. {bad[:5]}). Qwen3-32B needs ~65.5 GB in bf16; this GPU "
                f"is too small. Use an 80 GB card (A100 80GB / H100) -- the "
                f"40 GB A100 variant is insufficient.")

    off = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.device.type != "cuda":
            off.setdefault(t.device.type, []).append(name)
    if off:
        summary = {k: f"{len(v)} tensors, e.g. {v[0]}" for k, v in off.items()}
        raise SystemExit(f"ABORT: model is not fully CUDA-resident: {summary}")

    devs = sorted({str(t.device) for t in model.parameters()})
    n = sum(t.numel() for t in model.parameters())
    report(f"  all {n/1e9:.2f}B parameters CUDA-resident across {devs}")


class LastTokenTap:
    """Forward hooks that keep ONLY the last real token's vector per depth.

    Deliberately does not use output_hidden_states=True: that materialises all
    65 hidden states for the full sequence (10.9 GB at batch 8 x seq 2048),
    while these hooks keep batch x hidden per depth (~0.4 MB).
    """

    def __init__(self, layers, block_indices, depths):
        self.depths = depths
        self.store: dict[int, torch.Tensor] = {}
        self.last_idx: torch.Tensor | None = None
        self.handles = []
        for depth, block in zip(depths, block_indices):
            self.handles.append(
                layers[block].register_forward_hook(self._make(depth)))

    def _make(self, depth):
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(h.shape[0], device=h.device)
            self.store[depth] = h[rows, self.last_idx.to(h.device)].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def forward_batch(base, tap, norm, head, input_ids, attention_mask,
                  return_base_output=False):
    """One forward pass; returns (acts [B, n_depths, H], output features [B, F]).

    Runs the BASE transformer, so no vocabulary-sized tensor is ever produced by
    the forward itself. The only logits computed anywhere are the [B, vocab]
    slice built below from the last real token.
    """
    tap.last_idx = attention_mask.sum(dim=1) - 1
    out = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if hasattr(out, "logits"):
        raise RuntimeError(
            "base forward returned logits -- a vocabulary-sized tensor was "
            "materialised; check that the base transformer, not the CausalLM, "
            "is being called")

    acts = torch.stack([tap.store[d] for d in tap.depths], dim=1)  # [B, D, H]

    # Output statistics from the FINAL depth's last-token hidden state.
    # Applying norm + lm_head to the [B, H] slice avoids ever building a
    # [B, seq_len, vocab] logits tensor (151k vocab x long prefixes).
    h_last = tap.store[tap.depths[-1]]
    logits = head(norm(h_last)).float()                    # [B, V]
    logprobs = torch.log_softmax(logits, dim=-1)
    probs = logprobs.exp()

    think_id = tap.think_id
    think_logit = logits[:, think_id]
    think_logprob = logprobs[:, think_id]
    other = logits.clone()
    other[:, think_id] = -math.inf
    margin = think_logit - other.max(dim=-1).values
    entropy = -(probs * logprobs).sum(dim=-1)
    top1 = logits.argmax(dim=-1)
    top1_logprob = logprobs.gather(1, top1[:, None]).squeeze(1)

    feats = torch.stack([think_logit, think_logprob, margin, entropy,
                         top1.float(), top1_logprob], dim=1)
    if return_base_output:
        return acts.to(torch.float16).cpu(), feats.cpu(), out
    return acts.to(torch.float16).cpu(), feats.cpu()


# --------------------------------------------------------------------------
# work list
# --------------------------------------------------------------------------

def build_worklist(splits, tokenizer, build_prompt, max_positions, smoke_n=0):
    """Tokenise every selected row. No model is loaded yet.

    Aborts loudly if any example exceeds the model's configured context limit;
    R001 forbids truncation, so an over-long example is a hard stop, never a
    silent trim.
    """
    rows: list[dict] = []
    for split in splits:
        recs = T.load_records(split)
        if split == "train":
            recs = T.sample_train_rows(recs)
        else:
            for r in recs:
                r["family"] = "eval"
        rows.extend(recs)

    if smoke_n:
        rows = pick_smoke_rows(rows, smoke_n)

    over = []
    for r in rows:
        prompt = build_prompt(tokenizer, r["prompt_text"], r["cot_prefix"])
        if not prompt.endswith(r["cot_prefix"]):
            raise RuntimeError(
                f"constructed input does not end with the released cot_prefix: "
                f"{r['split']}/{r['filename']}")
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r["prompt"] = prompt
        r["input_ids"] = ids
        r["n_tokens"] = len(ids)
        r["n_words"] = len(r["cot_prefix"].split())
        if len(ids) > max_positions:
            over.append((r["split"], r["filename"], len(ids)))

    if over:
        raise RuntimeError(
            f"{len(over)} example(s) exceed the model context limit "
            f"({max_positions}); R001 forbids truncation. First few: {over[:5]}")
    return rows


def pick_smoke_rows(rows, n):
    """10 examples spanning labels and splits, chosen deterministically."""
    picked, seen = [], set()
    plan = [("train", "yes", 2), ("train", "no", 2),
            ("val", "yes", 1), ("val", "no", 1),
            ("test", "yes", 1), ("test", "no", 1),
            ("ood_test", "yes", 1), ("ood_test", "no", 1)]
    for split, label, k in plan:
        cands = sorted((r for r in rows
                        if r["split"] == split and r["label"] == label),
                       key=lambda r: r["filename"])
        for r in cands[:k]:
            if r["filename"] not in seen:
                seen.add(r["filename"])
                picked.append(r)
    return picked[:n]


def make_batches(rows, token_budget, max_batch):
    """Length-bucketed, token-budgeted batches.

    Sorting by length keeps the long tail (train prefixes reach ~16.5k tokens
    while eval tops out near 2.9k) from forcing a global batch size of 1.
    Batching affects only grouping, never per-row results.
    """
    ordered = sorted(rows, key=lambda r: (r["n_tokens"], r["split"], r["filename"]))
    batches, cur, cur_max = [], [], 0
    for r in ordered:
        new_max = max(cur_max, r["n_tokens"])
        if cur and ((len(cur) + 1) * new_max > token_budget or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [r], r["n_tokens"]
        else:
            cur.append(r)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------

def run_smoke(base, tokenizer, tap, norm, head, rows, device, report):
    report("\n=== SMOKE TEST ===")
    report(f"{'#':>2} {'split':9s} {'label':4s} {'tokens':>7s}  filename")
    for i, r in enumerate(rows):
        report(f"{i:>2} {r['split']:9s} {r['label']:4s} {r['n_tokens']:7d}  {r['filename']}")

    report("\n-- input reconstruction --")
    for r in rows:
        decoded = tokenizer.decode(r["input_ids"])
        ok_str = r["prompt"].endswith(r["cot_prefix"])
        ok_dec = decoded.endswith(r["cot_prefix"][-80:])
        report(f"  {r['filename'][:52]:52s} prompt_endswith_prefix={ok_str} "
               f"decoded_endswith_prefix_tail={ok_dec}")
        if not (ok_str and ok_dec):
            raise RuntimeError(f"input reconstruction failed for {r['filename']}")
    report(f"  qid preserved on all rows: "
           f"{all(r.get('question_id') for r in rows)}")

    report("\n-- singleton vs right-padded batch, all five depths --")
    # Pair each example with a longer one so it is genuinely padded.
    ordered = sorted(rows, key=lambda r: r["n_tokens"])
    worst_cos, worst_abs = 1.0, 0.0
    for a, b in [(ordered[0], ordered[-1]), (ordered[1], ordered[-2])]:
        single = run_rows(base, tap, norm, head, [a], tokenizer, device)[0][0]
        pair = run_rows(base, tap, norm, head, [a, b], tokenizer, device)[0][0]
        report(f"  {a['filename'][:44]:44s} ({a['n_tokens']} tok) padded to {b['n_tokens']}")
        for j, d in enumerate(tap.depths):
            x, y = single[j].float(), pair[j].float()
            cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
            mad = (x - y).abs().max().item()
            worst_cos, worst_abs = min(worst_cos, cos), max(worst_abs, mad)
            report(f"      depth {d:2d}: cosine={cos:.6f}  max_abs_diff={mad:.3e}")
    report(f"  worst cosine={worst_cos:.6f}  worst max_abs_diff={worst_abs:.3e}")

    report("\n-- no full-sequence logits are materialised --")
    acts_b, feats_b, out = run_rows(base, tap, norm, head, rows, tokenizer, device,
                                    return_base_output=True)
    lh = out.last_hidden_state
    vocab = head.weight.shape[0]
    report(f"  base forward returns last_hidden_state {tuple(lh.shape)} "
           f"(hidden={lh.shape[-1]}, vocab={vocab})")
    report(f"  forward output exposes .logits: {hasattr(out, 'logits')}")
    if lh.shape[-1] == vocab:
        raise RuntimeError("forward produced a vocabulary-sized tensor")
    report(f"  only vocab-sized tensor built anywhere: [batch, vocab] = "
           f"{(len(rows), vocab)} from the last real token")

    report("\n-- hooked final depth + norm == base model's own last hidden state --")
    idx = tap.last_idx
    ref = lh[torch.arange(lh.shape[0], device=lh.device), idx.to(lh.device)].float()
    manual = norm(tap.store[tap.depths[-1]]).float()
    mad = (ref - manual).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(ref, manual, dim=1).min().item()
    report(f"  max_abs_diff={mad:.3e}  min_cosine={cos:.6f}")
    if mad > 1e-2:
        raise RuntimeError(
            "manual norm(hook) disagrees with the model's last_hidden_state -- "
            "the activation index or the final-norm application is wrong")

    report("\n-- non-degeneracy (guards against a stale/shared hook tensor) --")
    acts, feats = run_rows(base, tap, norm, head, rows, tokenizer, device)
    pair_d = [(acts[i].float() - acts[j].float()).abs().max().item()
              for i in range(len(rows)) for j in range(i + 1, len(rows))]
    depth_d = [(acts[:, j].float() - acts[:, k].float()).abs().max().item()
               for j in range(len(tap.depths)) for k in range(j + 1, len(tap.depths))]
    report(f"  distinct examples differ: min max_abs_diff over pairs = {min(pair_d):.3e}")
    report(f"  distinct depths differ:   min max_abs_diff over pairs = {min(depth_d):.3e}")
    if min(pair_d) == 0.0 or min(depth_d) == 0.0:
        raise RuntimeError(
            "two examples or two depths produced identical activations -- the "
            "padding comparison above would be vacuous; inspect the hooks")

    report("\n-- shapes / finiteness --")
    report(f"  activations {tuple(acts.shape)} (rows, depths, hidden) dtype={acts.dtype}")
    report(f"  per-depth slice is [batch, hidden]: {tuple(acts[:, 0].shape)}")
    report(f"  all activations finite: {bool(torch.isfinite(acts.float()).all())}")
    report(f"  all output features finite: {bool(torch.isfinite(feats).all())}")
    if not torch.isfinite(acts.float()).all() or not torch.isfinite(feats).all():
        raise RuntimeError("non-finite values in smoke test output")
    report("  output features cached but NOT analysed (D007 embargo)")
    return worst_cos, worst_abs


def run_stress(base, tokenizer, tap, norm, head, rows, device, report,
               token_budget, max_batch):
    """Memory/throughput probes before committing to the full extraction.

    Two tests, in increasing order of importance:
      B. a representative ~1.5-2.5k-token batch at the configured budget;
      C. the single longest selected example alone at batch size 1.

    (C) is the one that de-risks the run: the train split has a long tail that
    the eval splits do not (up to ~16.5k tokens vs ~2.9k). If (C) OOMs, inspect
    the attention implementation -- do NOT truncate, and do not drop the example.
    """
    def peak():
        if device != "cuda":
            return None
        return (torch.cuda.max_memory_allocated() / 1e9,
                torch.cuda.max_memory_reserved() / 1e9,
                torch.cuda.mem_get_info()[0] / 1e9)

    def run_and_report(name, batch):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        toks = [r["n_tokens"] for r in batch]
        t0 = time.time()
        acts, feats = run_rows(base, tap, norm, head, batch, tokenizer, device)
        dt = time.time() - t0
        padded = len(batch) * max(toks)
        report(f"  {name}: batch={len(batch)} tokens={min(toks)}-{max(toks)} "
               f"padded_total={padded} time={dt:.2f}s")
        pk = peak()
        if pk:
            report(f"    peak allocated={pk[0]:.1f} GB  reserved={pk[1]:.1f} GB  "
                   f"free after={pk[2]:.1f} GB")
            if pk[2] < 3.0:
                report("    WARNING: under 3 GB free at peak -- lower --token-budget")
        report(f"    finite={bool(torch.isfinite(acts.float()).all())} "
               f"shape={tuple(acts.shape)}")
        return dt

    report("\n=== STRESS TEST ===")
    report(f"token_budget={token_budget} max_batch={max_batch}")

    mid = sorted((r for r in rows if 1500 <= r["n_tokens"] <= 2500),
                 key=lambda r: r["filename"])
    if mid:
        batches = make_batches(mid, token_budget, max_batch)
        dt = run_and_report("B representative", batches[0])
        n_rows, n_batches = len(rows), len(make_batches(rows, token_budget, max_batch))
        report(f"    extrapolated: {n_batches} batches for {n_rows} rows "
               f"~= {dt*n_batches/60:.0f} min at this batch's rate (crude)")
    else:
        report("  B representative: no rows in the 1.5-2.5k token band")

    longest = max(rows, key=lambda r: r["n_tokens"])
    report(f"  C longest example: {longest['split']}/{longest['filename']}")
    run_and_report("C longest", [longest])

    report("\n=== STRESS TEST PASSED ===")
    report("If free memory at peak was tight, rerun the full extraction with a "
           "LOWER --token-budget. Never raise it above what was tested here.")


def run_rows(base, tap, norm, head, rows, tokenizer, device, **kw):
    enc = tokenizer.pad({"input_ids": [r["input_ids"] for r in rows]},
                        padding=True, return_tensors="pt")
    return forward_batch(base, tap, norm, head,
                         enc["input_ids"].to(device),
                         enc["attention_mask"].to(device), **kw)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="run the 10-example smoke test and stop")
    ap.add_argument("--stress", action="store_true",
                    help="run the representative-batch and longest-example "
                         "memory tests and stop (GPU steps B and C)")
    ap.add_argument("--splits", nargs="+", default=T.SPLITS)
    ap.add_argument("--model", default=T.MODEL_ID)
    ap.add_argument("--revision", default=T.MODEL_REVISION)
    ap.add_argument("--depths", type=int, nargs="+", default=None,
                    help="LOCAL PLUMBING ONLY -- overriding the preregistered "
                         "depths is recorded in config.json and invalidates R001")
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max padded tokens per batch (batch_size * longest)")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.depths and args.model == T.MODEL_ID:
        ap.error("--depths may only be used together with a non-default --model")
    depths = args.depths or T.DEPTHS
    blocks = [d - 1 for d in depths]

    run_dir = T.REPO / "artifacts/runs" / args.run_id
    act_dir = run_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    log = log_path.open("a")

    def report(msg=""):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "right"          # deliberate; see module docstring
    build_prompt = T.load_build_thinking_prompt()

    report(f"model={args.model} revision={args.revision}")
    report(f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
           f"context_limit={cfg.max_position_embeddings}")
    report(f"depths={depths} -> zero-indexed blocks={blocks}"
           f"{'  [OVERRIDDEN -- not R001]' if args.depths else ''}")
    if max(blocks) >= cfg.num_hidden_layers:
        raise SystemExit(f"depth {max(depths)} exceeds {cfg.num_hidden_layers} layers")

    t0 = time.time()
    rows = build_worklist(args.splits, tokenizer, build_prompt,
                          cfg.max_position_embeddings,
                          smoke_n=10 if args.smoke else 0)
    if args.stress and args.smoke:
        raise SystemExit("--stress and --smoke are mutually exclusive")
    report(f"work list: {len(rows)} rows tokenised in {time.time()-t0:.1f}s; "
           f"max tokens={max(r['n_tokens'] for r in rows)}")

    # Freeze the sample manifest BEFORE any forward pass.
    manifest_path = run_dir / "sample_manifest.csv"
    T.write_sample_manifest(manifest_path, rows)
    digest = T.sample_hash(rows)
    report(f"frozen sample manifest: {manifest_path.name}  sha256={digest[:16]}")

    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id,
        "model": args.model,
        "revision": args.revision,
        "depths": depths,
        "block_indices": blocks,
        "preregistered_depths_used": args.depths is None,
        "splits": args.splits,
        "train_sample": {"seed": T.SAMPLE_SEED,
                         "per_family_per_label": T.PER_FAMILY_PER_LABEL},
        "sample_sha256": digest,
        "n_rows": len(rows),
        "truncation": "none",
        "padding_side": "right",
        "use_cache": False,
        "activation_site": "last real prefix token (attention_mask.sum(1)-1)",
        "activation_dtype": "float16",
        "output_features": OUTPUT_FEATURES,
        "output_feature_embargo": "D007 -- cached, not to be analysed",
        "token_budget": args.token_budget,
        "max_batch": args.max_batch,
        "smoke": args.smoke,
    }, indent=2))

    (run_dir / "metadata.json").write_text(json.dumps({
        "run_id": args.run_id,
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 cwd=T.REPO, capture_output=True, text=True
                                 ).stdout.strip() or "unknown",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "torch": torch.__version__,
    }, indent=2))

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    report(f"device={device} dtype=bfloat16")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            report(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        report(f"  free before load: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None)
    if device != "cuda":
        model.to(device)
    model.eval()

    if device == "cuda":
        assert_cuda_resident(model, report)
        free, total = torch.cuda.mem_get_info()
        report(f"  free after load:  {free/1e9:.1f} GB / {total/1e9:.1f} GB")

    base, layers, norm, head = resolve_modules(model)
    tap = LastTokenTap(layers, blocks, depths)
    tap.think_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    report(f"{THINK_CLOSE} token id={tap.think_id} "
           f"(single token: {len(tokenizer.encode(THINK_CLOSE, add_special_tokens=False))==1})")

    if args.stress:
        run_stress(base, tokenizer, tap, norm, head, rows, device, report,
                   args.token_budget, args.max_batch)
        return 0

    if args.smoke:
        cos, mad = run_smoke(base, tokenizer, tap, norm, head, rows, device, report)
        report("\n=== SMOKE TEST PASSED ===")
        report("Do NOT launch the full extraction until this output is reviewed.")
        return 0

    # ---- full extraction, resumable ----
    done_path = run_dir / "done.jsonl"
    done = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            d = json.loads(line)
            done.add((d["split"], d["filename"]))
        report(f"resuming: {len(done)} rows already extracted")
    todo = [r for r in rows if (r["split"], r["filename"]) not in done]
    batches = make_batches(todo, args.token_budget, args.max_batch)
    report(f"{len(todo)} rows remaining in {len(batches)} batches")

    shard_idx = len(list(act_dir.glob("shard_*.npz")))
    buf_acts, buf_feats, buf_rows = [], [], []
    t0 = time.time()

    def flush():
        nonlocal shard_idx, buf_acts, buf_feats, buf_rows
        if not buf_rows:
            return
        name = f"shard_{shard_idx:05d}"
        np.savez(act_dir / f"{name}.npz",
                 activations=torch.cat(buf_acts).numpy(),
                 output_features=torch.cat(buf_feats).numpy())
        with (act_dir / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_COLUMNS)
            w.writeheader()
            for i, r in enumerate(buf_rows):
                w.writerow({**{k: r.get(k) for k in ROW_COLUMNS},
                            "shard": name, "row_in_shard": i})
        with done_path.open("a") as fh:
            for r in buf_rows:
                fh.write(json.dumps({"split": r["split"], "filename": r["filename"]}) + "\n")
        report(f"  wrote {name} ({len(buf_rows)} rows)")
        shard_idx += 1
        buf_acts, buf_feats, buf_rows = [], [], []

    for bi, batch in enumerate(batches):
        acts, feats = run_rows(base, tap, norm, head, batch, tokenizer, device)
        if not torch.isfinite(acts.float()).all():
            raise RuntimeError(f"non-finite activations in batch {bi}")
        buf_acts.append(acts)
        buf_feats.append(feats)
        buf_rows.extend(batch)
        if len(buf_rows) >= args.shard_size:
            flush()
        if bi % 25 == 0:
            el = time.time() - t0
            report(f"batch {bi+1}/{len(batches)}  elapsed={el/60:.1f}min  "
                   f"eta={el/(bi+1)*(len(batches)-bi-1)/60:.1f}min")
    flush()

    report(f"extraction complete in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
