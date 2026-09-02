#!/usr/bin/env python3
"""R007: causal steering along the frozen depth-40 probe direction (D013).

Everything here was frozen in D013 before any steering behaviour was observed.
Nothing in this file may be tuned after seeing results: not the layer, not the
strengths, not the schedule, not the replicate count, not the orthogonal seed.

Intervention: at block-39 output (depth 40), add
    dh = (ds / ||beta_raw||^2) * beta_raw
so the frozen probe score moves by exactly `ds`. Applied to the final real prefix
token at prefill and to every newly generated token during decoding (sustained
newest-token schedule).

Conditions (7): beta at ds in {-2,-1,0,+1,+2} * sigma_val, plus one fixed seeded
raw-space direction orthogonal to beta, matched in ||dh||, at -2 and +2 only.

    --smoke   Stage 1: mandatory propagation checks on VAL examples only.
    (default) Stage 2: the frozen test run, executed exactly once.

Usage:
    python src/steer_termination.py --smoke --run-id r007_steer_smoke
    python src/steer_termination.py --run-id r007_steer_test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import task1_data as T

DEPTH = 40                     # val-selected (D010); block index 39
BLOCK = DEPTH - 1
SIGMA_VAL = 17.178             # frozen: SD of the depth-40 score on val (preflight)
BETA_STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]
ORTHO_STRENGTHS = [-2.0, 2.0]
ORTHO_SEED = 20260901          # fixed, frozen
TEMPERATURE = 0.7              # released setting
MAX_NEW_TOKENS = 60            # frozen cap
N_REPLICATES = 4               # frozen; never increased after results
YES_TOKEN_MIN = 20             # released YES window
YES_TOKEN_MAX = 60


def conditions():
    out = [("beta", k) for k in BETA_STRENGTHS] + [("ortho", k) for k in ORTHO_STRENGTHS]
    return out


def row_seed(filename: str, replicate: int) -> int:
    h = hashlib.sha256(f"{filename}|{replicate}".encode()).digest()
    return int.from_bytes(h[:4], "big")


class Steerer:
    """Adds a per-row vector to the block-39 output at the newest token."""

    def __init__(self, block):
        self.deltas = None          # [B, H] on device, or None
        self.enabled = False
        self.edit_index = None      # int position within the current chunk
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if not self.enabled or self.deltas is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        idx = self.edit_index if self.edit_index is not None else hs.shape[1] - 1
        hs[:, idx, :] = hs[:, idx, :] + self.deltas.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def close(self):
        self.handle.remove()


def build_directions(preflight_dir: pathlib.Path, device):
    beta = np.load(preflight_dir / "beta_raw_depth40.npy").astype(np.float64)
    nb2 = float(beta @ beta)
    rng = np.random.default_rng(ORTHO_SEED)
    u = rng.standard_normal(beta.size)
    u -= (u @ beta) / nb2 * beta                      # project beta out
    u /= np.linalg.norm(u)
    assert abs(float(u @ beta)) / np.linalg.norm(beta) < 1e-10, "ortho failed"

    vecs = {}
    for kind, k in conditions():
        ds = k * SIGMA_VAL
        if kind == "beta":
            dh = (ds / nb2) * beta
        else:
            ref = abs(2.0 * SIGMA_VAL) / np.sqrt(nb2)   # ||dh|| at 2 SD along beta
            dh = np.sign(k) * ref * u
        vecs[(kind, k)] = torch.tensor(dh, dtype=torch.float32, device=device)
    return beta, nb2, u, vecs


@torch.no_grad()
def generate_batch(model, tok, ids, deltas, steerer, seeds, think_id,
                   max_new=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Sample `B` continuations of one identical prompt under per-row deltas.

    Common random numbers: row i draws from its own CPU generator seeded by
    seeds[i], so the same (prefix, replicate) sees identical uniforms in every
    condition. Sampling is pure temperature (no top-p / top-k), matching the
    released settings.
    """
    B = deltas.shape[0]
    device = ids.device
    batch = ids.unsqueeze(0).expand(B, -1).contiguous()

    steerer.deltas = deltas
    steerer.enabled = True
    steerer.edit_index = batch.shape[1] - 1            # last real prefix token
    # Base transformer only, then the LM head at the LAST position: a full
    # forward would materialise [B, L, 151936] logits and OOM on a long prefix.
    # This mirrors the R001 extraction, which never built full-sequence logits.
    out = model.model(batch, use_cache=True)
    past = out.past_key_values
    logits = model.lm_head(out.last_hidden_state[:, -1, :]).float()
    steerer.edit_index = 0                             # newest token from now on

    gens = [[] for _ in range(B)]
    fired = np.full(B, -1, dtype=np.int64)
    cpu_gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    first_logits = logits.clone()

    for step in range(max_new):
        probs = torch.softmax(logits / temperature, dim=-1)
        cum = probs.cumsum(-1)
        u = torch.tensor([torch.rand(1, generator=g).item() for g in cpu_gens],
                         device=device, dtype=cum.dtype)
        nxt = torch.searchsorted(cum, u.unsqueeze(1)).clamp(max=probs.shape[-1] - 1)
        for i in range(B):
            t = int(nxt[i, 0])
            gens[i].append(t)
            if t == think_id and fired[i] < 0:
                fired[i] = step + 1                    # 1-indexed token position
        if (fired >= 0).all():
            break
        out = model.model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[:, -1, :]).float()

    steerer.enabled = False
    return gens, fired, first_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight-run", default="r007_steering_preflight")
    ap.add_argument("--source-run", default="r001_qwen32b")
    args = ap.parse_args()

    run_dir = T.REPO / "artifacts/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((T.REPO / "artifacts/runs" / args.source_run
                          / "config.json").read_text())

    tok = AutoTokenizer.from_pretrained(src_cfg["model"], revision=src_cfg["revision"])
    build_prompt = T.load_build_thinking_prompt()
    think_id = tok("</think>", add_special_tokens=False)["input_ids"]
    assert len(think_id) == 1
    think_id = int(think_id[0])

    model = AutoModelForCausalLM.from_pretrained(
        src_cfg["model"], revision=src_cfg["revision"],
        dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    # Nothing here ever trains. Without this the Stage-1 propagation forward
    # builds an autograd graph across all 64 blocks for a batch of 7 long
    # sequences and OOMs; only generate_batch carried @torch.no_grad.
    torch.set_grad_enabled(False)
    dev = next(model.parameters()).device
    offloaded = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if offloaded:
        raise SystemExit(f"ABORT: {len(offloaded)} parameters not CUDA-resident")
    print(f"model on {dev}; free "
          f"{torch.cuda.mem_get_info()[0]/2**30:.1f} GB / "
          f"{torch.cuda.mem_get_info()[1]/2**30:.1f} GB")

    blocks = model.model.layers
    steerer = Steerer(blocks[BLOCK])
    beta, nb2, u_perp, vecs = build_directions(
        T.REPO / "artifacts/runs" / args.preflight_run, dev)
    conds = conditions()
    delta_mat = torch.stack([vecs[c] for c in conds])          # [7, H]
    print(f"directions ready: ||beta||={np.sqrt(nb2):.4f}, "
          f"||dh|| at 2SD = {float(torch.linalg.norm(vecs[('beta', 2.0)])):.3f}, "
          f"ortho ||dh|| = {float(torch.linalg.norm(vecs[('ortho', 2.0)])):.3f}")

    split = "val" if args.smoke else "test"
    recs = T.load_records(split)
    print(f"{split}: {len(recs)} prefixes")

    # ---------------- Stage 1: mandatory propagation checks (val only) ----------
    if args.smoke:
        checks = []
        for rec in recs[:3]:
            prompt = build_prompt(tok, rec["prompt_text"], rec["cot_prefix"])
            ids = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"],
                               device=dev)
            cap = {}
            h40 = blocks[BLOCK].register_forward_hook(
                lambda m, i, o: cap.__setitem__("h40", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu()))
            h64 = blocks[-1].register_forward_hook(
                lambda m, i, o: cap.__setitem__("h64", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu()))
            steerer.deltas = delta_mat
            steerer.enabled = True
            steerer.edit_index = ids.shape[0] - 1
            out = model.model(ids.unsqueeze(0).expand(len(conds), -1).contiguous(),
                              use_cache=False)
            steerer.enabled = False
            h40s = cap["h40"].numpy().astype(np.float64)
            h64s = cap["h64"].numpy().astype(np.float64)
            lg = (model.lm_head(out.last_hidden_state[:, -1, :])
                  .float().cpu().numpy().astype(np.float64))
            h40.remove(); h64.remove()

            base = conds.index(("beta", 0.0))
            lp = torch.log_softmax(torch.tensor(lg), dim=-1).numpy()
            for ci, c in enumerate(conds):
                ds_req = c[1] * SIGMA_VAL if c[0] == "beta" else 0.0
                ds_obs = float(beta @ (h40s[ci] - h40s[base]))
                checks.append({
                    "filename": rec["filename"], "condition": f"{c[0]}{c[1]:+g}",
                    "ds_requested": ds_req, "ds_observed": ds_obs,
                    "d_h64": float(np.linalg.norm(h64s[ci] - h64s[base])),
                    "h64_norm": float(np.linalg.norm(h64s[base])),
                    "d_logits": float(np.abs(lg[ci] - lg[base]).max()),
                    "think_logprob": float(lp[ci, think_id]),
                    "d_think_logprob": float(lp[ci, think_id] - lp[base, think_id]),
                })
        for c in checks[:len(conds)]:
            print(f"  {c['condition']:>9s}  ds req={c['ds_requested']:+8.2f} "
                  f"obs={c['ds_observed']:+8.2f}  |dh64|={c['d_h64']:8.2f} "
                  f"(||h64||={c['h64_norm']:.0f})  max|dlogit|={c['d_logits']:7.3f}  "
                  f"d_think_logprob={c['d_think_logprob']:+.3f}")

        beta_rows = [c for c in checks if c["condition"].startswith("beta")
                     and c["ds_requested"] != 0]
        err = max(abs(c["ds_observed"] - c["ds_requested"]) / abs(c["ds_requested"])
                  for c in beta_rows)
        nz = [c for c in checks if c["condition"] != "beta+0"]
        ok_h64 = min(c["d_h64"] for c in nz) > 1e-2
        ok_lg = min(c["d_logits"] for c in nz) > 1e-3
        print(f"\nPROPAGATION: ds relative error {err:.3%} | "
              f"depth-64 changes: {ok_h64} | logits change: {ok_lg}")
        if err > 0.10 or not ok_h64 or not ok_lg:
            raise SystemExit("ABORT: mandatory propagation checks failed")

        # coherence: one short generation at the strongest strengths
        rec = recs[0]
        prompt = build_prompt(tok, rec["prompt_text"], rec["cot_prefix"])
        ids = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"], device=dev)
        gens, fired, _ = generate_batch(model, tok, ids, delta_mat, steerer,
                                        [row_seed(rec["filename"], 0)] * len(conds),
                                        think_id)
        print("\ncoherence sample (val, 60 tokens):")
        for c, g, f in zip(conds, gens, fired):
            txt = tok.decode(g).replace("\n", " ")[:110]
            print(f"  {c[0]}{c[1]:+g}  think@{f if f>0 else '-':>3}  {txt!r}")
        (run_dir / "metrics.json").write_text(json.dumps(
            {"run_id": args.run_id, "stage": "1 (val smoke)",
             "propagation_checks": checks,
             "ds_max_relative_error": err,
             "depth64_changes": bool(ok_h64), "logits_change": bool(ok_lg),
             "samples": [{"condition": f"{c[0]}{c[1]:+g}",
                          "think_token_position": int(f),
                          "text": tok.decode(g)}
                         for c, g, f in zip(conds, gens, fired)]}, indent=2))
        print(f"\nSTAGE 1 PASSED -> {run_dir/'metrics.json'}")
        steerer.close()
        return 0

    # ---------------- Stage 2: the frozen test run, once ----------------------
    lengths = {p.name: json.loads(p.read_text())["token_length"]
               for p in (T.DATA_ROOT / "qwen-3-32b" / split).glob("*.json")}
    out_rows = []
    t0 = time.time()
    for pi, rec in enumerate(recs):
        prompt = build_prompt(tok, rec["prompt_text"], rec["cot_prefix"])
        ids = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"], device=dev)
        for rep in range(N_REPLICATES):
            seeds = [row_seed(rec["filename"], rep)] * len(conds)
            gens, fired, _ = generate_batch(model, tok, ids, delta_mat, steerer,
                                            seeds, think_id)
            for c, g, f in zip(conds, gens, fired):
                out_rows.append({
                    "filename": rec["filename"], "question_id": rec["question_id"],
                    "label": rec["label"], "token_length": lengths[rec["filename"]],
                    "replicate": rep, "condition": f"{c[0]}{c[1]:+g}",
                    "kind": c[0], "strength": c[1],
                    "think_pos": int(f), "n_generated": len(g),
                })
        if (pi + 1) % 10 == 0:
            el = (time.time() - t0) / 60
            print(f"  prefix {pi+1}/{len(recs)}  elapsed={el:.1f}min  "
                  f"eta={el/(pi+1)*(len(recs)-pi-1):.1f}min", flush=True)

    with (run_dir / "generations.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id, "decision": "D013", "depth": DEPTH,
        "sigma_val": SIGMA_VAL, "beta_strengths": BETA_STRENGTHS,
        "ortho_strengths": ORTHO_STRENGTHS, "ortho_seed": ORTHO_SEED,
        "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS,
        "n_replicates": N_REPLICATES, "split": split,
        "schedule": "sustained newest-token at block-39 output",
        "seeding": "sha256(filename|replicate), shared across all conditions",
        "model": src_cfg["model"], "revision": src_cfg["revision"],
    }, indent=2))
    print(f"\nwrote {run_dir/'generations.csv'} ({len(out_rows)} rows) "
          f"in {(time.time()-t0)/60:.1f} min")
    steerer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
