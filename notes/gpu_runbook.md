# R001 GPU runbook — Qwen3-32B activation extraction on the Mila cluster

Everything that can be done on a laptop is done and committed. This is the exact
sequence for the GPU stage. Read `STATE.md` and `DECISIONS.md` first; do not deviate
from the frozen decisions to make something fit.

Compute: **Mila cluster**, H100 (80 GB) or `a100l` (A100 80 GB). Colab was evaluated
and rejected — see `DECISIONS.md` D009.

---

## 0. What this run needs

- 80 GB of GPU memory, single card. Qwen3-32B is 65.5 GB in bf16.
  The extractor **aborts** if any parameter lands on CPU/disk (D004); it will not
  silently offload, so an undersized card fails fast.
- ~70 GB of disk for the HF model cache (put it on `$SCRATCH`, not `$HOME`).
- ~1.5 GB for the upstream dataset clone.
- Roughly 30–50 min of H100 time for the extraction itself, plus model load.
- Output is tiny: ~215 MB of activations for all 4,216 rows.

**Verify the partition/GPU names on the cluster rather than trusting this file:**

```bash
sinfo -o "%20P %10G %10D %t" | sort -u        # partitions and gres names
sacctmgr show assoc where user=$USER format=account,partition,qos
```

---

## 1. First-time setup (login node)

```bash
cd $SCRATCH
git clone https://github.com/raphaelmck/nanda-w27-app.git
cd nanda-w27-app

# upstream dataset -- gitignored here, pinned by commit (D003)
git clone https://github.com/Centrattic/cot-proxy-tasks.git
git -C cot-proxy-tasks checkout 4482324b5e4a6277fa3bd544785cbd9875e11694
```

Python env. The lockfile resolves for linux/x86_64, so `uv sync` reproduces the
validated environment exactly:

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `uv` is unavailable, `module avail python` and build a 3.12 venv from
`pyproject.toml` — but prefer `uv sync`, because the lockfile is what makes a
`RESULTS.md` commit hash meaningful.

Point the HF cache at scratch so 65 GB does not land in `$HOME` (which has a small
quota) and survives between jobs:

```bash
export HF_HOME=$SCRATCH/hf_cache
mkdir -p $HF_HOME
```

Add that export to your `~/.bashrc` or to every job script — a job that silently
re-downloads 65 GB wastes most of its allocation.

**Prefetch the weights from a login node** (login nodes have internet; this avoids
burning GPU time on a download):

```bash
HF_HOME=$SCRATCH/hf_cache uv run python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-32B',
                      revision='9216db5781bf21249d130ec9da846c4624c16137')
print(p)
"
du -sh $SCRATCH/hf_cache
```

---

## 2. GATE: re-verify the prompt on the cluster

This catches a silent activation-site mismatch and must pass **on the cluster**, not
just on the laptop. The prompt string comes from `tokenizer.apply_chat_template`, so
the transformers version and the model revision jointly determine it (D004).

```bash
HF_HOME=$SCRATCH/hf_cache uv run python -c "
import sys; sys.path.insert(0,'src')
import task1_data as T, transformers, torch
from transformers import AutoTokenizer
print('transformers', transformers.__version__, '| torch', torch.__version__)
tok = AutoTokenizer.from_pretrained(T.MODEL_ID, revision=T.MODEL_REVISION)
p = T.load_build_thinking_prompt()(tok, 'What is 2+2?', 'Let me think. Two plus two')
print(repr(p))
assert p.count('<think>') == 1 and '</think>' not in p, 'PROMPT FORMAT CHANGED - STOP'
assert p.endswith('Two plus two')
print('PROMPT OK')
"
```

Expected, exactly:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nLet me think. Two plus two
```

A second `<think>`, or an empty `<think>\n\n</think>` block, means **stop**: every
activation cached under it would sit at the wrong site. Fix the version, never the
science.

(No GPU needed for this step — it runs on a login node.)

---

## 3. Steps A/B/C — interactive session

Grab a card interactively so failures are immediate:

Verified on the Mila cluster 2026-09-01 with `sinfo`: `a100l` (80 GB A100) is
available in `main`, `unkillable` and `short-unkillable`; `h100` exists **only** in
`short-unkillable` (cn-n[001-002]). Plain `a100` (cn-d001, cn-k[001-004]) is the
40 GB variant and must not be used. So the default request is `main` + `a100l`.

```bash
salloc --partition=main --gres=gpu:a100l:1 --cpus-per-task=8 --mem=64G --time=2:00:00

cd $SCRATCH/nanda-w27-app
export HF_HOME=$SCRATCH/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**A — real 32B smoke test:**

```bash
uv run python src/extract_task1_activations.py --smoke --run-id r001_qwen32b_smoke
```

Non-interactively, A and B/C are the single batch job `scripts/r001_gpu_checks.sbatch`
(`set -e`, so B/C never run if A fails).

All of these must hold:

- pinned revision `9216db5…` loads;
- `all 32.76B parameters CUDA-resident across ['cuda:0']`;
- free memory after load printed (expect ~10–14 GB free on an 80 GB card);
- all ten inputs report `prompt_endswith_prefix=True`;
- singleton vs right-padded batch: cosine ~1.0 at all five real depths;
- `hooked final depth + norm == base model's own last hidden state` passes.
  The local 0.6B run gave `max_abs_diff=0` exactly; at 32B in bf16 a small non-zero
  value is normal and anything under 1e-2 passes the built-in check. Do not treat
  1e-3 as a failure;
- non-degeneracy passes; no OOM.

**If A fails, fix infrastructure only.** Do not touch the scientific design.

**B and C — memory stress:**

```bash
uv run python src/extract_task1_activations.py --stress --run-id r001_qwen32b_stress \
    --token-budget 16384 --max-batch 8
```

- **B**, representative ~1.5–2.5k-token batch at the configured budget.
- **C**, the longest selected example (~16.5k tokens) alone at batch 1. This is the
  one that de-risks the run: the train split has a long tail that eval does not.

Target **several GB free at peak**, not maximum utilization — one unlucky batch must
not OOM a 40-minute run. The script warns below 3 GB free.

If C OOMs despite `use_cache=False`, inspect the attention implementation
(`sdpa` vs flash-attention). **Do not truncate and do not drop the example** — that
would change what R001 measures. If either test is tight, lower `--token-budget`
(8192, then 4096) and never raise it above what you tested.

---

## 4. Step D — full extraction as a batch job

`scripts/r001_extract.sbatch` is in the repo. Check the partition/gres line against
what `sinfo` reported, then:

```bash
sbatch scripts/r001_extract.sbatch
squeue -u $USER
tail -f slurm-<jobid>.out
```

4,216 rows (4,000 train + 72 val + 86 test + 58 ood_test). Progress prints every 25
batches with an ETA.

**The run is resumable.** If the job is pre-empted or times out, resubmit the same
script: it reports `resuming: N rows already extracted` and continues from the
shards on disk. Confirm `sample_sha256` in `config.json` still begins
`c261306dde08c8b9` (the full 4,216-row worklist: 4,000 frozen train rows plus all
of val/test/ood_test) — if it does not, the sample changed and the run is not R001.
The train-only 4,000-row hash is `9ae14f9e27a5f66d`, and `d9eb713bcd366b6a` is the
10-row `--smoke` worklist.

---

## 5. Fit the probes

No GPU needed — run it on a login node or locally after copying the run directory back:

```bash
uv run python src/fit_task1_probe.py --run-id r001_qwen32b
```

Outputs: `artifacts/runs/r001_qwen32b/{metrics.json,probe_scores.csv}`,
`artifacts/tables/reproduction_layer_auroc.csv`,
`artifacts/figures/reproduction_layer_auroc.png`.

To bring results home (the activations are only ~215 MB):

```bash
# from your laptop
scp -r mila:$SCRATCH/nanda-w27-app/artifacts/runs/r001_qwen32b artifacts/runs/
```

All later analysis runs from these cached activations. **No further GPU time is
needed for anything in Phase B or C.**

---

## 6. After the numbers exist

1. Apply the `STATE.md` decision rule to **max OOD AUROC across the five depths**.
2. Write the `RESULTS.md` R001 entry: question, per-hypothesis predictions, setup,
   the AUROC table with question-clustered CIs, interpretation, **evidence against
   that interpretation**, decision, artifact paths.
3. Commit the run directory, the table, and the figure.

Only then does the D007 embargo lift (the cached `</think>` logit / margin / entropy
features) and the D008 paired analysis become available. Both were preregistered so
that they are tested *after* the reproduction, not alongside it.

---

## Cluster hygiene

- Keep `$HOME` clear of the model cache; use `$SCRATCH`.
- `$SLURM_TMPDIR` is fast node-local disk but is wiped when the job ends — fine for
  transient files, wrong for the HF cache or the run directory.
- Write run artifacts under the repo checkout on `$SCRATCH`, not `$SLURM_TMPDIR`, so
  a pre-empted job can resume.
- If jobs are pre-empted often, request a shorter time limit and rely on resume
  rather than asking for one long uninterruptible block.
