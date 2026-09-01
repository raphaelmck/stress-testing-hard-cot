# R001 GPU runbook — Qwen3-32B activation extraction

Everything that can be done locally is done. This is the exact sequence for the GPU
stage. Read `STATE.md` first; do not deviate from the frozen decisions in
`DECISIONS.md` to make something fit.

---

## 0. Choose the path: Colab CLI, not the VS Code extension

Two official routes exist as of 2026:

| | Colab **CLI** (`google-colab-cli`) | Colab **VS Code extension** |
|---|---|---|
| Shape | run a local script on a remote VM, retrieve files | notebook cells against a remote kernel |
| Fits R001? | **yes** | poorly |
| Long unattended run | yes, with keep-alive daemon | needs the editor connected |
| Artifact retrieval | `colab download` | manual |

**Use the CLI.** R001 is a script that runs ~40 minutes and emits ~215 MB of shards;
it is not notebook work. The extension is the better tool for interactive poking,
and you can attach it to the same session later via `colab url --open` if you want
to inspect something by hand.

Install:

```bash
uv tool install google-colab-cli
colab version
```

macOS and Linux only (no Windows). Auth is `adc` by default; `colab new` will walk
you through sign-in on first use.

---

## 1. Provision the GPU

```bash
colab new -s r001 --gpu H100
colab status -s r001
```

**Prefer H100.** It is 80 GB. Colab's A100 exists in both 40 GB and 80 GB variants
and you do not control which you get; Qwen3-32B needs 65.5 GB in bf16, so a 40 GB
card cannot hold it. If you do use `--gpu A100`, verify before going further:

```bash
echo "import torch; p=torch.cuda.get_device_properties(0); print(p.name, p.total_memory/1e9)" | colab exec -s r001
```

Anything under ~79 GB: `colab stop -s r001` and provision again. Do not try to make
it fit. The extractor will abort by design rather than offload to CPU (D004).

---

## 2. Stage the code and data on the VM

Our repo has no remote, so upload the four source files. The upstream dataset is
public and clones far faster on the VM than it uploads.

```bash
colab exec -s r001 <<'PY'
import subprocess, pathlib
root = pathlib.Path("/content/nanda-app")
(root / "src").mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "clone", "--quiet",
                "https://github.com/Centrattic/cot-proxy-tasks.git",
                str(root / "cot-proxy-tasks")], check=True)
subprocess.run(["git", "-C", str(root / "cot-proxy-tasks"), "checkout", "--quiet",
                "4482324b5e4a6277fa3bd544785cbd9875e11694"], check=True)
print("upstream data ready")
PY

for f in task1_data.py extract_task1_activations.py fit_task1_probe.py; do
  colab upload -s r001 "src/$f" "/content/nanda-app/src/$f"
done
colab ls -s r001 /content/nanda-app/src
```

The scripts resolve every path from `__file__`, so this layout
(`nanda-app/src/*.py` beside `nanda-app/cot-proxy-tasks/`) is all they need.
`artifacts/` is created on the VM automatically.

---

## 3. Install dependencies — and do NOT force our lockfile

Colab ships its own CUDA-matched torch. Replacing it risks a broken CUDA build for
no benefit; torch version is not what determines our prompt format.

```bash
colab install -s r001 "transformers==5.16.1" accelerate scikit-learn hf_transfer
```

`transformers` **is** pinned, deliberately: the prompt string comes from
`tokenizer.apply_chat_template`, so the transformers version plus the model revision
jointly determine it (D004). Pinning to the version validated locally is what makes
the next step meaningful rather than decorative.

---

## 4. GATE: re-verify the prompt on the VM

This is the check that catches a silent activation-site mismatch. It must pass on
the VM, not just on the laptop.

```bash
colab exec -s r001 <<'PY'
import sys; sys.path.insert(0, "/content/nanda-app/src")
import task1_data as T
from transformers import AutoTokenizer
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__)
tok = AutoTokenizer.from_pretrained(T.MODEL_ID, revision=T.MODEL_REVISION)
p = T.load_build_thinking_prompt()(tok, "What is 2+2?", "Let me think. Two plus two")
print(repr(p))
assert p.count("<think>") == 1 and "</think>" not in p, "PROMPT FORMAT CHANGED - STOP"
assert p.endswith("Two plus two")
print("PROMPT OK")
PY
```

Expected, exactly:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nLet me think. Two plus two
```

If a second `<think>` or an empty `<think>\n\n</think>` block appears, **stop**. The
chat template changed; every activation cached under it would sit at the wrong site.
Fix the template/version, do not adjust the science.

---

## 5. Step A — real 32B smoke test

Downloads ~65 GB of weights (a few minutes on Colab's network; `hf_transfer` is
already installed).

```bash
colab exec -s r001 <<'PY'
import subprocess, os
env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
subprocess.run([ "python", "/content/nanda-app/src/extract_task1_activations.py",
                 "--smoke", "--run-id", "r001_qwen32b_smoke"], env=env, check=True)
PY
```

Must all hold:

- pinned revision `9216db5…` loads;
- `all 32.76B parameters CUDA-resident across ['cuda:0']`;
- free memory after load printed (expect roughly 10–14 GB free on an 80 GB card);
- all ten inputs report `prompt_endswith_prefix=True`;
- singleton vs padded: cosine ~1.0 at all five real depths;
- `hooked final depth + norm == base model's own last hidden state`: `max_abs_diff` ~0
  (a small non-zero value is fine in bf16 at this scale; the local run was exactly 0
  only because the 0.6B model is tiny — anything under ~1e-2 passes the built-in check);
- non-degeneracy passes;
- no OOM.

**If this fails, fix infrastructure only.** Do not touch the scientific design.

---

## 6. Steps B and C — memory stress

```bash
colab exec -s r001 <<'PY'
import subprocess
subprocess.run(["python", "/content/nanda-app/src/extract_task1_activations.py",
                "--stress", "--run-id", "r001_qwen32b_stress",
                "--token-budget", "16384", "--max-batch", "8"], check=True)
PY
```

- **B, representative batch** (~1.5–2.5k tokens at the configured budget): note peak
  allocated/reserved and free-after. The script warns below 3 GB free.
- **C, longest example** (~16.5k tokens, batch 1): the one that actually de-risks the
  run, because the train tail has no counterpart in eval.

Target **several GB free at peak**, not maximum utilization — one unlucky batch
should not OOM a 40-minute run.

If C OOMs despite `use_cache=False`: inspect the attention implementation
(`attn_implementation="sdpa"` / flash-attention availability). **Do not truncate and
do not drop the example** — that would break the "no truncation" decision and change
what R001 measures.

If either is tight, lower `--token-budget` (8192, then 4096). Never raise it above
what you tested.

---

## 7. Step D — launch

```bash
colab exec -s r001 <<'PY'
import subprocess, os
env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
subprocess.run(["python", "/content/nanda-app/src/extract_task1_activations.py",
                "--run-id", "r001_qwen32b",
                "--token-budget", "16384",   # or whatever B/C established
                "--max-batch", "8"], env=env, check=True)
PY
```

4,216 rows (4,000 train + 72 val + 86 test + 58 ood_test). Expect roughly 30–50 min
on an 80 GB card. Progress prints every 25 batches with an ETA.

**If the session dies**, the run is resumable: rerun the identical command and it
reports `resuming: N rows already extracted` and continues. The frozen sample hash
is recorded in `config.json`; confirm it still begins `d9eb713bcd366b6a`.

---

## 8. Fit the probes and bring the results home

```bash
colab exec -s r001 <<'PY'
import subprocess
subprocess.run(["python", "/content/nanda-app/src/fit_task1_probe.py",
                "--run-id", "r001_qwen32b"], check=True)
PY

# retrieve everything (activations are only ~215 MB)
colab exec -s r001 <<'PY'
import shutil
shutil.make_archive("/content/r001_qwen32b", "zip",
                    "/content/nanda-app/artifacts/runs/r001_qwen32b")
print("archived")
PY
colab download -s r001 /content/r001_qwen32b.zip ./r001_qwen32b.zip
colab download -s r001 /content/nanda-app/artifacts/tables/reproduction_layer_auroc.csv \
                       artifacts/tables/reproduction_layer_auroc.csv
colab download -s r001 /content/nanda-app/artifacts/figures/reproduction_layer_auroc.png \
                       artifacts/figures/reproduction_layer_auroc.png
colab stop -s r001
```

Unzip into `artifacts/runs/r001_qwen32b/`. The probe script can be re-run locally
from the cached activations at any time — no GPU needed for any later analysis.

---

## 9. After the numbers exist

1. Apply the `STATE.md` decision rule to **max OOD AUROC across the five depths**.
2. Write the `RESULTS.md` R001 entry: question, per-hypothesis predictions, setup,
   the AUROC table with question-clustered CIs, interpretation, **evidence against
   that interpretation**, decision, artifact paths.
3. Only then does the D007 embargo lift and the D008 paired analysis become available.

Do not inspect the cached output features or the opposite-label pairs before that
entry is written. Both were preregistered specifically so that they are tested after
the reproduction, not alongside it.

---

## Cost notes

- Compute units: an H100 hour is the dominant cost; budget ~1–1.5 h total including
  the 65 GB model download.
- The download does not persist across sessions. If you need a second run, either keep
  the session alive (`colab` runs a keep-alive daemon) or accept re-downloading.
- `colab drivemount` is available, but reading 65 GB of weights from Drive is usually
  slower than re-downloading from HF — not recommended.
