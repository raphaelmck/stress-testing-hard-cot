# Literature notes

Keep claims sourced. Distinguish "the paper reports" from "we verified".

## Hard-CoT termination probes

- Reported: linear probes on Qwen3-32B hidden activations predict imminent `</think>`
  with >0.90 OOD AUROC on Task 1 of `Centrattic/cot-proxy-tasks`.
- Not yet verified here. Reproduction target is R001.
- TODO: record exact citation, layer(s) reported, probe type, N_train, and whether the
  reported OOD number used the same `ood_test` split shipped in the repo.

## Causal analysis of Qwen3 termination

- Reported: the stopping mechanism is high-dimensional and concentrated in late-model
  computation; no single residual-stream direction constitutes the mechanism.
- TODO: record exact citation, which layers, and what intervention was used.
- This is the source of the project's central puzzle. Get the claim precise before relying
  on it — "not a single direction" and "not linearly decodable" are different claims.

## Upstream repo

- `Centrattic/cot-proxy-tasks` @ `4482324` ("tasks 1 and 2", 2026-05-04).
- README documents 9 tasks; Task 1 is reasoning termination.
- Generation code for Task 1 labels: `cot-proxy-tasks/src/tasks/reasoning_termination/`.
  TODO: read it — it is the ground truth for how both label types were built.
