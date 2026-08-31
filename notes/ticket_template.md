# Agent ticket template

Copy, fill in, paste as the whole prompt. One ticket = one bounded job.
Do not ask an agent to implement infrastructure, run experiments, reinterpret the project,
and decide what comes next in the same ticket.

```text
Read PROJECT_BRIEF.md, STATE.md, RESULTS.md and DECISIONS.md first.

Task:
<one job, stated concretely>

Acceptance criteria:
- <verifiable condition>
- <verifiable condition>
- does NOT <the adjacent work this ticket must not do>

When done:
- <artifacts to write>
- update STATE.md (propose, do not assume)
- add a RESULTS.md entry ONLY if an empirical scientific result was produced
- report: files changed, validation actually performed, quantitative results,
  scientific caveats, exactly one recommended next step
```

## Worked example — the current prerequisite ticket

```text
Read PROJECT_BRIEF.md, STATE.md, RESULTS.md and DECISIONS.md first.

Task:
Implement and validate last-real-prefix-token hidden-state extraction for Qwen3-32B
at 5 layers spanning the network.

Acceptance criteria:
- input is built as prompt_text + Qwen chat/thinking template + cot_prefix;
- works unbatched and with padded batches, and asserts the padded-batch activation
  equals the unbatched activation for the same example;
- caches fp16 activations plus a metadata sidecar (question_id, rollout_idx, prefix_idx,
  label, split, layer, token index used);
- includes a 10-example smoke test that can run without the full dataset;
- records the HF model revision in the config;
- does NOT train probes and does NOT run full extraction.

When done:
- commit code;
- update STATE.md;
- add no RESULTS entry (this produces no scientific result);
- report changed files, validation performed, and any scientific caveats.
```
