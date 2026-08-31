# Task 1 dataset audit

Source: `cot-proxy-tasks/datasets/1/`, upstream commit `4482324`.

## Automated inventory (2026-08-31, verified)

| split | prompts | rollout records | no / yes | label construction |
|---|---|---|---|---|
| train | 796 | 46,160 | 23,080 / 23,080 | proxy: `distance_from_end` |
| val | 30 | 72 | 36 / 36 | behavioral: resampled continuations |
| test | 22 | 86 | 43 / 43 | behavioral |
| ood_test | 32 | 58 | 29 / 29 | behavioral |

All four splits are exactly balanced.

### Record schemas

`prompts/<split>/<question_id>.json`:
`question_id`, `prompt_text`

`qwen-3-32b/train/<question_id>_rollout_NNN_prefix_N.json`:
`question_id`, `rollout_idx`, `prefix_idx`, `label`, `cot_prefix`,
`distance_from_end`, `prefix_words`, `total_words`

`qwen-3-32b/{val,test,ood_test}/...json`:
`question_id`, `rollout_idx`, `prefix_idx`, `label`, `cot_prefix`,
`token_length`, `yes_count`, `no_count`, `total_resamples`, `mean_yes_position`

The train and eval schemas share no label-construction field. Confirmed: the training target
is a word-distance proxy; the evaluation target is a resampling-behavioral measurement.

## Manual inspection — TODO

Not yet done. Required before R001 is interpreted:

- [ ] Read ~10 positive and ~10 negative `cot_prefix` values from `ood_test` end to end.
      Are negatives obviously mid-reasoning, or just long?
- [ ] Check the empirical `distance_from_end` values present in `train` against the brief's
      claim of {25, 35, 45, 55} words for positives and 300+ for negatives.
- [ ] Check `prefix_words` / `token_length` distributions by label in every split.
      If length alone separates the classes, that is an H2 result and a length-matched
      control is mandatory before any activation claim.
- [ ] Check question_id overlap between splits — confirm `ood_test` questions are genuinely
      held out from `train`.
- [ ] Check how many distinct questions the 58 `ood_test` records come from (32 prompts),
      i.e. how correlated the evaluation examples are. Bootstrap should resample by question,
      not by record, if records cluster.
- [ ] Inspect the `mean_yes_position` distribution — it may give a continuous target for the
      "does probe score track termination probability" experiment.
