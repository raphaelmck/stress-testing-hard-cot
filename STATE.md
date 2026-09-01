# Current state

Last updated: 2026-09-01

## Completed

- Task 1 dataset audit complete (`notes/manual_dataset_audit.md`).
- H0/H1/H2 frozen in `PROJECT_BRIEF.md` before any activation result existed.
- Extraction, probe fitting and the frozen 4,000-row train sample are validated and
  committed (`5413fa1`, `d4b72a4`, `7367186`, `416fe40`).
  Sample is deterministic from seed 42: train-only 4,000 rows `9ae14f9e27a5f66d`;
  full 4,216-row worklist `c261306dde08c8b9` (this is what a run `config.json`
  records). The value `d9eb713bcd366b6a` recorded here previously was the 10-row
  `--smoke` worklist hash, not the frozen sample; corrected 2026-09-01, no result
  depended on it.
- **R001 complete on the Mila cluster** (A100-80GB, 53.7 min, job 10621220).
  See `RESULTS.md` R001 for the full table and caveats.
- **R002 complete** (no GPU): D007 embargo lifted, output-level baselines fit,
  comparator/inference rules recorded as D010. See `RESULTS.md` R002.
- **R003 complete** (no GPU): the preregistered D008 within-question ood control.
  The probe separates 15/16 questions -- and so does raw `token_length`, exactly.
  Recorded as D011; `ood_test` is now closed to further inspection.

## R001 headline

Max OOD AUROC **0.964 at depth 56** [0.899, 1.000], question-clustered.
Depth profile 8/24/40/56/64 -> 0.699 / 0.832 / 0.904 / 0.964 / 0.892 OOD.
Val-selected depth is 40 (OOD 0.904); the two are not separable given the CIs.
Reproduction bar (>= 0.85) cleared; >= 0.90 rule reached, so **probe optimisation
stops**. All pre-run gates passed (CUDA residency, activation site == model's own
`last_hidden_state`, no truncation, 14.4 GB free at the longest example).

## Key audit findings that constrain interpretation

- Zero `question_id` overlap between any pair of splits.
- Raw length is a train-only shortcut: AUROC 0.606 train, 0.510 test, 0.584 ood.
- Empirical purity differs by split: val/test 40–50/50, ood_test 45–50/50.
  val is both the noisiest-label split and the selection split.
- `yes_count + no_count < total_resamples` in 139/216 eval rows, so `yes_count/50`
  is not a calibrated termination probability.
- 16 of 32 ood_test questions carry both labels (D008 paired control, now available).

## Current beliefs

Informal research beliefs, not statistical probabilities. R003 moved these sharply.

- H0 immediate-output: 20% (down from 45%) -- output features are at chance on the
  training distribution, and within question the probe beats `think_logprob`.
- H1 broader termination-ready latent state: 35% (down from 40%)
- H2 generic depth-into-trace / proxy structure: 45% (up from 15%) -- within
  question, prefix length alone matches the probe.

## Where the evidence stands

| claim | status |
|---|---|
| the phenomenon reproduces (OOD AUROC 0.90+) | R001, solid |
| it is reducible to immediate `</think>` propensity | R002+R003, largely no |
| it is more than between-question topic structure | R003, yes (15/16, p=0.0005) |
| it is more than prefix length | **untested; R003 found no advantage over length** |

## R003 headline

16 ood_test questions carry both labels. Macro within-question concordance:
activation depth-40 **0.938** [0.812, 1.000]; `think_logprob` 0.844 [0.656, 1.000];
**`token_length` 0.938** [0.844, 1.000], positive on 16/16 questions.
Paired deltas: activation − think +0.094 [+0.000, +0.250], P(Δ>0)=0.88;
activation − length **+0.000 [−0.188, +0.125]**, P(Δ>0)=0.44.

## Next experiment (exactly one)

**R004 — the length control, on `val` and `test`.** `ood_test` is closed (D011).
val has 30 questions and test 22, with more multi-row questions than ood's mostly
1v1 pairs, so this is where the comparison has power. No GPU, no refitting of the
activation probe.

1. Within-question macro concordance for depth-40 activation, `think_logprob` and
   `token_length` on val and on test, with paired question bootstraps.
2. The conditional test: does the probe rank YES above NO on pairs where
   `token_length` does **not** (or where the two prefixes are close in length)?
   Report the length-discordant subset the way R003 reported the think-discordant one.
3. Report the length-residualized probe score as the secondary analysis.

Decision rule:

- probe clearly beats length within question on val and test -> H2's cheap
  explanation is insufficient; residualization against the output features (the
  originally planned experiment) becomes R005 to separate H1 from H2.
- probe ties length again with real power -> the honest headline of this project is
  that a high OOD termination probe is matched, within question, by prefix length,
  and the writeup should say exactly that.
- probe loses to length -> same conclusion, stated more strongly.

Do not reopen `ood_test`. Do not fit new activation probes at other depths.

## Current blocker

None. R004 needs no GPU: frozen R001 probe scores, cached D007 features, and the
released `token_length` field.
