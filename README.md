# Alignment's Diversity Cost Is Carried by a Small Part of the Shift

Stage-resolved measurement of diversity collapse across four post-training
pipelines. Code, raw results and pre-registrations for the paper of the same
name.

**Author:** Amit Das · amitdas@gmail.com
**Paper:** [`paper/paper.md`](paper/paper.md) (source of record),
[`paper/main.tex`](paper/main.tex) (arXiv submission source)
**Licence:** CC BY 4.0 — see [`LICENSE`](LICENSE)

---

## What this measures

Alignment reduces output diversity. The loss is usually attributed to RL on human
preferences, with *typicality bias* — a reweighting toward what the base model
already found likely — offered as the mechanism. This work measures both against
four pipelines that publish every intermediate checkpoint, so each stage can be
isolated:

| family | stages measured |
|---|---|
| `allenai/OLMo-2-0425-1B` | base → SFT → DPO → Instruct |
| `allenai/OLMo-2-1124-7B` | base → SFT → DPO → Instruct |
| `allenai/OLMo-2-1124-13B` | base → SFT → DPO → Instruct |
| `allenai/Llama-3.1-Tulu-3-8B` | base → SFT → DPO → Instruct (RLVR) |

Throughout §§2–5 of the paper, *diversity* means next-token entropy at the scored
positions. §6 uses mean pairwise Jaccard, with median GPT-2 perplexity as an
independent fluency judge.

Three headline results:

1. **Composition and consequence come apart.** The typicality-aligned component is
   a minority of each stage's shift (≤2.2% of variance for instruction tuning,
   ≤7.1% for the preference stage, unweighted; ~30% at most weighted) yet an exact
   counterfactual Shapley attribution assigns it nearly all of the entropy loss
   (87–139% instruction tuning, 102–291% preference stage). The part of the shift
   that removes diversity is not the part that is large.
2. **The larger component is a shared direction**, not typicality: ρ_reg exceeds ψ
   across the plausible set at every stage of all four models, against null
   controls at +0.008 and +0.004.
   But neither axis explains most of the shift — squared, the shared direction is
   roughly 1–31% of it.
3. **Where you measure changes the answer.** Tülu-3-8B's preference stage accounts
   for 29.2% [22.4, 35.0] of total entropy loss at a chat-template position and
   52.8% [50.5, 55.4] over generated continuations — same checkpoints, same
   prompts. Prompt-final scoring, closest to common practice, gives intervals
   4.5–6× wider.

### The four measurement conventions

Every stage-resolved number in the paper is reported under a named convention,
because the choice moves the answer by tens of points:

| convention | scored positions |
|---|---|
| raw prompt-final | one position per prompt, no chat template |
| chat-templated | one position per prompt, chat template applied |
| **continuation, base-written** (primary) | ~190 positions per prompt, continuations written by the base model |
| continuation, final-model-written | ~190 positions per prompt, continuations written by the final aligned model |

The primary convention is the base-written continuation. Results under all four
are in `results/`.

---

## Layout

```
code/                    analysis and generation
  dump_logprobs_mps.py   per-checkpoint log-prob dumps (Apple Silicon / MPS)
  dump_logprobs.py       same, CUDA/CPU
  analyse_psi_family.py  stage-resolved psi and entropy, bootstrap over prompts
  analyse_psi.py         single-family version
  register_v2.py         shared-shift geometry (rho_reg), both arms, null controls
  decomp.py              counterfactual Shapley entropy decomposition
  decomp_resid.py        non-linear typicality, cross-fitted shared directions
  check_tokenizers.py    token-ID-level tokeniser identity verification
  gen_scale.py           generation sweep; mode `ctrl` is the shuffled-delta control
  score_scale.py         scores the sweep
  grind.py               sweep driver
  matched.py, matched6.py  matched-diversity scoring
  exp1d_misspec.py       the withdrawn psi* threshold (see "What did not work")
  exp1e_peakedness.py    the peakedness probe that withdrew it
  prompts.py             the prompt set
  pilot135m_port/        135M gamma sweep and beta measurement
  dumps/135m/            135M sweep outputs: gen_*.json, beta135m.json, scored sweep
                         (the 7B/13B/Tulu dumps are ~53GB and are not shipped;
                          regenerate them with dump_logprobs_mps.py)
logs/                    run scripts and the evaluation outputs behind the numbers
  frontier_135m.txt      the 135M diversity/fluency frontier (0.98x, 0.17-0.37)
  inline_checks.py/.txt  reproduces every figure that was computed inline rather
                         than by a script: the sampled beta and its sign rate, the
                         400-position R^2 table, the CI widths, the paired totals,
                         the branching ratios and the plausible-set sizes
  score135m.txt          scored 135M sweep
  beta135m via code/dumps/135m/beta135m.json   the beta = +0.081 measurement
pilot-135m/              the original 135M pilot — gamma sweep, psi measurement,
                         gen_ctrl.py (135M direction control), exp1d_misspec.py
                         (the withdrawn psi* threshold), and stats.py, which is
                         where the 3.00x figure comes from
  lineage_eval.txt       the cross-lineage statistic (-0.194 both arms)
  prereg13b_eval.py/.txt the pre-registered 13B evaluation (+15.6 [12.7, 18.6])
  r2_full.py/.txt        R^2 point estimates and intervals, unweighted and weighted
  offset_check.py        the whose-text offset check
  *_analysis.txt         per-run analysis output
  *.log, *.sh            raw run logs and the shell drivers that produced them
results/
  PREREG-*.md            six pre-registrations, each hashed before its data existed
  PREREG-*.sha256        the hashes
  results_*.json         raw per-run output
  verify_prereg.py       checks all six hashes, including the two with notes
  TASK1-REPORT.md        verification reports
  TASK2-REPORT.md
paper/
  paper.md               source of record
  main.tex               arXiv submission source
```

Table regeneration: `logs/make_task2_tables.py` produces the numbers behind the
tables in §2.3, §3.1, §4.2 and §5, and `logs/r2_full.py` the R² table. An earlier
`make_tables.py` was removed — it read `results_psi_1b.json` and built the
superseded 1B-era tables, so keeping it would have been worse than keeping nothing.

## Reproducing

```bash
cd code
pip install -r requirements.txt
python dump_logprobs_mps.py      # or dump_logprobs.py off Apple Silicon
python analyse_psi_family.py
python register_v2.py
python decomp.py
python decomp_resid.py
```

Checkpoints download from the Hugging Face Hub on first use; the 13B family is
the large one. `check_tokenizers.py` verifies at token-ID level that every
checkpoint in a family shares a tokeniser — stage deltas are meaningless
otherwise, and it is worth running first.

Seeds: 20261012 (1B), 20260803 (135M). N = 6 samples, 32 new tokens, top-p 0.95
held constant. Fixed seeds do not guarantee bit-identical output across hardware;
batch-invariant kernels were not attempted, so expect small numeric drift rather
than exact reproduction.

## Pre-registrations

Six, each hashed before its data existed: the 13B scale test, the register test,
the lineage test, the entropy decomposition, the residual test and the 135M γ
sweep.

```bash
cd results && python3 verify_prereg.py
```

```
PREREG-135M-GAMMA  OK    whole file
PREREG-13B         OK    original text only: first 1826 of 3245 bytes (56%), note appended after hashing
PREREG-DECOMP      OK    whole file
PREREG-LINEAGE     OK    whole file
PREREG-REGISTER    OK    whole file
PREREG-RESIDUAL    OK    original text only: first 1819 of 2765 bytes (66%), note appended after hashing

All 6 pre-registrations verify.
```

Two of the six — `PREREG-13B` and `PREREG-RESIDUAL` — carry a dated note appended
*after* the hash was taken, so hashing the whole file fails for those by design;
the hash covers the original text only, and `verify_prereg.py` finds that
boundary and checks it. Both notes record a decision made before the run they
govern produced anything, and both say what changed and why:

- **`PREREG-13B`** moved the comparison basis from raw prompt-final positions to
  base-written continuations, because the register run had meanwhile shown the
  raw prompt-final share to be protocol-dependent (−0.6% vs 25.5% vs 27.6% on the
  same checkpoints). The original thresholds are still computed and reported,
  labelled as the artefact arm. Appended before any 13B dump existed.
- **`PREREG-RESIDUAL`** withdrew a "position-specific" claim before the run, on
  the grounds that the leave-one-out regressor absorbs only the cross-prompt
  component by construction, so the remainder can only be called *unexplained by
  either axis*. Appended before any code or result existed.

Read the notes rather than taking this summary for them. Whether an appended note
is as good as an untouched pre-registration is a judgement for the reader; the
point of publishing the boundary is that the judgement is available to make.

## What did not work

`exp1e_peakedness.py` is here because the paper reports it as a failure. An
earlier draft claimed a threshold in the peakedness of the aligned distribution
above which a particular intervention helps. That threshold is not a constant: it
moves across the full 0.000–1.000 range with the matched-diversity target alone,
and the model it came from sits far off the measured regime (47× collapse against
the 1.5–2.2× actually observed). The claim was withdrawn. §7 of the paper lists
this and three other overturned conclusions.

## Citing

```bibtex
@misc{das2026diversitycost,
  title        = {Alignment's Diversity Cost Is Carried by a Small Part of the
                  Shift: Stage-resolved measurement across four post-training
                  pipelines},
  author       = {Das, Amit},
  year         = {2026},
  howpublished = {\url{https://github.com/amitd01/axisalignment-paper}}
}
```

The arXiv identifier will replace `howpublished` once the paper is posted.
