# Pre-registration — 135M decontamination gamma sweep

Written 2026-10-13 23:28:21 IST, before any new 135M run. Already seen (published, pilot-135m/results_exp3_scored.json,
reinterpreted in logs/frontier_135m.txt): at matched Jaccard, decontamination's median GPT-2 ppl / temperature-frontier ppl
= 1.30 (gamma 0.2), 0.98 (0.5), 1.53 (0.8). The published "gamma=0.5 -> ppl 19,956" was never compared at matched diversity.

## Hypothesis (paper owner)
The DPO-stage diversity cost is a small temperature-like sharpening (beta ~ 0.1 at 7-8B); gamma = 0.5 overcorrects.
A small gamma (~beta) should behave like mild temperature rather than degenerate.

## Design (faithful port of pilot-135m/gen_one.py + score.py; only paths and the condition list change)
SmolLM2-135M / -Instruct, fp32, CPU; 5 prompts x 8 samples x 32 tokens; top-p 0.95; seed 20260803 (+1000*prompt).
Conditions: temperature T in {1.0, 1.1, 1.2, 1.3, 1.6, 2.0}; decontamination gamma in {0.05, 0.1, 0.2, 0.3, 0.5}.
Step 0: beta for the 135M pair (prob-weighted fit of shift on base log-prob over the plausible set, median over
prompt-final + base-continuation positions), reported first, before generation results are read.
Reproduction check: gamma=0.5 and T=1.0/1.3 vs published (ppl 19,956 / 65.0 / 476.0; Jaccard 0.0022 / 0.0781 / 0.0318).
Same-order agreement expected; exact match not expected (hardware).

## Statistic
ratio(gamma) = median GPT-2 ppl at gamma / temperature-frontier ppl at the same mean Jaccard (log-log interpolation
between the NEW temperature points). Uncertainty: bootstrap over the 5 prompts (resample prompts, recompute both
curves), 20k draws.

## Reading
- gamma in {0.05, 0.1}: "temperature-like" if ratio CI lies within [0.67, 1.5]; "degenerate" if ratio > 2 (CI lower bound > 2).
  Hypothesis supported if both small gammas are temperature-like; FALSIFIED if either is degenerate.
- If ratio(0.5) is again ~1, the paper's §4 decontamination failure is reframed as overshoot along the temperature
  frontier, not off-frontier harm. Chat-template token leakage claim is not tested here (qualitative).
- With 5 prompts the CIs will be wide; an inconclusive outcome is reported as such.
