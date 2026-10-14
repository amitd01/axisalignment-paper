# Pre-registration — is the base->SFT residual position-specific, or merely unmodelled?

Written 2026-10-13 23:13:07 IST, before code or results. CPU analysis on existing arm-B arrays
(base-written continuations), OLMo-1B / OLMo-7B / Tulu-8B, deltas base->SFT and SFT->DPO.

## Claim under test (paper owner)
SFT's entropy loss comes from a context-specific residual that is different at every position, not from
structure the linear fit (a + beta*ref + gamma*g_-p) merely failed to model.

## Extended per-position decomposition over the plausible set
d = a + beta*ref                      (linear typicality)
      + f(ref)                        (NONLINEAR typicality: piecewise-linear in ref, knots at the 20/40/60/80th
                                       percentiles of ref within the position's plausible set, linear part removed)
      + gamma*g_-p + sum_k c_k u_k    (SHARED: LOO mean shift + top-K=8 principal directions of the shift,
                                       CROSS-FITTED: estimated on the other half of prompts, odd/even split)
      + eps                           (POSITION-SPECIFIC remainder)
      + outside                       (tokens off the plausible set)
Players for exact Shapley entropy attribution: {linear typ, nonlinear typ, shared, eps, outside}.

## Reading
- Claim SUPPORTED for a model if eps (position-specific remainder) carries > 50% of base->SFT entropy loss,
  bootstrap CI over prompts above 50%.
- Claim FAILS if nonlinear-typ or shared carries the majority instead -> SFT's loss is structured (truncation-like
  or low-rank shared), and the §4 explanation "no single operator could reach it" does not follow.
- In between (eps share CI straddling 50%) -> reported as unresolved.
SFT->DPO reported alongside as a comparison, no prediction. Any outcome reported.

---
## Appended note — 2026-10-13 23:13 IST, before any code or result existed

Text above unchanged (sha256 in PREREG-RESIDUAL.sha256 covers the bytes above this note).
The paper owner withdrew the "position-specific" claim before the run: by construction the LOO regressor absorbs
only the cross-prompt component, so the remainder can only be called "unexplained by either axis". The eps-share
reading above is therefore no longer a test of anything claimed, and is reported descriptively only.
Remaining question (the reason to run): does a NONLINEAR function of ref log-prob carry the majority of base->SFT
entropy loss? If yes, the claim "no decoding-time operator built on the typicality axis can reach SFT's loss"
is too strong (a nonlinear, truncation-like reference-based operator could). If no, that claim stands for the
nonlinear case too. Decision: nonlinear-typ share CI above 50% = claim too strong; CI below 50% = claim stands.
