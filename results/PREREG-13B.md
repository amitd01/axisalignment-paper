# Pre-registration — OLMo-2-1124-13B stage attribution

Written 2026-10-13 20:20:28 IST, before any 13B checkpoint was downloaded or dumped
(code/dumps contains no olmo13b_* files at time of writing).

Question: does the DPO share of total entropy loss keep rising with scale within one post-training
release (OLMo-2-1124: 7B -> 13B)? Removes the 0425-vs-1124 release confound of the 1B->7B pair.

Reference: OLMo-2-1124-7B DPO share = 25.7% [19.9, 31.9] (results/results_psi_olmo7b.json).

## Primary decision rule (verbatim from the request; applied to the 13B point estimate)
- DPO share > 25.7%          -> scale trend continues; 1B -> 7B -> 13B monotone
- DPO share ~ 25.7%          -> 7B -> 13B flat; trend is 1B -> 7B only; 1B is the outlier, not the bottom end
  (operationalised: 20.0% <= share <= 25.7%)
- DPO share < 20%            -> NON-MONOTONE; the two-point trend behind section 2.3 is suspect. Report loudly.

## Secondary qualifier (decided now, computed after)
Paired bootstrap over the same 36 prompts (20k resamples, seed 0) of
  share_13B(DPO) - share_7B(DPO), each share = mean drop / mean total on the resampled prompt set.
If its 95% CI includes 0, the result is labelled "statistically indistinguishable from flat" alongside
whichever primary branch the point estimate falls in. Does not override branch 3 being reported loudly.

## Fixed protocol (unchanged scripts, no new measurement code)
dump_logprobs_mps.py / analyse_psi_family.py / check_tokenizers.py; 36 prompts (prompts.FLAT), raw text,
no chat template, prompt-final positions, bf16 load with per-parameter assertion, 20k bootstrap over prompts.
Tags: olmo13b_{base,sft,dpo,inst}.

## Also reported, no prediction made
SFT and RLVR shares; psi on all deltas (preference delta, all five regions + weighted); special-token mass by stage.

---

## Appended note — 2026-10-13 21:21 IST, before any 13B dump existed (code/dumps had 0 olmo13b_* files)

The text above is unchanged. Its sha256 (results/PREREG-13B.sha256) covers the first 1826 bytes of this file.

**Why the comparison basis moved.** The register-v2 run (results/PREREG-REGISTER.md) showed that the DPO share
measured at raw-prompt-final positions is protocol-dependent: OLMo-2-0425-1B gives -0.6% there, but 25.5%
chat-templated and 27.6% at continuation positions. The thresholds above were written against the raw-prompt-final
quantity, so judging 13B against them would test the artefact rather than the trend.

**What it moved to.** Primary: 13B DPO share at arm-B continuation positions (base-written continuations, the
pre-registered arm-B source), compared with OLMo-2-1124-7B arm B = 33.3% [30.9, 35.8]. The same three-branch reading
applies, re-anchored: above 33.3% = trend continues; roughly equal = flat 7B->13B; clearly below = non-monotone (report
loudly). The paired prompt-level bootstrap of share_13B - share_7B decides "roughly equal", as specified above.
Secondary: arm B with final-model-written continuations (7B = 23.9% [22.0, 25.9]) and chat-templated arm A
(7B = 26.1% [21.9, 30.5]).

**Kept, demoted.** The raw-prompt-final number is still computed and reported against the original thresholds
(> 25.7 / ~25.7 / < 20), labelled as the artefact arm, not the headline.
