# Paper controls

A register of every counter-argument this paper has faced, and every modelling choice
that was revised along the way. Each row says what was claimed or objected to, what we
did about it, and where the evidence lives. The aim is that a reader with an objection
can check here first whether it has already been raised, answered, conceded or left open.

Paper version: draft v1.0 (in revision). Section numbers refer to `paper/paper.md`.

**Status key**

| status | meaning |
|---|---|
| **Changed** | the objection was right; the paper was changed |
| **Answered** | the objection was checked; the paper already handled it, or the check came back in the paper's favour |
| **Rejected** | the suggestion was checked and found wrong or out of scope; the reason is given |
| **Open** | the objection stands and is not yet addressed; listed so nobody mistakes silence for an answer |

Objections come from internal and simulated review rounds. Following `CLAUDE.md`,
the reviews themselves stay out of the public record. This file restates each
argument in our own words and attributes none of them.

---

## Part A — Objections and responses

### A1. The central claim: a small component carries most of the loss

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A1.1 | The contrast between variance share and entropy share may depend on how the component is specified: weighting, functional form, plausible-set threshold, how the shared direction is estimated. | Ran a 12-cell grid: weighted/unweighted × p_ref > 10⁻⁵/10⁻⁶/10⁻⁷ × shared direction from other prompts or other task families. **Weighted:** typicality share 77.3–156.8% (instruction tuning) and 89.6–345.5% (preference), against a variance share of 14.4–34.2%. **Unweighted:** −104.8–56.1% and 35.1–178.4%. The paper now says plainly that "nearly all" holds only under the weighted fit. | Changed | §5.1 Robustness; `code/robustness.py`; `logs/robustness_s4.txt` |
| A1.2 | The headline sentence pairs two numbers with different standing. | The entropy shares are pre-registered; the variance shares are exploratory. The paper now says so where the sentence appears. | Changed | §5.1 |
| A1.3 | "At most about 30% of the shift" is understated. | The grid's median per-position weighted R² reaches 34.2% (Tülu, preference stage). Wording changed to "at most about a third" in the abstract, §5.1 and §10. | Changed | `logs/robustness_s4.txt` |
| A1.4 | The preference stage does not lose "all" of its entropy through this component in every specification. | At p_ref > 10⁻⁵ the 13B and Tülu fall to 89.6% and 90.8%, because 16.5% and 18.1% of the loss moves to tokens outside the plausible set. Text says "nearly all or all". | Changed | §5.1 |
| A1.5 | Why prefer the weighted fit at all? | Entropy is −Σ p log p, so each token's contribution is already weighted by its probability. A fit weighted the same way is judged on the tokens that move entropy; the unweighted fit treats a token at p = 10⁻⁶ like one at p = 0.3. | Changed (justification added) | §5.1 |
| A1.6 | The conclusion claimed that under an unweighted fit the component "at most 7%" of the shift carries nearly all of the loss. | False on the grid. Removed; §10 now says the unweighted share is smaller and unstable in sign at the 1B's instruction tuning. | Changed | §10 |
| A1.7 | A few task families could be driving the result. | Within each of the six families the pre-registered share is 81.7–167.4% (instruction tuning) and 99.0–495.7% (preference). Leaving one family out moves it by at most 4.4 points, except at the 1B preference stage (components nearly cancel). | Answered | §5.1; `logs/robustness_s4.txt` |
| A1.8 | Are the typicality and shared components separately identifiable? | At the 1B preference stage the two nearly cancel (typicality 290.5%, shared −236.2%). The paper reads only their signs there. A direct per-position collinearity diagnostic has not been run. | Open | §5.1; this file, Part C |

### A2. Attribution is not causation

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A2.1 | §5 said each stage's entropy change is attributed "causally". A Shapley split of counterfactual distributions is an attribution under a chosen decomposition, not causal identification of what training did. | "Causally" removed. The text now says Shapley efficiency allocates interactions rather than removing them, and that this is not causal identification. | Changed | §5 |
| A2.2 | Shares above 100% are not explained. | Worked example added. At the 1B preference stage (0.118 nats), typicality removes 0.342, the shared part restores 0.278, the remainder removes 0.054 and off-set tokens 0.000; the four sum to the loss. | Changed | §5.1 |
| A2.3 | An offered intuition: the shared part *always* restores the entropy typicality removes. | Wrong in general. At the 1B's instruction tuning the offset comes from the remainder (−30.6%); the shared share there is 2.1%. Not adopted. | Rejected | §5.1 |
| A2.4 | Do the Shapley values actually sum to the measured change? | Yes. The maximum Shapley sum error is recorded for every run and is at most about 1e-15. | Answered | `results/results_decomp_*.json` (`max_shapley_error`); `logs/robustness_s4.txt` |

### A3. Terminology outruns the measurement

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A3.1 | "Typicality" names both the fitted object and the theory that human preferences favour the conventional. | The fitted term β·log p_ref is now defined as a reference-probability-aligned component; reading it as typicality bias is called an interpretation. | Changed | §5; limitation 4 |
| A3.2 | "The larger of the two components is a shared direction" does not hold under every measure. | True. Under the weighted measure typicality's R² reaches 34.2% against the shared direction's at most about 31%. Scoped to "by correlation". §4.5 already reported the reversal in the weighted region. | Changed | Abstract; §4.5 |
| A3.3 | The shared direction was called "context-independent". | ρ_reg tops out at 0.558, so squared it is at most about 0.31 of the shift. Renamed a partially shared direction. | Changed (earlier round) | commit `161a6c6` |
| A3.4 | "The shared shift dominates the plausible set." | Overclaim. Now "exceeds typicality across the plausible set". | Changed (earlier round) | §4.2 heading; commit `a267f5d` |

### A4. Statistics and the unit of inference

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A4.1 | ~6,800 positions per model are not 6,800 independent observations. | Agreed. The unit of inference is the prompt (36 prompts, six task families), and the bootstrap resamples prompts. Stated in limitation 1; the grid uses a family-stratified bootstrap. | Changed | §9 limitation 1; `code/robustness.py` |
| A4.2 | The fitted components should be refitted inside every bootstrap replicate. | They are fitted at every position, so resampling prompts needs no refit. The shared direction g is computed once on the full sample and held fixed, which the paper now states. | Answered | §5; `code/decomp.py` |
| A4.3 | The shared direction could leak information from the evaluated prompt's own task family. | Tested: estimating g only from other families changes no share by more than 3.4 points. | Answered | §5.1; `logs/robustness_s4.txt` |
| A4.4 | Some of the robustness numbers come from a subsampled run. | Yes: the grid scored every fourth position. At the pre-registered cell it reproduces the full-position shares within 2.1 points at the 7B, Tülu and 13B, and runs 8.6 and 31.2 points high at the 1B. The full-position run is pending. | Open | §5.1; `logs/robustness_s4.txt` (stride check) |

### A5. The measurement convention

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A5.1 | Different scoring positions measure different conditional distributions, so differences between conventions are not simply a precision effect. | Agreed and stated: the conventions differ in estimand as well as precision. Absolute drops in nats are reported beside every share. | Answered | §2.2–§2.3; §3.1 |
| A5.2 | Separate sample-count effects from context effects by subsampling continuation positions to match prompt-final counts. | Not run. | Open | Part C |
| A5.3 | The primary convention was chosen after some results were seen. | True, and dated: base-written continuations were adopted for stage shares in a note appended to the 13B pre-registration, after the 1B/7B/8B shares and before any 13B data. Listed in the audit table. | Answered | §2.2; Appendix A; `results/PREREG-13B.md` |
| A5.4 | "Prompt-final makes the 7B and Tülu look different." | Wrong. Prompt-final cannot distinguish them; it is a precision point, not a difference. | Changed (earlier round) | `results/TASK2-REPORT.md` §8 |

### A6. Generality and scope

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A6.1 | Four pipelines are not four independent recipes. | Agreed. Two base families with related instruction mixtures; the OLMo 7B and 13B share one release, so their agreement is a within-release scale comparison. | Changed | §9 limitation 1 |
| A6.2 | The OLMo 1B and 7B come from different post-training releases, so the 1B→7B change is not purely scale. | Agreed; this is why the within-release 13B was run. | Answered | §2.3; `LEARNINGS.md` L-01 |
| A6.3 | Nothing above 13B. | Agreed and stated. Frontier scale is out of reach for this study's compute. | Open (scope) | §9 limitation 8 |
| A6.4 | "Instruction tuning does all of it" as a general claim. | Refuted by the second pipeline and withdrawn. Tülu-8B and the 13B invert "SFT is the larger stage" under continuation scoring. | Changed (earlier round) | §7; `results/TASK2-REPORT.md` §8 |

### A7. The shared direction and the lineage test

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A7.1 | The shared direction could be generic token-frequency or formatting bias rather than a shift in register. | Partly addressed. The held-out-family direction gives the same shares (A4.3), and null controls on identical weights give +0.008 and +0.004. A comparison against frequency-based or formatting-based directions has not been run. | Open (partly) | §4.2; §5.1 |
| A7.2 | Is the punctuation promotion caused by the preference data or by the DPO loss? | Not tested. The paper reports the tokens and claims no cause. | Open | §4.3 |
| A7.3 | The lineage result could come from shared training data. | Agreed and stated: the test rules out checkpoint ancestry, not data. | Answered | §9 limitation 2 |
| A7.4 | The lineage result (register-conditional concentration) and the shared-shift geometry are different quantities and should not be read as one claim. | Kept separate. The abstract calls the lineage quantity "a second, distinct quantity". | Answered | Abstract; §4.4 |
| A7.5 | Chat-templated scoring could show a shared shift that isn't real. | Yes: every prompt ends in the same template suffix, so even a noise delta looks shared (null ρ_reg 0.149 at the 1B). That arm is not used as register evidence. | Answered (earlier round) | `results/TASK2-REPORT.md` §5 |

### A8. Decoding-time interventions (§6)

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A8.1 | Five prompts at 135M and twelve at 1B cannot support claims about methods. | Agreed. §6 is labelled exploratory and framed as counterexamples to earlier predictions. | Changed | §6 |
| A8.2 | A single GPT-2 judge (124M) measures fluency, rewards predictable text and cannot judge quality. | Agreed and stated; human evaluation is the limitation we most want addressed. A modern LLM judge at matched diversity is a reasonable next step but has not been run. | Open | §9 limitation 5 |
| A8.3 | Explain the 1B λ null mechanistically: a "more non-linear feature manifold" at 1B pushes extrapolation off well-formed text. | Not adopted. It cannot be tested as stated, and §6.3 already explains why the 7–8B account does not transfer. The 1B result stays unexplained. | Rejected | §6.3 |
| A8.4 | Describe the 1B result as "degenerating". | Wrong. It is a null (1.04×, CI [0.73, 1.50]). What degenerates at the 1B is the shuffled-direction control: both of its arms do so by matched diversity, which makes the control uninformative there. | Rejected | §6.1 |
| A8.5 | Diagnose the 1B null by comparing logit-norm ratios at 135M and 1B. | Cheap and reasonable; not run. | Open | Part C |
| A8.6 | Reference subtraction was reported as catastrophic. | That figure compared it against T = 1.0, not at matched diversity. At matched diversity it is 0.98× the temperature frontier. Corrected. | Changed (earlier round) | §6.2; §7 |

### A9. Pre-registration and reproducibility

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A9.1 | A hash proves the text is unchanged, not that it was written before the relevant results were seen. | Agreed. Appendix A has an audit table for each pre-registration: when it was written, what had already been seen, any appended note, the outcome, and whether it stands as confirmatory or exploratory. | Changed | Appendix A |
| A9.2 | PREREG-DECOMP's primary hypothesis (preference loss head-driven, head > 50%) failed, and the paper never said so. | Now stated: the head carries 15–23%, so the hypothesis is not supported in any model. Its secondary condition (typicality > shared) holds in all four. | Changed | §5.2; `results/TASK2-REPORT.md` §6 |
| A9.3 | Cited numbers must be reproducible from released files. | Every number cited in the paper comes from a released script and log (`logs/inline_checks.py`, `logs/r2_full.py`, `logs/make_task2_tables.py`, `logs/robustness_tables.py`). `paper/numdiff.py` checks that `paper.md` and `main.tex` carry the same measurements. | Answered | Appendix A; README |
| A9.4 | Raw dumps are not shipped. | They are about 53 GB. Scripts regenerate them; every table regenerates from the released JSON without them. | Answered | README |

### A10. Novelty and related work

| # | objection | response | status | evidence |
|---|---|---|---|---|
| A10.1 | The claim that no published stage comparison reports the scoring convention rests on a short related-work section. | Stated as "we have not found", not as a verified absence. A systematic comparison table has not been built. | Open | Abstract; §8 |
| A10.2 | Tülu's vocabulary is wider than Llama's; does restricting to shared IDs bias the entropy? | Comparisons use the shared 128,256 IDs, renormalised. Excluded mass is at most 7×10⁻⁶ at prompt-final positions, 1.5×10⁻⁴ over base-written continuations and 7.8×10⁻⁴ over final-written ones. | Answered | §2.1 |

---

## Part B — Modelling choices revised along the way

Each row records an earlier belief, what overturned it, and what the paper says now.
These are the deviations a careful reader might otherwise rediscover.

### B1. Conclusions overturned (also in §7)

| # | earlier belief | what overturned it | now |
|---|---|---|---|
| B1.1 | Instruction tuning carries ~99% of the entropy loss. | Position convention (prompt-final on the 1B only). | Holds only at the 1B under prompt-final scoring. |
| B1.2 | A 1B → 7B jump in the preference share. | Position convention. | Reported with all four conventions side by side. |
| B1.3 | Reference subtraction fails catastrophically. | Wrong baseline (T = 1.0 instead of matched diversity). | 0.98× the temperature frontier. |
| B1.4 | No reference-axis operator can reach instruction tuning's loss. | Linear-only typicality fit. | Not supported; a non-linear operator is not ruled out. |
| B1.5 | λ < 0 would be harmful at any scale (analytical prediction). | 135M generation showed the opposite; the model assumed quality independent of typicality. | Prediction withdrawn. |
| B1.6 | A ψ threshold ψ* ≈ 0.71–0.83. | Not a constant: it moves from 0.000 to 1.000 with the matched-diversity target, and the model was off-regime (47× collapse against 1.5–2.2× measured). | Threshold withdrawn; ψ reported as a magnitude. `pilot-135m/exp1d_misspec.py`, `code/exp1e_peakedness.py`. |
| B1.7 | Negative ψ sign on the 1B preference delta. | Did not survive a change of numerical backend. | Not claimed. |
| B1.8 | The instruction-tuning cost is non-linear while the preference cost is a temperature. | Artefact of comparing the two stages under different fits. | Under the weighted fit both look like a local sharpening; no difference in mechanism claimed. |

### B2. Scope corrections

| # | earlier wording | correction | where |
|---|---|---|---|
| B2.1 | Context-invariance is "a property of the stage rather than of the scale". | Within OLMo, instruction tuning's ρ_reg rises with scale (0.092, 0.320, 0.445) while the preference stage's falls (0.558, 0.489, 0.386). It depends on stage and scale together. | §10; commit `161a6c6` |
| B2.2 | The remainder's preference-stage trend included the 1B. | The 1B decomposition nearly cancels, so the trend is stated within the OLMo-1124 release only (25.7% → 7.4%). | §5.1 |
| B2.3 | The 1B instruction-tuning remainder (−30.6%) is "small". | It is negative: that part raises entropy. Readable, because that decomposition does not cancel. | §5.1 |
| B2.4 | "SFT is the larger stage." | True for the OLMo 1B/7B only; Tülu-8B and the 13B invert it under continuation scoring. | `results/TASK2-REPORT.md` §8 |
| B2.5 | "The 7B is the outlier." | Unsupported at n = 4. What can be supported: the total collapse is non-monotone within OLMo. | `results/TASK2-REPORT.md` §8 |
| B2.6 | A constant ~9-point whose-text offset. | Dropped. The effect runs through instruction tuning (the denominator), not the preference stage. | `logs/offset_check.py` |
| B2.7 | A ψ upper bound of 0.25 / 0.234. | Broken by the 13B. Plausible-set ψ now tops out at 0.281. | Abstract; §4 |
| B2.8 | "The part that removes diversity is not the part that is large." | Overclaimed against "most of the shift is explained by neither axis". Replaced by the title's phrasing. | commit `a267f5d` |
| B2.9 | Section 10 offered the 7–8B temperature account as a reading of the 1B results. | §6.3 shows it cannot apply at the 1B. Recast as a hypothesis for 7–13B. | §6.3; §10 |

### B3. Measurement and protocol decisions

| # | decision | reason | where |
|---|---|---|---|
| B3.1 | Base-written continuations are the primary convention. | One position per prompt gives intervals 4.5–6× wider; continuations give ~190 positions per prompt on paired contexts. | §2.2 |
| B3.2 | The first register test (special-token mass) is kept as a null. | In these chat templates the role markers are ordinary text tokens, so special-token mass cannot see register. Replaced by the shared-shift geometry test. | `results/TASK1-REPORT.md` §3; `results/PREREG-REGISTER.md` |
| B3.3 | The 13B was added as a within-release scale pair. | The OLMo 1B and 7B differ in release, so "same recipe" could not be claimed. | `LEARNINGS.md` L-01 |
| B3.4 | Every checkpoint is loaded in bf16, asserted on every parameter. | Stored formats differ (some fp32, some pickled `.bin`); bf16 everywhere keeps comparisons like for like. | `results/TASK1-REPORT.md` §5 |
| B3.5 | The ungated Llama-3.1-8B mirror is not used as the base. | Its shards are byte-identical to Meta's, but it edits tokeniser metadata (pad token, left padding), which matters for batched use. Meta's repo is used. | `results/TASK1-REPORT.md` §5 |
| B3.6 | Numerical checks compare results only within one platform. | MPS, Mac CPU and cloud CPU drift by up to about 0.027 on the smallest delta. Numerical-backend nulls are reported as controls. | `LEARNINGS.md` L-03; §4.2 |
| B3.7 | Tokeniser identity is verified at the token-ID level. | Logit arithmetic across checkpoints needs shared IDs: 0 ID-map differences and identical prompt encodings in both families. | `code/check_tokenizers.py` |
| B3.8 | Scripts resolve paths relative to themselves. | Release runs broke when paths followed the caller's working directory. | `LEARNINGS.md` L-04 |
| B3.9 | The superseded 1B-era table script was removed, not kept. | It rebuilt tables the paper no longer uses; keeping it was worse than keeping nothing. | README |
| B3.10 | Pre-registration notes are appended after a fixed separator; the hash covers only the original text. | So that later notes stay visible and the original stays verifiable. | `results/verify_prereg.py` |

---

## Part C — Open items

These are known gaps. Each is listed so that its absence from the paper is not read as an answer.

| # | item | what would close it |
|---|---|---|
| C1 | The grid's numbers come from the stride-4 run. | Run `code/robustness.py` without `--stride`; update §5.1 from `logs/robustness_tables.py ""`. |
| C2 | Whether the typicality and shared components are separately identifiable (A1.8). | Report the per-position correlation between reference log-probability and the shared direction. |
| C3 | Sample-count versus context effects across conventions (A5.2). | Subsample continuation positions to match prompt-final counts. |
| C4 | The shared direction against frequency or formatting baselines (A7.1). | Compare ρ_reg against a token-frequency direction, and recompute it with punctuation, whitespace and special tokens excluded. |
| C5 | Where the punctuation promotion comes from (A7.2). | Compare against the preference datasets' own token statistics. |
| C6 | Quality, not just fluency, in §6 (A8.2). | Human evaluation, or a modern LLM judge at matched diversity. |
| C7 | The unexplained 1B λ null (A8.5). | Compare logit-norm ratios at 135M and 1B. |
| C8 | Independent training recipes (A6.1). | A third base family with published intermediate checkpoints. |
| C9 | Systematic related-work comparison (A10.1). | A literature table: stages, metric, scoring convention, attribution method. |
| C10 | The γ sweep is underpowered at five prompts. | Rerun on the 36-prompt set. |
| C11 | Frontier scale (A6.3). | Out of scope for this study. |
