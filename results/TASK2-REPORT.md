# Task 2 report — OLMo-2-13B, register v2, entropy decomposition, lineage, 135M γ sweep

**From:** terminal session, 13–14 Oct 2026. Numbers only; `paper-v5.md` not edited.
**Scope of every claim below:** the four checkpoint families listed, 36 prompts (`prompts.FLAT`), raw-text prompting,
bf16 weights with a per-parameter assertion, 20k bootstrap over prompts unless stated. Family-specific statements
are labelled by family. Nothing here is stated as general across post-training pipelines.

Pre-registrations (each hashed before its data existed; hashes in `results/*.sha256`):
`PREREG-13B.md` (+ appended basis-change note), `PREREG-REGISTER.md`, `PREREG-LINEAGE.md`, `PREREG-DECOMP.md`,
`PREREG-RESIDUAL.md` (+ appended narrowing note), `PREREG-135M-GAMMA.md`.

## 0. Verdicts at a glance

| pre-registered test | verdict |
|---|---|
| 13B DPO share, PRIMARY (arm B, vs 7B 33.3%) | **"trend continues"**: 49.0%, paired 13B−7B **+15.6 pts [+12.7, +18.6]**, *but* DPO's absolute drop **falls** (0.255 → 0.202 nats, −0.053 [−0.078, −0.028]) and total loss nearly halves (0.764 → 0.413). |
| 13B DPO share, artefact arm (raw prompt-final, original thresholds) | point 31.4% > 25.7% → "continues", paired +5.7 [−2.7, +12.9] includes 0 → **statistically indistinguishable from flat** |
| Register v2 (arm B, base→SFT, plausible) | **SUPPORTED in all four models**, both continuation sources; nulls ≈ 0 |
| Cross-lineage register transfer | **SUPPORTED**: X = −0.194 [−0.224, −0.163], identical to same-lineage −0.194 |
| DPO entropy loss head-driven AND typicality > shared (weighted) | **FAILED.** Head leg failed (head 15–23%); component leg supported (+118 to +527 pts); conjunction failed |
| Residual: nonlinear typicality carries base→SFT majority ("cannot touch" too strong?) | split: **too strong** OLMo-1B, Tulu; **unresolved** OLMo-7B; **stands** OLMo-13B |
| 135M small-γ temperature-like | **not falsified, not formally supported**: γ=0.05 temperature-like (fragile), γ=0.1 misses band by 0.03; nothing degenerate |

## 1. OLMo-2-1124-13B (within-release scale pair with 7B)

Mechanics: base fp32 (54.9 GB) and pickled-`.bin` SFT both loaded as bf16, assertion passed on every checkpoint;
tokeniser identity PASS across all four (0 ID-map diffs, identical prompt encodings). Correction to an earlier
statement of mine: the 13B **DPO** repo is not fp32 — it holds bf16 weights twice (safetensors + `.bin`).

Paired 7B → 13B comparison (`logs/prereg13b_eval.txt`):

| convention | DPO share 7B → 13B | 13B − 7B | DPO drop (nats) 7B → 13B | total loss 7B → 13B |
|---|---|---|---|---|
| **B base-written (primary)** | 33.3% → **49.0%** | **+15.6 [+12.7, +18.6]** | 0.255 → **0.202** (−0.053 [−0.078, −0.028]) | 0.764 → **0.413** (−0.351 [−0.384, −0.319]) |
| B final-model-written | 23.9% → 39.7% | +15.8 [+13.1, +18.4] | 0.220 → 0.185 (−0.035 [−0.060, −0.011]) | 0.919 → 0.466 |
| A chat-templated | 26.1% → 26.8% | +0.7 [−3.6, +4.7] | 0.978 → 0.995 (n.s.) | 3.743 → 3.716 (n.s.) |
| raw prompt-final (artefact arm) | 25.7% → 31.4% | +5.7 [−2.7, +12.9] | 0.289 → 0.147 (−0.141 [−0.217, −0.074]) | 1.123 → 0.469 |

**Reading.** Within the 1124 release, DPO's *share* rises from 7B to 13B at continuation positions because SFT's
entropy cost collapses (0.460 → 0.167 nats), not because DPO costs more — DPO's absolute cost falls. Under the chat
template nothing changes. "DPO's share rises with scale" must never be read as "preference optimisation costs more
diversity at scale"; these numbers contradict that. Total collapse is non-monotone within OLMo (1B < 7B > 13B under
both prompt-final and arm B) — there is no scale trend in total collapse across what was measured.

## 2. Full stage tables (every share beside its absolute drop and total)
### T1. Stage shares with absolute drops and totals, by measurement convention

| model | convention | SFT share | DPO share | DPO drop (nats) | RLVR share | total loss (nats) |
|---|---|---|---|---|---|---|
| OLMo-2-0425-1B | raw prompt-final (artefact arm) | 97.0% [88.2, 110.1] | -0.6% [-15.4, 9.1] | -0.004 | 3.6% [-0.9, 8.8] | 0.602 |
| OLMo-2-0425-1B | A chat-templated | 66.8% [57.4, 74.5] | 25.5% [18.0, 35.2] | 0.574 | 7.7% [4.6, 10.4] | 2.254 |
| OLMo-2-0425-1B | **B base-written (primary)** | 66.1% [62.2, 69.7] | 27.6% [24.1, 31.2] | 0.118 | 6.3% [5.0, 7.7] | 0.426 |
| OLMo-2-0425-1B | B final-model-written | 77.7% [74.5, 80.4] | 17.4% [15.1, 20.1] | 0.107 | 4.9% [4.1, 5.8] | 0.612 |
| OLMo-2-1124-7B | raw prompt-final (artefact arm) | 69.4% [62.8, 75.4] | 25.7% [19.9, 31.9] | 0.289 | 4.9% [4.1, 5.9] | 1.123 |
| OLMo-2-1124-7B | A chat-templated | 69.5% [64.8, 74.0] | 26.1% [21.9, 30.5] | 0.978 | 4.4% [3.1, 5.7] | 3.741 |
| OLMo-2-1124-7B | **B base-written (primary)** | 60.2% [57.5, 62.8] | 33.3% [30.9, 35.8] | 0.255 | 6.5% [6.0, 7.0] | 0.765 |
| OLMo-2-1124-7B | B final-model-written | 70.7% [68.7, 72.8] | 23.9% [22.0, 25.9] | 0.220 | 5.4% [5.0, 5.8] | 0.919 |
| Tulu-3-8B | raw prompt-final (artefact arm) | 52.7% [45.0, 59.6] | 40.8% [33.0, 50.0] | 0.393 | 6.5% [4.1, 9.1] | 0.963 |
| Tulu-3-8B | A chat-templated | 62.1% [56.5, 68.6] | 29.2% [22.4, 35.0] | 0.984 | 8.7% [6.1, 11.3] | 3.367 |
| Tulu-3-8B | **B base-written (primary)** | 40.7% [37.9, 43.2] | 52.8% [50.5, 55.4] | 0.399 | 6.5% [5.8, 7.3] | 0.756 |
| Tulu-3-8B | B final-model-written | 49.7% [46.6, 52.6] | 44.3% [41.8, 47.0] | 0.357 | 6.0% [5.2, 6.9] | 0.805 |
| OLMo-2-1124-13B | raw prompt-final (artefact arm) | 59.4% [51.8, 67.8] | 31.4% [22.3, 38.7] | 0.147 | 9.2% [6.9, 12.0] | 0.469 |
| OLMo-2-1124-13B | A chat-templated | 63.5% [57.9, 69.1] | 26.8% [22.0, 31.6] | 0.996 | 9.7% [8.1, 11.3] | 3.716 |
| OLMo-2-1124-13B | **B base-written (primary)** | 40.5% [36.7, 44.1] | 49.0% [46.0, 52.2] | 0.202 | 10.5% [9.8, 11.4] | 0.413 |
| OLMo-2-1124-13B | B final-model-written | 50.5% [47.3, 53.6] | 39.7% [37.1, 42.4] | 0.185 | 9.7% [9.0, 10.5] | 0.466 |

## 3. Methodology: the measurement convention changes the answer

- **Lead example (Tulu-3-8B):** DPO share 29.2% [22.4, 35.0] chat-templated vs 52.8% [50.5, 55.4] on base-written
  continuations — same checkpoints, same prompts, 24 points apart, neither choice obviously wrong.
- **Raw prompt-final breaks at OLMo-1B (n = 1 small model):** there it is 18–28 points below every other convention
  (−0.6% vs 17.4–27.6%); at 7B, 8B and 13B the raw value falls inside or near the range of the others. Total spread
  across conventions is *not* a scale signature: Tulu-8B's spread (23.6 pts) is as wide as OLMo-1B's (28.2 pts).
- **Precision, not bias:** prompt-final total-loss CIs are ~6× wider than arm B's (width ≈0.42 vs ≈0.07 nats), because
  arm B averages ~190 positions per prompt. OLMo-7B vs Tulu-8B total loss: prompt-final 1.123 vs 0.963, paired
  difference 0.160 [−0.016, 0.325] (not distinguishable); arm B 0.764 vs 0.756, difference 0.009 [−0.038, 0.060]
  (identical within ±0.05 nats). The convention moves both the point estimates and how much can be concluded.
- **Whose text:** scoring the final model's own text instead of base-written text lowers the DPO share by 8–10 points
  (OLMo-1B +10.2 [7.7, 12.7], OLMo-7B +9.4 [7.8, 11.0], Tulu +8.5 [6.2, 10.8]; 13B +9.3), mainly because SFT already
  concentrates much more on aligned-register text — so "whose text" is a knob on the denominator, not a cleaner measure
  of the preference stage. (SFT drop grows +0.19 / +0.19 / +0.09 nats; DPO drop changes −0.01 / −0.03 / −0.04.)
- **Boring reading ruled out:** base-model entropy is *not* higher on final-model text (−0.055 n.s., −0.21, −0.27 nats),
  so the larger SFT drop is not the base model being out of distribution. SFT's absolute entropy is lower on that text
  (−0.25, −0.40, −0.37) and the extra concentration appears at SFT and does not grow after.

## 4. ψ — absolute magnitudes (no threshold framing)

Plausible-set ψ across four models, four deltas and three conventions: **≤ 0.281** (max: OLMo-13B SFT→DPO,
final-written). Under the pre-registered base-written arm: ≤ 0.256 (OLMo-13B SFT→DPO). Raw prompt-final: ≤ 0.251.
**The earlier "≤ 0.25 / ≤ 0.234" sentence is broken by 13B and must not be used.** Probability-weighted ψ_w ≤ 0.566
(OLMo-7B DPO→final, final-written). Plausible-set size per position (ref p > 1e-6): medians ≈ 900–3,800 tokens,
IQR ≈ 300–5,300, range 1–45,000.
### T2. psi on the preference delta (SFT->DPO), all regions

| model | convention | top-10 | top-50 | top-1000 | plausible | prob-weighted |
|---|---|---|---|---|---|---|
| OLMo-2-0425-1B | prompt-final | -0.104 [-0.214, +0.009] | +0.030 [-0.016, +0.075] | +0.001 [-0.036, +0.033] | +0.007 [-0.020, +0.035] | -0.028 [-0.116, +0.061] |
| OLMo-2-0425-1B | B base-written | -- | +0.085 [+0.070, +0.100] | -- | +0.013 [+0.005, +0.021] | +0.163 [+0.139, +0.188] |
| OLMo-2-0425-1B | B final-written | -- | +0.102 [+0.087, +0.118] | -- | +0.022 [+0.011, +0.033] | +0.202 [+0.175, +0.229] |
| OLMo-2-1124-7B | prompt-final | +0.087 [-0.034, +0.205] | +0.262 [+0.210, +0.311] | +0.170 [+0.128, +0.209] | +0.148 [+0.108, +0.183] | +0.349 [+0.249, +0.443] |
| OLMo-2-1124-7B | B base-written | -- | +0.255 [+0.240, +0.270] | -- | +0.142 [+0.127, +0.157] | +0.414 [+0.397, +0.432] |
| OLMo-2-1124-7B | B final-written | -- | +0.272 [+0.259, +0.286] | -- | +0.161 [+0.145, +0.177] | +0.445 [+0.429, +0.462] |
| Tulu-3-8B | prompt-final | +0.120 [+0.006, +0.232] | +0.278 [+0.206, +0.346] | +0.234 [+0.190, +0.270] | +0.221 [+0.193, +0.246] | +0.400 [+0.291, +0.495] |
| Tulu-3-8B | B base-written | -- | +0.313 [+0.294, +0.333] | -- | +0.234 [+0.223, +0.245] | +0.520 [+0.499, +0.542] |
| Tulu-3-8B | B final-written | -- | +0.347 [+0.328, +0.365] | -- | +0.253 [+0.241, +0.264] | +0.552 [+0.530, +0.575] |
| OLMo-2-1124-13B | prompt-final | +0.063 [-0.044, +0.170] | +0.179 [+0.109, +0.248] | +0.245 [+0.194, +0.287] | +0.251 [+0.218, +0.283] | +0.292 [+0.171, +0.400] |
| OLMo-2-1124-13B | B base-written | -- | +0.262 [+0.240, +0.286] | -- | +0.256 [+0.244, +0.269] | +0.433 [+0.405, +0.461] |
| OLMo-2-1124-13B | B final-written | -- | +0.285 [+0.259, +0.312] | -- | +0.281 [+0.267, +0.296] | +0.451 [+0.419, +0.483] |

Max psi per convention (4 models x 4 deltas): prompt-final: 0.251 (OLMo-2-1124-13B SFT -> DPO); pf_w: 0.495 (OLMo-2-1124-7B DPO -> Instruct); B base: 0.256 (OLMo-2-1124-13B sft->dpo); B base_w: 0.520 (Tulu-3-8B sft->dpo); B final: 0.281 (OLMo-2-1124-13B sft->dpo); B final_w: 0.566 (OLMo-2-1124-7B dpo->inst)

## 5. Register test v2 — shared-shift geometry (pre-registered; SUPPORTED)

Why v1 was blind: in OLMo-2 and Tulu-3 chat templates the role markers (`<|assistant|>`) are plain text tokens, so
special-token mass cannot see the register. v1 is kept and reported as a null (T4). Arm A (chat-templated) is
confounded — every prompt ends in an identical template suffix, so even a pure-noise delta looks shared
(null ρ_reg 0.149 at 1B) — and is not used as register evidence. Arm B is the test.

- **Pre-registered (base→SFT, plausible):** ρ_reg > 0, ρ_reg > ψ, partial > 0, all CIs excluding 0, in OLMo-1B, 7B,
  Tulu-8B and 13B, and under both continuation sources. Nulls (same weights MPS vs CPU): +0.008 (1B), +0.004 (8B).
- **Scope, exactly:** shared-shift (ρ_reg) dominates across the plausible set; in the probability-weighted region the
  shift is reweighting-like (ψ_w > ρ_w for base→SFT in all four models, and for SFT→DPO in 7B, Tulu and 13B — e.g.
  13B 0.433 vs 0.096; OLMo-7B base→SFT ρ_w − ψ_w = −0.203). OLMo-1B SFT→DPO is the exception (ρ_w 0.435 > ψ_w 0.163).
- **Measured register tokens (SFT→DPO shared shift, every model, both sources):** promotes em-dashes, dashes,
  ellipses, brackets (and hedges such as *effectively, typically, essentially* across stages); demotes *I*, *you*,
  newlines, EOS.
- **Terminology:** "shared-shift" = ρ_reg; "register-conditional concentration" = the whose-text entropy effect (§3).
- **Cross-lineage (PREREG-LINEAGE):** 1B stages scoring OLMo-2-1124-7B-Instruct text (no 1B checkpoint is its ancestor):
  X[SFT] = −0.194 [−0.224, −0.163], identical to same-lineage −0.194 [−0.230, −0.158]; DPO/final add nothing beyond
  SFT (−0.16). Supports a register-conditional concentration installed at SFT that transfers across scale and release
  within the OLMo family. Rules out checkpoint lineage, not training-data overlap (both use Tulu-3-style mixtures).
### T3. Register geometry, arm B base-written, plausible region (pre-registered test = base->SFT)

| model | delta | psi | rho_reg | partial | rho - psi | psi_w | rho_w |
|---|---|---|---|---|---|---|---|
| OLMo-2-0425-1B | base->sft | -0.017 [-0.023, -0.010] | +0.092 [+0.083, +0.101] | +0.089 [+0.080, +0.099] | +0.109 [+0.098, +0.120] | +0.210 [+0.193, +0.228] | +0.046 [+0.035, +0.055] |
| OLMo-2-0425-1B | sft->dpo | +0.013 [+0.005, +0.021] | +0.558 [+0.549, +0.566] | +0.578 [+0.569, +0.586] | +0.545 [+0.531, +0.559] | +0.163 [+0.139, +0.188] | +0.435 [+0.419, +0.451] |
| OLMo-2-0425-1B | dpo->inst | +0.062 [+0.050, +0.074] | +0.500 [+0.488, +0.510] | +0.500 [+0.488, +0.511] | +0.438 [+0.422, +0.452] | +0.118 [+0.087, +0.149] | +0.374 [+0.360, +0.388] |
| OLMo-2-0425-1B | NULL (MPS vs CPU) | -0.005 [-0.012, +0.002] | +0.008 [+0.003, +0.014] | +0.009 [+0.003, +0.015] | +0.013 [+0.005, +0.022] | -0.003 [-0.016, +0.010] | +0.008 [-0.002, +0.018] |
| OLMo-2-1124-7B | base->sft | +0.082 [+0.071, +0.094] | +0.320 [+0.310, +0.331] | +0.324 [+0.313, +0.335] | +0.238 [+0.222, +0.253] | +0.397 [+0.373, +0.422] | +0.195 [+0.179, +0.209] |
| OLMo-2-1124-7B | sft->dpo | +0.142 [+0.127, +0.157] | +0.489 [+0.480, +0.499] | +0.525 [+0.515, +0.535] | +0.347 [+0.329, +0.365] | +0.414 [+0.397, +0.432] | +0.241 [+0.223, +0.257] |
| OLMo-2-1124-7B | dpo->inst | +0.205 [+0.190, +0.221] | +0.377 [+0.366, +0.388] | +0.391 [+0.379, +0.402] | +0.172 [+0.153, +0.189] | +0.513 [+0.498, +0.530] | +0.268 [+0.252, +0.283] |
| Tulu-3-8B | base->sft | +0.069 [+0.059, +0.080] | +0.263 [+0.248, +0.276] | +0.267 [+0.252, +0.281] | +0.194 [+0.178, +0.208] | +0.339 [+0.310, +0.367] | +0.123 [+0.109, +0.135] |
| Tulu-3-8B | sft->dpo | +0.234 [+0.223, +0.245] | +0.486 [+0.475, +0.496] | +0.536 [+0.525, +0.548] | +0.252 [+0.238, +0.267] | +0.520 [+0.499, +0.542] | +0.123 [+0.108, +0.139] |
| Tulu-3-8B | dpo->inst | +0.148 [+0.139, +0.157] | +0.481 [+0.466, +0.496] | +0.492 [+0.477, +0.508] | +0.333 [+0.314, +0.353] | +0.343 [+0.319, +0.367] | +0.302 [+0.284, +0.320] |
| Tulu-3-8B | NULL (MPS vs CPU) | -0.006 [-0.011, -0.001] | +0.004 [-0.000, +0.009] | +0.004 [-0.001, +0.009] | +0.010 [+0.003, +0.018] | -0.002 [-0.009, +0.006] | +0.007 [-0.000, +0.013] |
| OLMo-2-1124-13B | base->sft | +0.118 [+0.103, +0.133] | +0.445 [+0.431, +0.457] | +0.447 [+0.433, +0.459] | +0.326 [+0.307, +0.345] | +0.304 [+0.273, +0.335] | +0.267 [+0.254, +0.280] |
| OLMo-2-1124-13B | sft->dpo | +0.256 [+0.244, +0.269] | +0.386 [+0.374, +0.398] | +0.411 [+0.397, +0.423] | +0.130 [+0.116, +0.143] | +0.433 [+0.405, +0.461] | +0.096 [+0.083, +0.108] |
| OLMo-2-1124-13B | dpo->inst | +0.218 [+0.203, +0.235] | +0.425 [+0.413, +0.436] | +0.434 [+0.422, +0.445] | +0.206 [+0.184, +0.227] | +0.479 [+0.453, +0.504] | +0.371 [+0.350, +0.391] |

### T4. Special-token mass by stage (v1 register test, raw prompt-final) -- null

| model | base | SFT | DPO | final |
|---|---|---|---|---|
| OLMo-2-0425-1B | 5.1e-09 | 2.9e-08 | 4.5e-08 | 4.2e-08 |
| OLMo-2-1124-7B | 2.7e-09 | 9.2e-09 | 1.8e-08 | 1.7e-08 |
| Tulu-3-8B | 7.0e-06 | 1.0e-05 | 6.3e-06 | 5.6e-06 |
| OLMo-2-1124-13B | 3.8e-09 | 4.0e-09 | 4.7e-09 | 4.6e-09 |

## 6. Where each stage's entropy loss comes from (exact counterfactual Shapley; arm B base-written)

**PREREG-DECOMP: FAILED.** The hypothesis was a conjunction: DPO's loss head-driven AND typicality > shared-shift
(weighted). Head leg **failed** (head = ref-top tokens covering 50% mass, median 2 tokens, carries 15–23% of DPO's
loss; body 74–86%). Component leg **supported** (weighted typ − shared +118 to +527 pts, CIs clear). Conjunction failed.

What the components show (T5): in OLMo-7B, Tulu-8B and OLMo-13B, DPO's entropy loss is carried by a small
typicality-aligned component, while the shared-shift component *raises* entropy (−16% to −55%). For OLMo-7B and
Tulu-8B (sampled check, 400 positions each; not run for 13B): β > 0 at 85–97% of positions, weighted median
β ≈ +0.09 to +0.12 (≈ local T 0.9), explaining only 2.6–4.9% (unweighted R²) / 20–29% (weighted R²) of the
shift's variance. OLMo-1B is the exception (β ≈ 0; components near-cancelling, signs only). Shapley shares
above 100% with opposite signs mean the components interact; magnitudes there are not separable.

**Stage asymmetry, stated by family.** DPO's loss is characterised positively (above). SFT's loss is **not** "neither
typicality nor shared" once typicality may be nonlinear (T6, PREREG-RESIDUAL): a piecewise-linear function of ref
log-prob carries a large part of base→SFT's loss. Pre-registered statistic (nonlinear share) → "cannot touch" is too
strong at OLMo-1B and Tulu, unresolved at OLMo-7B, stands at OLMo-13B. Exploratory merged typicality (linear +
nonlinear as one player; the split between them is arbitrary because they are collinear): 122% / 56% / 67% / 53%.
So "no decoding-time operator built on the typicality axis can reach SFT's loss" is not supported as stated; a
nonlinear (truncation-like) reference-based operator is not ruled out. Which stage is larger is family-specific:
SFT in OLMo-1B/7B, DPO in Tulu-8B and OLMo-13B at arm B.
### T5. Entropy-loss decomposition, arm B base-written (Shapley shares of each stage's loss)

| model | delta | loss (nats) | head | body | typ (unw) | shared (unw) | resid (unw) | typ (w) | shared (w) | typ(w) - shared(w) |
|---|---|---|---|---|---|---|---|---|---|---|
| OLMo-2-0425-1B | base->sft | 0.281 | 14.9% [11.9, 17.6] | 95% | -29% | -5% | 144% | 139% | 2% | +137 [+129, +146] |
| OLMo-2-0425-1B | sft->dpo | 0.118 | 14.8% [11.0, 17.7] | 86% | 153% | -209% | 157% | 291% | -236% | +527 [+452, +627] |
| OLMo-2-0425-1B | dpo->inst | 0.027 | 25.2% [23.0, 27.3] | 66% | -20% | 97% | 14% | 4% | 84% | -80 [-139, -42] |
| OLMo-2-1124-7B | base->sft | 0.460 | 23.0% [22.0, 23.9] | 75% | 27% | -4% | 75% | 95% | -3% | +98 [+95, +101] |
| OLMo-2-1124-7B | sft->dpo | 0.255 | 20.7% [19.3, 22.0] | 80% | 60% | -47% | 87% | 128% | -55% | +183 [+175, +193] |
| OLMo-2-1124-7B | dpo->inst | 0.050 | 25.0% [23.6, 26.2] | 72% | 37% | -4% | 64% | 90% | -2% | +92 [+85, +99] |
| Tulu-3-8B | base->sft | 0.307 | 23.0% [21.9, 24.0] | 75% | 33% | 3% | 61% | 97% | -1% | +97 [+94, +101] |
| Tulu-3-8B | sft->dpo | 0.399 | 22.7% [22.2, 23.2] | 74% | 69% | -28% | 54% | 105% | -28% | +133 [+129, +136] |
| Tulu-3-8B | dpo->inst | 0.049 | 24.7% [23.4, 25.9] | 72% | 51% | -3% | 48% | 91% | -0% | +91 [+88, +95] |
| OLMo-2-1124-13B | base->sft | 0.167 | 21.8% [19.7, 23.3] | 75% | 51% | 4% | 41% | 87% | 4% | +83 [+78, +86] |
| OLMo-2-1124-13B | sft->dpo | 0.202 | 21.2% [20.2, 22.2] | 74% | 77% | -18% | 35% | 102% | -16% | +118 [+114, +121] |
| OLMo-2-1124-13B | dpo->inst | 0.043 | 26.1% [25.1, 27.0] | 70% | 48% | 3% | 46% | 90% | 1% | +88 [+86, +92] |

### T6. Residual test: base->SFT entropy loss with NONLINEAR typicality and cross-fitted shared directions

| model | nonlinear typ (prereg stat) | linear typ | merged typ (exploratory) | shared | eps (merged run) | prereg verdict on 'cannot touch' |
|---|---|---|---|---|---|---|
| OLMo-2-0425-1B | 200.4% [170.2, 235.2] | -110.5% [-136.2, -88.3] | 122.3% [106.8, 139.8] | 10.1% [6.8, 13.9] | -28.0% [-45.5, -12.0] | too strong |
| OLMo-2-1124-7B | 54.0% [45.0, 63.8] | -2.2% [-7.9, 3.3] | 55.7% [48.8, 63.2] | -6.8% [-8.6, -5.1] | 40.7% [33.2, 47.7] | unresolved |
| Tulu-3-8B | 75.6% [65.5, 85.7] | -12.3% [-19.5, -5.4] | 67.0% [59.8, 75.1] | -3.5% [-7.4, -0.2] | 26.5% [17.9, 34.3] | too strong |
| OLMo-2-1124-13B | 23.3% [8.7, 38.6] | 28.2% [19.5, 36.5] | 53.2% [41.4, 64.9] | 3.9% [1.5, 6.3] | 33.0% [21.2, 44.7] | stands |

## 7. 135M decontamination γ sweep (PREREG-135M-GAMMA)

Faithful port of `pilot-135m/gen_one.py` + `score.py` (diff: paths, condition list, per-prompt output only).
Reproduction: T = 1.0 exact (Jaccard 0.078, ppl 65.0); γ = 0.5 0.004 / 17,225 vs published 0.0022 / 19,956; γ = 0.2
0.042 / 366 vs 0.042 / 339 — same order, as pre-registered. Step 0: β (SmolLM2 base → Instruct, whole pipeline)
median +0.081 at continuation positions (IQR [+0.017, +0.152], 80% > 0), +0.142 at prompt-final (n = 5).

| γ | Jaccard | ppl | ratio to temperature frontier at same Jaccard | 95% CI (5 prompts) | verdict |
|---|---|---|---|---|---|
| 0.05 | 0.078 | 79 | 1.21 | [0.79, 1.33] (52% of draws in range) | temperature-like (fragile) |
| 0.10 | 0.063 | 106 | 1.05 | [0.71, 1.53] | inconclusive (band missed by 0.03) |
| 0.20 | 0.042 | 366 | 1.62 | [0.37, 4.06] | inconclusive |
| 0.30 | 0.023 | 1,949 | 1.59 | [0.57, 4.02] | inconclusive |
| 0.50 | 0.004 | 17,225 | 1.05 | [0.48, 1.85] | inconclusive; point on frontier |

Hypothesis not falsified (no small γ degenerates), not formally supported. Independently of the sweep, re-reading the
*published* numbers at matched diversity (`logs/frontier_135m.txt`): γ = 0.5's ppl 19,956 is 0.98× the temperature
frontier at its Jaccard (≈ T 1.9). **The published "γ = 0.5 raised perplexity to 19,956" compares against T = 1.0, not
matched diversity; it is overshoot along the temperature frontier, not off-frontier harm** (the original
matched-diversity table stopped at Jaccard 0.015, so γ = 0.5 was never compared). The chat-template-token leakage claim
is qualitative and not tested here. λ < 0 does beat temperature at 135M (ratios 0.17–0.37).

## 8. Corrections made during the session (so the draft does not inherit them)
- 13B DPO storage: bf16 ×2 formats, not fp32. Tulu DPO: bf16 16 GB, not fp32.
- OLMo-7B arm-B DPO share is 33.3% [30.9, 35.8] (not 33.4).
- ρ_reg ≫ ψ holds in plausible/top-50 regions, not in the weighted region.
- "Constant ~9 pt" whose-text offset dropped; mechanism is SFT (denominator), not DPO.
- "SFT is the larger stage" is OLMo-1B/7B only; Tulu-8B and OLMo-13B invert it at arm B.
- ψ bound "≤ 0.25 / ≤ 0.234" broken by 13B (see §4).
- "7B is the outlier" unsupported at n = 4; supportable: total collapse non-monotone within OLMo.
- "Prompt-final makes 7B and Tulu look different": no — it cannot distinguish them (precision point, §3).

## 9. Files
Scripts: `code/register_v2.py`, `code/decomp.py`, `code/decomp_resid.py`, `code/pilot135m_port/`, plus
`logs/{prereg13b_eval,offset_check,make_task2_tables}.py` (lineage evaluation ran inline; output in `logs/lineage_eval.txt`). Results: `results/results_psi_olmo13b.json`,
`results/results_register_*.json`, `results/results_decomp*_*.json`, `code/dumps/135m/results_sweep_scored.json`,
`code/dumps/135m/beta135m.json`. Logs: `logs/*_analysis.txt`, `logs/prereg13b_eval.txt`, `logs/lineage_eval.txt`,
`logs/prereg135m_eval.txt`, `logs/frontier_135m.txt`, `logs/decomp*.log`.
