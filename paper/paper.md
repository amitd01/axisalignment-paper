# Alignment's Diversity Cost Is Carried by a Small Part of the Shift

### Stage-resolved measurement across four post-training pipelines

**Draft v1.0 (in revision) — 17 October 2026. Supersedes v0.9.**

---

## Abstract

Post-training alignment reduces the diversity of language model outputs, and the
loss is widely attributed to reinforcement learning on human preferences, with
*typicality bias* — a reweighting toward what the base model already found likely
— offered as its mechanism. We measure both against four pipelines that publish
every intermediate checkpoint: OLMo-2-0425-1B, OLMo-2-1124-7B, OLMo-2-1124-13B
and Llama-3.1-Tülu-3-8B. Throughout, *diversity* means next-token entropy at the
scored positions.

**Composition and consequence come apart.** A typicality-aligned component of each
stage's log-probability shift accounts for a minority of that shift's variance: a
median of at most 2.2% for instruction tuning and 7.1% for the preference stage
under an unweighted fit allowing linear and piecewise-linear terms, and at most
about a third under a probability-weighted linear fit. An exact counterfactual Shapley
attribution of each stage's entropy change assigns that same component **nearly
all (87–139%) of instruction tuning's entropy loss and all (102–291%) of the
preference stage's**, under the pre-registered weighted fit. Across six weighted
specifications these become 77–157% and 90–346%; under an unweighted fit they fall
to −105–56% and 35–178%, so the magnitude is a property of the weighting. Comparing
like with like, a component accounting for at most about a third of the shift
carries nearly all of its entropy cost. The diversity cost is carried by a small part of the shift.

**By correlation, the larger of the two components we measured is a shared direction.** Writing ρ_reg for the correlation between
a stage's per-token shift and the mean shift on *other* prompts
(leave-one-prompt-out), and ψ for its correlation with reference log-probability,
ρ_reg exceeds ψ across the plausible set at every stage of all four models, with
intervals excluding zero and null controls at +0.008 and +0.004. The
preference-stage direction promotes dashes, em-dashes, ellipses and brackets and
demotes *you* in all four. Neither component is most of the shift: squared, the
shared direction accounts for roughly 1–31% of a stage's shift and under 1% for the
1B's instruction tuning, so the majority is explained by neither axis. A second,
distinct quantity — how far a stage's entropy
falls on aligned-register text, which we call *register-conditional
concentration* — transfers across lineage: OLMo-1B's instruction-tuning stage
concentrates on text written by OLMo-2-1124-7B-Instruct exactly as much as on its
own family's (−0.194 both).

**Where you measure changes the answer.** Scoring the same checkpoints on the same
prompts, Tülu-3-8B's preference stage accounts for 29.2% [22.4, 35.0] of total
entropy loss at a chat-template position and 52.8% [50.5, 55.4] over generated
continuations. Prompt-final scoring, closest to common practice, gives confidence
intervals 4.5–6× wider because it averages one position per prompt rather than
~190, and on the 1B model it falls 18–28 points below every other convention. We
have not found a published stage comparison that reports this choice.

We also report two decoding-time interventions and a correction to our own earlier
work. Extrapolating along the base–aligned logit axis recovers diversity at a
third of the fluency cost of temperature at 135M (3.00×, 95% CI [1.58, 5.14]) and
does not survive to 1B (1.04×, [0.73, 1.50], n = 12, Wilcoxon p = 0.97).
Reference subtraction, which an earlier draft reported as catastrophic, is not: at
matched diversity it sits at 0.98× the temperature frontier, and the earlier
figure compared it against T = 1.0 instead. Plausible-set ψ never exceeds 0.281
across four models, four stage deltas and three conventions.

---

## 1. Introduction

That language models homogenise expression has moved from commentary into
measurement. Doshi and Hauser (2024) found that writers given LLM story ideas
produced work rated more creative individually while the corpus became more
similar in aggregate. Sourati, Ziabari and Dehghani (2026) synthesised evidence
across four disciplines that models reinforce dominant styles.

The technical literature localises the loss to the alignment step. Kirk et al.
(2024) established a generalisation–diversity trade-off under RLHF, and the phrase
"RLHF reduces diversity" is standard. *Typicality bias* (Zhang et al., 2025)
supplies a mechanism: human preference data favours the conventional, and
KL-regularised optimisation against it amplifies majority-style responses.

Both are usually evaluated by comparing a base model with a released instruct
model, collapsing instruction tuning, preference optimisation and any later stages
into one delta. Pipelines publishing intermediate checkpoints allow the stages to
be separated. We separate them on four, and ask two questions of each stage: what
is its shift made of, and which part of it removes entropy. The answers are
different components.

### 1.1 Contributions

1. **Composition is not consequence.** In every model and at both stages, a
   typicality-aligned component accounting for a minority of the shift's variance
   carries nearly all of its entropy loss.
2. **A shared-shift geometry, measured and replicated.** ρ_reg > ψ across the
   plausible set at every stage of all four models, under two continuation
   sources, with null controls near zero and the token direction stated
   explicitly. In the probability-weighted region the ordering reverses (§4.5).
3. **Cross-lineage transfer of register-conditional concentration**, ruling out
   checkpoint ancestry
   though not shared training data.
4. **The measurement convention determines the stage attribution.** Four
   conventions on four models; the same checkpoints yield preference-stage shares
   24 points apart, and the effect is one of statistical precision as well as
   bias.
5. **Two decoding-time negative results**, one scale-dependent, plus corrections
   to a matched-diversity error and a withdrawn threshold in our own earlier
   drafts.

Every claim below is scoped to the models, conventions and regions it was measured
on. None is stated as general across post-training pipelines.

---

## 2. Measurement

### 2.1 Models and protocol

| pipeline | base | stages | params |
|---|---|---|---|
| OLMo-2-0425 | OLMo-2 | base → SFT → DPO → Instruct (RLVR) | 1B |
| OLMo-2-1124 | OLMo-2 | base → SFT → DPO → Instruct (RLVR) | 7B |
| OLMo-2-1124 | OLMo-2 | base → SFT → DPO → Instruct (RLVR) | 13B |
| Llama-3.1-Tülu-3 | Llama 3.1 | base → SFT → DPO → Instruct (RLVR) | 8B |

The 7B and 13B share a release, so comparisons between them vary scale alone. The
1B is a different release (April 2025 vs November 2024), so comparisons involving
it hold the recipe near-constant rather than constant.

**Diversity, in this paper, is next-token entropy at the scored positions.** The
generation experiments of §6 use mean pairwise Jaccard instead, and say so.

36 prompts across six task families. Each checkpoint is loaded alone, in bf16 with
a per-parameter assertion after load, and its next-token log-probability vectors
cached before the next is loaded; all delta arithmetic happens afterward on saved
arrays. Confidence intervals are bootstrap over prompts, 20,000 resamples.

The stage, register and decomposition analyses ran on Apple MPS; the 1B generation
sweep of §6.1 and all 135M runs ran on CPU. The Tülu pipeline was reproduced in
full on CPU under the prompt-final convention and agrees on every share to within
0.3 percentage points.

Tokeniser identity was verified at the token-ID level for every pipeline, a
requirement for logit arithmetic rather than a formality. OLMo needed no
restriction. Tülu's SFT, DPO and final heads are 128,264 wide against
Llama-3.1-8B's 128,256, so Tülu comparisons use the shared 128,256 IDs,
renormalised; excluded mass reaches 7×10⁻⁶ at prompt-final positions,
1.5×10⁻⁴ over base-written continuations and 7.8×10⁻⁴ over final-written ones.

### 2.2 Four conventions

Entropy and the per-token shift must be evaluated at some set of positions, and
the literature does not state which.

- **Raw prompt-final.** One position per prompt: the last token of the raw prompt.
  Closest to common practice and what our own earlier drafts used.
- **Chat-templated.** The prompt wrapped in the aligned model's chat template,
  scored at the assistant-start position, with identical token IDs fed to all four
  stages.
- **Continuation, base-written (primary).** The base model generates 6 × 32 tokens
  per prompt; all four stages are teacher-forced through that text so contexts are
  paired. ~190 positions per prompt.
- **Continuation, final-model-written.** The same, with the final model as writer.

Base-written continuations were pre-registered as the source for the register test
of §4 before any of that data existed. They were adopted as primary for the stage
shares later, in a dated note appended after the 1B, 7B and 8B continuation shares
had been seen but before any 13B data was collected. Both pre-registrations are
released with their hashes at https://github.com/amitd01/axisalignment-paper, and the appended note is part of the record.

### 2.3 The convention changes the answer

**Preference-stage share of total entropy loss, by convention:**

| model | prompt-final | chat-template | continuation (base) | continuation (final) |
|---|---|---|---|---|
| OLMo-2-1B | **−0.6%** [−15.4, 9.1] | 25.5% [18.0, 35.2] | 27.6% [24.1, 31.2] | 17.4% [15.1, 20.1] |
| OLMo-2-7B | 25.7% [19.9, 31.9] | 26.1% [21.9, 30.5] | 33.3% [30.9, 35.8] | 23.9% [22.0, 25.9] |
| Tülu-3-8B | 40.8% [33.0, 50.0] | **29.2%** [22.4, 35.0] | **52.8%** [50.5, 55.4] | 44.3% [41.8, 47.0] |
| OLMo-2-13B | 31.4% [22.3, 38.7] | 26.8% [22.0, 31.6] | 49.0% [46.0, 52.2] | 39.7% [37.1, 42.4] |

Tülu is the clearest case: 29.2% and 52.8% are the same checkpoints and the same
prompts, 24 points apart, and neither position is obviously the wrong one to
score.

**Three things follow.**

*Prompt-final breaks at small scale.* On the 1B model it sits 18–28 points below
every other convention, reporting that preference optimisation does nothing
measurable — a conclusion no other convention supports. At 7B, 8B and 13B the raw
value falls inside or near the range of the others. We observe this at n = 1 small
model and do not generalise it.

*The spread is not monotone in scale*: 28.2 points at 1B, 9.4 at 7B, 23.6 at 8B
and 22.2 at 13B. The 7B and 13B share a release and differ by more than a factor
of two, so sensitivity to position is not a fixed property of a pipeline either.

*The effect is about precision as much as bias.* Prompt-final averages one
position per prompt; continuation scoring averages ~190. Total-loss confidence
intervals are correspondingly 4.5–6× wider by model. Comparing OLMo-7B against
Tülu-8B: under prompt-final the totals read 1.123 and 0.963, a paired difference
of 0.160 [−0.016, 0.325] — indistinguishable; under continuation scoring they read
0.765 and 0.756, difference 0.009 [−0.038, 0.060] — identical to within ±0.05
nats. The convention moves both the point estimates and how much can be concluded
from them.

*Whose text matters too, through the denominator.* Scoring the final model's own
text rather than base-written text lowers the preference share by 8–10 points in
every model. This is not the preference stage behaving differently — its absolute
drop barely moves (−0.01 to −0.04 nats) — but instruction tuning doing more
(+0.07 to +0.19 nats), enlarging the total. The obvious explanation, that the base
model is out of distribution on aligned text, is ruled out: base entropy is not
higher there but lower at 7B, 8B and 13B (−0.21, −0.27, −0.14) and unchanged at
1B (−0.055 [−0.112, +0.008]).

**Reporting standard.** State the position convention. Report every stage share
beside its absolute drop in nats and the pipeline total, because the three move
independently — §3.1 gives a case where a share rises while its absolute cost
falls and the total halves.

---

## 3. Stage attribution

### 3.1 Shares, drops and totals

Under the primary convention:

| model | SFT | preference | RLVR | total loss |
|---|---|---|---|---|
| OLMo-2-1B | **66.1%** [62.2, 69.7] | 27.6% [24.1, 31.2] | 6.3% [5.0, 7.7] | 0.426 |
| OLMo-2-7B | **60.2%** [57.5, 62.8] | 33.3% [30.9, 35.8] | 6.5% [6.0, 7.0] | 0.765 |
| Tülu-3-8B | 40.7% [37.9, 43.2] | **52.8%** [50.5, 55.4] | 6.5% [5.8, 7.3] | 0.756 |
| OLMo-2-13B | 40.5% [36.7, 44.1] | **49.0%** [46.0, 52.2] | 10.5% [9.8, 11.4] | 0.413 |

Which stage is larger varies by model, and within the OLMo-1124 release it flips
between the 7B and the 13B. The universal form of "RLHF reduces diversity" is not
supported, and neither is its negation.

**A share is not a cost.** Between the 7B and 13B — same release, scale the only
variable — the preference share rises from 33.3% to 49.0%, a paired difference of
+15.6 points [+12.7, +18.6]. Over the same step its *absolute* entropy cost falls,
0.255 → 0.202 nats (−0.053 [−0.078, −0.028]), and the pipeline total nearly
halves, 0.765 → 0.413, because instruction tuning's cost collapses from 0.460 to
0.167. Under the chat-template convention nothing changes at all (+0.7
[−3.6, +4.7]). "The preference share rises with scale" is true here and means
close to the opposite of what it sounds like.

Total collapse is non-monotone within OLMo (1B < 7B > 13B under both prompt-final
and continuation scoring). The 13B loses the least of any model under prompt-final
(0.469 against 0.602 for the next lowest), but under continuation scoring it is
indistinguishable from the 1B (−0.013 [−0.053, +0.027]), and under the chat
template it loses more. There is no scale trend in total collapse across these
four.

RLVR is small everywhere: 3.6–10.5% across all models and conventions, though the
7B and 13B differ within the primary convention (6.5% [6.0, 7.0] against 10.5%
[9.8, 11.4]).

---

## 4. What the shift is made of

### 4.1 Two geometries

For each stage delta we compute the per-token shift
δ(v) = log p_target(v | c) − log p_ref(v | c) and correlate it with two things:

- **ψ** — the reference model's own log-probability, log p_ref(v | c). This is the
  quantity the typicality account makes a claim about.
- **ρ_reg** — the mean shift at all positions of the *other* 35 prompts
  (leave-one-prompt-out, so no in-sample leakage). High ρ_reg means the stage
  applies the same shift regardless of context.

We also report the partial correlation of δ with the shared direction controlling
for log p_ref, which asks whether the shared direction is typicality under another
name.

### 4.2 The shared shift exceeds typicality across the plausible set

Pre-registered test, base → SFT, plausible region (ref p > 10⁻⁶), base-written
continuations:

| model | ψ | ρ_reg | partial | ρ_reg − ψ |
|---|---|---|---|---|
| OLMo-2-1B | −0.017 | **+0.092** [+0.083, +0.101] | +0.089 | +0.109 [+0.098, +0.120] |
| OLMo-2-7B | +0.082 | **+0.320** [+0.310, +0.331] | +0.324 | +0.238 [+0.222, +0.253] |
| Tülu-3-8B | +0.069 | **+0.263** [+0.248, +0.276] | +0.267 | +0.194 [+0.178, +0.208] |
| OLMo-2-13B | +0.118 | **+0.445** [+0.431, +0.457] | +0.447 | +0.326 [+0.307, +0.345] |

All four pass the pre-registered criterion: ρ_reg > 0, ρ_reg > ψ, partial > 0, all
intervals excluding zero, under both continuation sources (the smallest margin
under final-written text is 0.097 [0.085, 0.111], at the 13B preference delta).
**Null controls** — identical weights evaluated on two numerical backends, run on
OLMo-1B and Tülu-8B — give ρ_reg = +0.008 [+0.003, +0.014] and +0.004
[−0.000, +0.009]. The partial correlation is within 0.003 of, or larger than, the
raw one in every case, so the shared direction is not typicality in disguise.

The preference delta shows the same pattern more strongly at the 1B, 7B and Tülu
(ρ_reg = +0.558, +0.489, +0.486) and less so at the 13B, where it is +0.386 —
smaller than that model's base → SFT value, with ρ_reg − ψ shrinking from 0.326 to
0.130.

### 4.3 The direction, in tokens

The preference-stage shared shift promotes dashes, em-dashes, ellipses and
brackets, and demotes *you*, in all four models and under both continuation
sources. *I* is among the fifteen most-demoted tokens in the 1B, 7B and Tülu under
base-written text, but not in the 13B under either source, nor in Tülu under
final-written text. In the 1B and 7B it also demotes newlines and end-of-sequence — which
instruction tuning had *promoted* in those same models — while Tülu and the 13B
promote paragraph-break tokens instead. Hedges (*effectively*, *essentially*,
*clearly*) are top-ranked only in the 1B.

Token lists are per delta and per model; the stable core across all four is
punctuation up, second person down.

### 4.4 A related quantity transfers across lineage

This subsection concerns a different quantity from §4.2. **Register-conditional
concentration** is how much further a stage's entropy falls when scoring
aligned-register text rather than base-written text; **shared shift** (ρ_reg) is
the context-invariance of the log-probability shift. The lineage test below
measures the former, not the latter.

Register-conditional concentration could be nothing more than a model being
sharper on text written by its own descendants. OLMo-1B and
OLMo-2-1124-7B share a tokeniser, which permits a direct test: score
7B-Instruct-written text through the 1B stages, where no 1B checkpoint is an
ancestor of the writer.

The concentration is identical: **X[SFT] = −0.194 [−0.224, −0.163]** against
same-lineage −0.194 [−0.230, −0.158]. Later stages add nothing beyond SFT.

This rules out checkpoint ancestry as the explanation for
register-conditional concentration. It says nothing directly about whether the
shared direction of §4.2 transfers; we did not test that. It does not rule out
shared training data —
both pipelines use Tülu-3-style instruction mixtures — and that remains the live
alternative explanation.

### 4.5 At the head of the distribution it looks different

The picture above is of the broad plausible set, with a median of roughly
900–3,800 tokens per position. In the probability-weighted region the ordering
reverses: ψ_w exceeds ρ_w for base → SFT in all four models, and for the
preference delta at 7B, 8B and 13B (the 13B reads ψ_w 0.433 against ρ_w 0.096;
OLMo-7B's base → SFT gives ρ_w − ψ_w = −0.203). OLMo-1B's preference delta is the
exception, with ρ_w 0.435 > ψ_w 0.163.

The probability-weighted statistic is heavily concentrated: the tokens covering
50% of base mass number a *median of two*. So "the head looks like reweighting" is
a statement about the top one or two candidates per position, not about half the
distribution. Both readings hold at their own scope.

**ψ in absolute terms.** Across four models and four stage deltas, plausible-set ψ
never exceeds **0.281** under prompt-final and the two continuation conventions,
and never exceeds 0.256 under the primary convention. Probability-weighted, it
never exceeds 0.566 under those three; including chat-template positions it
reaches 0.663 (13B, DPO → final), whose plausible-set maximum is 0.237.

---

## 5. What costs the diversity

Composition and consequence are different questions, and this section separates
them. We attribute each stage's entropy change using counterfactual
distributions — apply only part of the stage's log-probability shift, renormalise,
recompute entropy — with Shapley values over the parts, which sum exactly to the
measured ΔH. Shapley efficiency does not remove interactions; it allocates them
among the parts. This is an attribution under a stated decomposition, not causal
identification of what training did.

Two terms are used narrowly. The *typicality-aligned component* is the fitted
term β · log p_ref, a reference-probability-aligned part of the shift; reading it
as the typicality bias of the source literature is an interpretation (§9,
limitation 4). The *shared-shift component* is γ · g, where g is the
leave-one-prompt-out mean shift. Both are fitted at every position, so the
bootstrap needs no refit; g itself is computed once on the full sample and held
fixed across resamples.

### 5.1 A component explaining a minority of the shift carries nearly all of the loss

**The pre-registered result.** Under the weighted linear fit specified in advance,
the typicality-aligned component carries **nearly all of instruction tuning's
entropy loss (87.0% [84.9, 88.9] at the 13B, 94.5% [92.7, 96.4] at the 7B, 96.6%
[94.5, 98.8] at Tülu, 139.0% [131.7, 147.5] at the 1B) and all of the preference
stage's (102.2% to 290.5%)**. Only the preference stage and the 1B's instruction
tuning reach 100%; the other three instruction-tuning intervals sit below it. Shares
above 100% with opposite-signed partners mean the components interact; magnitudes
are not separable there, but the sign and the ordering are.

*A worked example.* The 1B's preference stage loses 0.118 nats per position. Averaged
over the orders in which the parts can be added, the typicality-aligned part removes 0.342 nats (290.5%); the shared-shift part restores 0.278 (−236.2%);
the remainder removes 0.054 (45.8%) and the off-set tokens 0.000 (−0.2%). The four
sum to the measured loss. A share above 100% means the component removes more
entropy than the stage does in total, because another part puts some back.

**How much of the shift that component is — exploratory.** The variance analysis
below was run after the other results, to ask how large the costly component is.
It was not pre-registered and is reported as exploratory. Median R² of each
stage's shift over the plausible set explained by typicality (linear and
piecewise-linear in reference log-probability), base-written continuations,
400 sampled positions per model:

| model | base → SFT | SFT → DPO |
|---|---|---|
| OLMo-2-1B | 0.7% | 1.0% |
| OLMo-2-7B | 1.5% | 3.6% |
| Tülu-3-8B | 1.4% | 6.2% |
| OLMo-2-13B | 2.2% | 7.1% |

Recomputed over all ~6,800 positions per model with bootstrap intervals, these
hold: 0.73% [0.68, 0.76], 1.57% [1.37, 1.78], 1.45% [1.29, 1.64] and 2.20%
[1.80, 2.58] for instruction tuning across the 1B, 7B, Tülu and 13B, and
1.1–7.1% for the preference stage with an upper bound of 7.9%. Under a
probability-weighted *linear* fit — the pre-registered one, with no
piecewise-linear term — the same quantity is larger: up to 19.2% [16.9, 21.4] for
instruction tuning and 30.4% [27.96, 33.42] for the preference stage. An
exploratory variant merging linear and non-linear typicality assigns the component
53–122% and 95–204% of the two stages' losses.

**Comparing like with like — the weighted linear fit on both sides — a component
accounting for at most about a third of a stage's shift carries nearly all of its
entropy loss, in every model and at both stages.** Under an unweighted fit
admitting a piecewise-linear term, the same component accounts for at most 7.1%.
That is the paper's central measurement, and the weighted pairing is the
conservative one. The weighting is not arbitrary: entropy is −Σ p log p, so each
token's contribution to it is already weighted by its probability, and a fit
weighted the same way is judged on the tokens that move entropy. The unweighted
fit treats a token at p = 10⁻⁶ like one at p = 0.3. Note also that the two halves of
the sentence have different standing: the entropy shares are pre-registered, the
variance shares exploratory.

**Robustness — exploratory.** `code/robustness.py` repeats the decomposition over
twelve specifications: weighted or unweighted fit; a plausible set of p_ref > 10⁻⁵,
10⁻⁶ or 10⁻⁷; and a shared direction estimated from all other prompts or only from
prompts in other task families. It reports the variance share and the entropy share
under the same fit, with a bootstrap over prompts stratified by task family. These
runs score every fourth position; at the pre-registered specification they reproduce
the full-position shares to within 2.1 points at the 7B, Tülu and 13B, and run 8.6
and 31.2 points high at the 1B, whose loss is smallest (`logs/robustness_s4.txt`).

| stage | model | weighted fit | unweighted fit | weighted R² |
|---|---|---|---|---|
| instruction tuning | 1B | 127.9 to 156.8% | −104.8 to 35.2% | 14.4 to 15.4% |
| | 7B | 83.8 to 100.8% | 9.2 to 45.0% | 24.7 to 25.2% |
| | Tülu-8B | 86.0 to 102.5% | 5.1 to 52.1% | 20.5 to 20.9% |
| | 13B | 77.3 to 90.8% | 43.3 to 56.1% | 20.5 to 20.8% |
| preference | 1B | 266.8 to 345.5% | 111.1 to 178.4% | 15.7 to 16.4% |
| | 7B | 115.3 to 135.3% | 35.1 to 80.6% | 27.6 to 27.8% |
| | Tülu-8B | 90.8 to 111.4% | 51.1 to 73.3% | 33.8 to 34.2% |
| | 13B | 89.6 to 107.0% | 65.8 to 82.0% | 27.4 to 28.2% |

*Typicality-aligned share of each stage's entropy loss, minimum to maximum over the
six specifications of each fit; last column, median per-position variance share
under the weighted fit.*

Under the weighted fit the result survives every specification: the component
carries 77.3–156.8% of instruction tuning's loss and 89.6–345.5% of the preference
stage's while accounting for 14.4–34.2% of the shift's variance. "All" of the
preference stage's loss becomes "nearly all" at the loosest threshold, where the
13B and Tülu move 16.5% and 18.1% of the loss to tokens outside the plausible set.
Holding the shared direction out by task family changes no share by more than 3.4
points, so it is not a within-family formatting habit. Within each of the six
families the pre-registered share is 81.7–167.4% and 99.0–495.7%, and leaving one
family out moves it by at most 4.4 points — except at the 1B preference stage,
whose components nearly cancel. Under the unweighted fit only the contrast
survives: the entropy share exceeds the variance share in 92 of 96 cells, but the
share itself falls to −104.8–56.1% and 35.1–178.4%, and at the 1B's instruction
tuning its sign depends on the threshold (the four exceptions). **"Nearly all" is a
property of the probability-weighted fit.** The final stage, reported without
prediction, matches the others at the 7B, Tülu and 13B (typicality 89.8–91.2%) but
not at the 1B, where the shared direction carries 80.3% and typicality 9.0%
[−15.2, 25.2].

**The remainder is not inert.** For the preference stage, the component orthogonal
to both axes falls with scale within the OLMo-1124 release: 25.7% [22.1, 29.3] of
the entropy loss at the 7B against 7.4% [5.2, 9.7] at the 13B, with Tülu-8B — a
different family — at 14.7% [12.2, 17.5]. The 1B's 45.8% [37.0, 57.0] is not
comparable to those: its decomposition nearly cancels there (typicality 290.5%
against a shared share of −236.2%), so only the positive sign is interpretable.

At instruction tuning the 1B is the interesting case, and it runs the other way:
the remainder *raises* entropy, −30.6% [−37.7, −24.1], partly offsetting a
typicality component of 139.0%. That decomposition does not near-cancel — the
shared share is 2.1% — so unlike the 1B preference stage its magnitude can be
read. At the 7B, Tülu and 13B the remainder carries 1–6%.

Attributing nearly all of the loss to the typicality-aligned component is
therefore a statement about that component's share, not a demonstration that
nothing else moves entropy. We do not characterise what the remainder is.

### 5.2 Whether the two stages differ depends on the fit

**The preference stage.** Its entropy loss comes mostly from the body of the
distribution: the head (reference-top tokens covering 50% of mass, median 2 tokens)
carries 15–23%, the body 74–86%. The decomposition's pre-registration predicted the
opposite — a head share above 50% — so its primary hypothesis is **not supported**
in any model; its secondary condition, typicality above shared shift, holds in all
four. Decomposing by component, the loss is carried by
the typicality-aligned part while the shared-shift part pushes entropy back *up* —
by 16%, 28% and 55% at the 13B, Tülu-8B and OLMo-7B on the weighted fit (18%, 28%,
47% unweighted). The weighted typicality-minus-shared contrast runs +118 to +527
points with intervals clear of zero in all four models.

Sampling 400 positions each **at the 1B, 7B and Tülu-8B** (not the 13B), the
typicality-aligned
component is a positive sharpening at 85–97% of positions, with weighted median
**β ≈ +0.09 to +0.12 — a local temperature of about 0.9.** The 1B is the
exception **for the preference stage**: β ≈ 0 there, with components nearly
cancelling, so only their signs are interpretable. Its instruction-tuning stage
behaves like the others, β ≈ +0.10. β was not measured at the 13B, where only the
Shapley shares are available.

**Instruction tuning, and a fit-dependence we initially mistook for a mechanism.**
An earlier draft reported that instruction tuning's costly component was not a
temperature while the preference stage's was. That contrast is an artifact of
comparing the two stages under different fits.

Under the **weighted** fit — the pre-registered one — the two stages look alike.
Instruction tuning's sharpening coefficient is β ≈ +0.08 to +0.115, against the
preference stage's +0.09 to +0.12, and linear typicality alone carries 87–139% of
instruction tuning's loss.

Under an **unweighted** fit they diverge. A single linear coefficient then captures
between −29% and 51% of instruction tuning's loss, and a piecewise-linear function of
reference log-probability is needed: it captures 200%, 54%, 76% and 23% across the
four models on the pre-registered statistic, or 122% / 56% / 67% / 53% merging
linear and non-linear typicality (they are collinear, so the split between them is
arbitrary). The 7B and 13B interval lower bounds are 49% and 41%.

So: **under an unweighted fit instruction tuning's loss needs a non-linear term and
the preference stage's does not; under a weighted fit both look like a local
sharpening of similar size.** We report both and claim no mechanism difference
between the stages.

A further earlier claim — that no decoding-time operator built on the reference
axis could reach instruction tuning's loss — is **not supported**. On the
pre-registered non-linear statistic it is too strong at the 1B and at Tülu,
unresolved at the 7B, and holds only at the 13B. A non-linear, truncation-like
reference operator is not ruled out.

---

## 6. Two decoding-time interventions (exploratory)

The evidence in this section is thin by design: five prompts at 135M, twelve at
1B, and a single fluency judge (GPT-2, 124M). It is reported as counterexamples to
earlier predictions, not as validated methods for recovering diversity.

### 6.1 Extrapolating along the base–aligned axis

DeRa (Liu et al., ICML 2024) proves that aligned models at varying KL strength are
geometric mixtures differing only in mixing weight:

    z = z_b + λ (z_a − z_b)

λ = 1 recovers the aligned model, λ = 0 the base model, λ < 0 extrapolates past
the base model away from alignment. Published evaluations cover λ ∈ [0, 2].

Diversity in this section is mean pairwise Jaccard, and fluency is median GPT-2
perplexity — an independent judge sharing neither family nor tokeniser with either
model in the mixture.

At **135M** (SmolLM2 base/instruct), the per-prompt paired comparison at matched
diversity gives λ = −1.0 beating temperature by **3.00×, 95% CI [1.58, 5.14]**,
winning on 4/5 prompts. Re-reading the published scored numbers at matched
diversity gives λ < 0 ratios of 0.17–0.37 against the temperature frontier.

At **1B** it does not survive. With the full design of 12 prompts, the paired
comparison gives **1.04×, 95% CI [0.73, 1.50]**, λ winning on 5/12, Wilcoxon
p = 0.970 against an attainable floor of 0.0005. The estimate declines
monotonically from n = 7 onward: 1.16×, 1.08×, 1.04×. The interval excludes the
135M effect size, whose lower bound is 1.58, but does not exclude a modest effect
up to 1.5×.

A direction control — the delta shuffled across the vocabulary at each step,
preserving its norm and the full marginal distribution of its values — is decisive
at 135M (401.3 against temperature's 403.1 at matched diversity, while the true
delta gives 165.4) and **uninformative at 1B**, where both arms have degenerated
into the thousands by the time their diversity matches.

### 6.2 Reference subtraction, and a correction

Writing p_γ ∝ p_align · p_ref^(−γ) attenuates the reference's contribution — the
repair the typicality account recommends, and contrastive decoding with the base
model as amateur.

An earlier draft of this work reported that γ = 0.5 raised judge perplexity to
19,956 and treated this as catastrophic failure. **That comparison was against
T = 1.0, not at matched diversity.** Re-read at matched diversity, γ = 0.5 sits at
**0.98× the temperature frontier** at its own Jaccard. It overshot along the
frontier; it did no off-frontier harm. The original matched-diversity table stopped
short of that diversity level, so the condition was never properly compared.

A pre-registered sweep γ ∈ {0.05, 0.1, 0.2, 0.3, 0.5} on the 135M pair finds
nothing degenerate at small γ. γ = 0.05 is classified temperature-like
(1.21×, CI [0.79, 1.33]) but fragile — only 52% of bootstrap draws fall inside the
temperature curve's interpolation range; γ = 0.1 misses the band by 0.03. With
five prompts the sweep neither falsifies nor establishes the hypothesis.

The measured β on that pair is +0.081 at continuation positions. That is the
*whole* base → Instruct pipeline, since SmolLM2 publishes no intermediate
checkpoints, so it is comparable in size to §5.2's preference-stage β but is not
the same quantity.

### 6.3 A partial explanation

For the preference stage at 7–8B, §5.2 suggests why an operator on the reference
axis would track the temperature frontier rather than beat it: the entropy-relevant
part of that stage's shift is itself close to a temperature.

This cannot be the explanation for the 1B null. There the preference stage's β ≈ 0, and λ acts on
the whole base → Instruct delta, which at the 1B is 66% instruction tuning — whose costly
component looks non-linear under an unweighted fit and temperature-like under a
weighted one. We offer the account as a
hypothesis for 7–13B and note that the 1B result is unexplained.

---

## 7. What we got wrong

Four earlier conclusions in this work were overturned, by four different causes.
We list them with their causes because a vague account of one's own errors is
worth less than a precise one.

| earlier claim | what overturned it |
|---|---|
| "Instruction tuning carries ~99% of the entropy loss" | position convention |
| A 1B → 7B jump in the preference share | position convention |
| "Reference subtraction fails catastrophically" | comparison baseline (T = 1.0 instead of matched diversity) |
| "No reference-axis operator can reach SFT's loss" | operationalisation (linear-only fit of typicality) |

Two further claims fell to ordinary causes: "instruction tuning does all of it" as
a *general* claim was refuted by measuring a second pipeline, and a reported
negative sign on the 1B preference-delta ψ did not survive a change of numerical
backend.

Two analytical predictions also failed, in opposite directions.

**λ < 0 was predicted catastrophic.** An analytical model predicted λ < 0 would be
harmful at any scale, because λ rescales the whole reward and reversing it reverses
the quality term with the typicality term. At 135M, generation showed the opposite.
The model treated the reference as an abstract distribution over items with
independently drawn quality; real base models encode grammaticality, so both
endpoints of the λ axis are fluent language models and the segment between them
lies inside the manifold of well-formed text.

**A ψ threshold was derived and does not hold.** The same model was used to derive
a crossover ψ* ≈ 0.71–0.83 below which reference subtraction should lose to
temperature, and three earlier drafts compared measured ψ against it. ψ* is not a
constant: holding the model's other parameters fixed and varying only the
matched-diversity target moves it from 0.000 to 1.000, and the published value is
the most demanding point on that curve. The model is also off-regime — its aligned
distribution collapses 47-fold against its reference, where the measured collapse
is 1.5–2.2× under continuation scoring (1.51–3.22× under prompt-final) — and
rebuilt at measured peakedness and collapse it yields ψ* between 0.000 and 0.424,
below several measured values. **We withdraw the threshold and make no
decision-theoretic claim.** ψ is reported as an absolute magnitude throughout.

The assumption behind both failures — quality independent of typicality — is common
in this literature and reverses conclusions.

---

## 8. Related work

**Decoding-time model composition** is mature and has been used almost exclusively
to add capability. Contrastive decoding maximises expert-minus-amateur
log-likelihood to suppress generic output. Proxy-tuning adds a small-model tuning
delta to a large base model. DeRa proves the geometric-mixture structure and sweeps
λ ∈ [0, 2]. None evaluates against diversity metrics and none explores λ < 0.

**The typicality-bias literature** proposes prompt-level interventions: verbalized
sampling, diverse personas, chain-of-thought differentiation. These require the
user to want diversity, which matters given that the collective-diversity loss
Doshi and Hauser identify is a social dilemma no individual has an incentive to
solve. Related work on preference collapse (Xiao et al., 2024) shows minority
preferences being virtually disregarded under standard alignment objectives.

**Stage-resolved analysis.** Kirk et al. compare SFT against RLHF on generalisation
and diversity but do not decompose a single pipeline's entropy budget across its
stages. We have not found a cross-pipeline comparison of such a decomposition, nor
a stage comparison that reports the position convention of §2.

---

## 9. Limitations

1. **Four pipelines, two base families.** The shared-shift result replicates within
   two lineages that use related instruction mixtures. The OLMo 7B and 13B share
   one release, so their agreement is a within-release scale comparison, not an
   independent replication. The unit of inference is the prompt: 36 prompts in six
   task families, however many positions each contributes.
2. **The lineage test rules out ancestry, not data.** Both families train on
   Tülu-3-style mixtures, so shared training data remains a live explanation.
3. **β was measured at the 1B, 7B and Tülu-8B**, on 400 sampled positions, but not
   at the 13B. The 1B preference stage is the one place it is ≈ 0. The R² and Shapley results in §5.1 cover all four models.
4. **"Typicality" is operationalised** as a linear and piecewise-linear function of
   reference log-probability. It is one operationalisation of a claim the source
   literature does not state formally. §5.2 shows that two analyst choices — linear
   versus non-linear, and weighted versus unweighted — each reverse a conclusion.
5. **One judge.** GPT-2 perplexity measures fluency, not whether an answer is good,
   and it rewards predictable text, biasing against diverse conditions. Human
   evaluation is absent and is the limitation we would most want addressed.
6. **The γ sweep is underpowered** at five prompts and resolves nothing.
7. **The 1B λ null is unexplained** (§6.3).
8. **Scale.** 135M, 1B, 7B, 8B, 13B. Nothing here speaks to frontier scale.

---

## 10. Conclusion

Across four pipelines, alignment's diversity cost is carried by a small part of the
alignment shift. A typicality-aligned component accounting for at most about a
third of a stage's shift carries nearly all of its entropy loss under the
probability-weighted fit, at both stages, in every model measured and across the
specifications we tried; under an unweighted fit its share of the loss is smaller
and, at the 1B's instruction tuning, unstable in sign. Larger than that
component, though itself a minority of the shift, is a shared direction: measurable
at every stage of every model, surviving a partial correlation against reference
typicality, with null controls at 0.008 and 0.004. Its context-invariance depends
on stage and scale together, and in opposite directions: within OLMo, instruction
tuning rises with scale (0.092, 0.320, 0.445 at 1B, 7B and 13B) while the
preference stage falls (0.558, 0.489, 0.386). It is moderately context-invariant
everywhere — ρ_reg between 0.26 and 0.56 — except for instruction tuning at the
1B. Squared, that is at most roughly 0.31 of the shift — the square of a mean
correlation, so an approximation rather than a bound — and it is not a context-independent
direction so much as a partially shared one. Most of each stage's shift is
accounted for by neither axis, we do not characterise what the remainder is, and
it is not inert — at instruction tuning in the 1B it raises entropy by enough to
offset a third of the loss (§5.1).

Those findings suggest a hypothesis about larger scales, which this work does not
test: an operator on the reference axis acts on the axis that carries the entropy,
and at 7–8B the entropy-relevant part of the preference shift is already close to a
temperature, so there would be little for such an operator to recover that
temperature does not. It is a hypothesis and not an account of what we measured. No
decoding-time intervention was run above 1B. It cannot be the explanation of the 1B
null, where the preference stage's β ≈ 0 and λ acts mostly on instruction tuning
(§6.3); that null stays unexplained, and the same operator beat temperature
threefold at 135M. The results also raise a question this work does not answer. The
field measures alignment's effect through diversity metrics, and most of what
alignment does to the next-token distribution does not appear in them.

Underneath sits a methodological result we did not expect to be consequential. The
same Tülu-3-8B checkpoints yield preference-stage shares of 29% and 53% depending
on where the distribution is scored, and the 1B model's conventional measurement
sits 18–28 points from every alternative. Two of this work's own earlier
conclusions were reversed by that choice alone. Stage attribution is not
reproducible without stating it.

---

## Appendix A — Reproduction

| file | contents |
|---|---|
| `dump_logprobs_mps.py` | per-checkpoint log-prob dumps, single-checkpoint loading |
| `analyse_psi_family.py` | stage-resolved ψ and entropy, bootstrap over prompts |
| `register_v2.py` | shared-shift geometry, both arms, null controls |
| `decomp.py` | counterfactual Shapley entropy decomposition |
| `decomp_resid.py` | non-linear typicality, cross-fitted shared directions |
| `robustness.py` | specification grid for §5.1 (weighting × plausible threshold × held-out shared direction), R² and Shapley shares under the same fit, per-family and leave-one-family-out shares (exploratory, v1.0) |
| `check_tokenizers.py` | token-ID-level tokeniser identity verification |
| `grind.py`, `matched6.py` | generation sweep and matched-diversity scoring |
| `gen_scale.py` (mode `ctrl`), `pilot-135m/gen_ctrl.py` | shuffled-delta direction control, at scale and at 135M |
| `exp1e_peakedness.py` | the peakedness probe that withdrew the ψ\* threshold (§7) |
| `pilot135m_port/` | 135M γ sweep and β measurement, ported |
| `pilot-135m/` | the original 135M pilot, including `exp1d_misspec.py` (the withdrawn ψ\* threshold) and `stats.py`, the source of the 3.00× figure |
| `logs/` | the run scripts and evaluation outputs behind the reported numbers: the 135M frontier, the lineage statistic, the pre-registered 13B evaluation, the R² intervals and the whose-text offset check |
| `logs/inline_checks.py` | reproduces the figures that were originally computed inline rather than by a script — the sampled β and its sign rate, the 400-position R² table, the CI widths, the paired totals, the branching ratios, the whose-text base entropies and the plausible-set sizes |

Six pre-registrations, each hashed before its data existed, are released with the code
at https://github.com/amitd01/axisalignment-paper: the 13B scale test, the register test, the lineage test, the entropy
decomposition, the residual test and the 135M γ sweep. Two carry dated notes
appended after the fact; the hashes cover the original text and the notes state
what changed and when.

| pre-registration | written (IST) | already seen when written | note appended | outcome | standing |
|---|---|---|---|---|---|
| 13B scale test | 2026-10-13 20:20 | 1B and 7B prompt-final shares | 21:21, before any 13B dump: comparison basis moved to continuation positions | trend continues, +15.6 [+12.7, +18.6] (§3.1) | confirmatory, amended |
| register test | 2026-10-13 20:22 | no register data | — | ρ_reg > ψ, all four (§4.2) | confirmatory |
| lineage test | 2026-10-13 22:20 | same-lineage value −0.19 | — | −0.194, transfer (§4.4) | confirmatory |
| entropy decomposition | 2026-10-13 22:25 | stage shares; no decomposition | — | head > 50%: not supported; typicality > shared: holds (§5) | confirmatory |
| residual test | 2026-10-13 23:13 | decomposition shares | before the run: position-specific claim withdrawn | non-linear claim too strong at 1B, Tülu (§5.2) | confirmatory for the non-linear question; ε share descriptive |
| 135M γ sweep | 2026-10-13 23:28 | published 135M scores | — | unresolved (§6.2) | confirmatory, underpowered |

Not pre-registered, and reported as exploratory: the variance shares (R²) of §5.1
and the specification grid in `robustness.py`. The adoption of base-written
continuations as the primary convention for stage shares (§2.2) is the 13B row's
appended note.

Seeds: 20261012 (1B), 20260803 (135M). N = 6 samples, 32 new tokens, top-p 0.95
held constant. Fixed seeds do not guarantee bit-identical output across hardware;
batch-invariant kernels were not attempted.

**Consistency check.** T = 1.0, λ = +1.0 and γ = 0 are the same distribution by
construction and returned identical metrics on the original cloud run (1B: Jaccard
0.0746, ppl 58.8). It was not re-run on the hardware used for §§3–5.

**Metrics.** Entropy is the diversity measure throughout §§2–5. §6 uses mean
pairwise Jaccard; distinct-n saturates at 1.000 by T = 1.6 at these scales and is
reported only for comparability with prior work. Perplexity is a median across
samples.

## Appendix B — Key references

- Li et al. (2023), *Contrastive Decoding*, arXiv:2210.15097
- Liu et al. (2024), *Tuning Language Models by Proxy*, arXiv:2401.08565
- Liu et al. (2024), *Decoding-time Realignment of Language Models*, arXiv:2402.02992
- Kirk et al. (2024), *Understanding the Effects of RLHF on LLM Generalisation and Diversity*, arXiv:2310.06452
- Zhang et al. (2025), *Verbalized Sampling*, arXiv:2510.01171
- Xiao et al. (2024), *Preference Collapse and Matching Regularization*, arXiv:2405.16455
- Doshi & Hauser (2024), *Science Advances* 10(28)
- Sourati, Ziabari & Dehghani (2026), *Trends in Cognitive Sciences*
- Lambert et al. (2024), *Tülu 3*, arXiv:2411.15124
- OLMo 2 (Allen Institute for AI), base/SFT/DPO/Instruct checkpoints
