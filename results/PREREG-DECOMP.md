# Pre-registration — where does each stage's entropy loss come from? (head/body, typicality/shared-shift)

Written 2026-10-13 22:25:56 IST, before any decomposition code was written or run. Analysis-only on
existing arm-B arrays (code/dumps/register/{olmo,olmo7b,tulu}_B_*.npy, base-written continuations).

## Method (exact, counterfactual, Shapley)
Per position i, ref log-probs b, shift d = tgt - ref. A counterfactual applies a subset of d's parts:
q_S = softmax(b + sum_{k in S} d_k). Player k's Shapley value is its average marginal effect on H(q) over all
orderings; the values sum exactly to H(tgt) - H(ref). Averaged within prompt, bootstrapped over 36 prompts.
1. REGION players: head = smallest set of ref-top tokens covering >= 50% ref mass; body = rest of plausible
   (ref p > 1e-6); outside = everything else.
2. COMPONENT players, from a per-position fit over the plausible set d ~ a + beta*b + gamma*g_-p
   (g_-p = leave-one-prompt-out mean shift, as in register_v2):
   typicality = beta*b;  shared-shift = gamma*g_-p;  residual = a + eps (plausible set);  outside = d off-set.
   Fitted two ways: unweighted (matches psi/rho_reg plausible) and ref-probability-weighted (matches psi_w).

## Hypothesis under test (from the paper owner)
DPO's entropy loss is head-driven and carried by the typicality component, while its shared-shift lives in the body.
Supported for SFT->DPO in a model if: head share of the entropy loss > 50% AND typicality share (weighted fit)
> shared-shift share, with bootstrap CIs not overlapping on that comparison.
If supported, the primary claim narrows: typicality is roughly right about the head; the refutation holds for the body.
Same statistics reported for base->SFT and DPO->final, without prediction. Any outcome reported.
