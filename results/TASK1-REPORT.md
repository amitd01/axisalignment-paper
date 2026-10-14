# Task 1 report — stage replication (Tulu 3 8B + OLMo-2 7B)

**From:** terminal session, 13 Oct 2026, 19:35 IST. Numbers only — `paper-v3.md` not edited.
**Hardware:** MacBook Pro, Apple M5 Max, 48 GB; torch 2.14.1 MPS, transformers 5.19.0, `.venv` (Python 3.12).
**Protocol:** identical to the OLMo-1B run — 36 prompts × 6 families (`prompts.FLAT`), raw text (no chat
template), final-position next-token log-probs, bf16 weights / fp32 softmax, bootstrap (20k) over prompts.
Scripts: `code/dump_logprobs_mps.py`, `code/analyse_psi_family.py`, `code/check_tokenizers.py`.

## 1. Headline — share of total entropy loss by stage

| model | total drop (nats) | branching | SFT share | DPO share | RLVR/final share |
|---|---|---|---|---|---|
| OLMo-2-0425-**1B** (MPS) | 0.602 | 86.2 → 44.0 (−49%) | **97.0%** [88.2, 110.1] | **−0.6%** [−15.4, 9.1] | 3.6% [−0.9, 8.8] |
| OLMo-2-1124-**7B** (MPS) | 1.123 | 68.6 → 21.3 (−69%) | **69.4%** [62.8, 75.4] | **25.7%** [19.9, 31.9] | 4.9% [4.1, 5.9] |
| Tulu-3-**8B** (MPS) | 0.963 | 72.5 → 30.3 (−58%) | **52.7%** [45.0, 59.6] | **40.8%** [33.0, 50.0] | 6.5% [4.1, 9.1] |
| Tulu-3-8B (CPU, robustness) | 0.970 | 72.3 → 30.3 | 52.5% [44.2, 59.8] | 41.1% [33.1, 50.7] | 6.3% [4.2, 8.9] |

Absolute stage drops: SFT→DPO = −0.004 (1B) / **+0.289** [0.206, 0.378] (OLMo-7B) / **+0.393** [0.270, 0.541] (Tulu-8B).

**Reading (not a conclusion):**
- The 99%/0% attribution does not replicate. It holds only at OLMo-1B.
- Scale vs recipe: within the OLMo family, DPO goes from ~0 nats at 1B to 0.29 nats at 7B → **mostly scale**.
  The residual gap to Tulu-8B (CIs nearly touch, 31.9 vs 33.0) leaves some recipe/base-model contribution.
- Confound: OLMo-2-0425-1B and OLMo-2-1124-7B are different releases (April 2025 vs November 2024 post-training),
  so "same recipe" is close, not identical. A third OLMo point would firm this up — OLMo-2-1124-13B
  publishes all four stages (base stored fp32, 54.9 GB; SFT/DPO/Instruct 27.4 GB — the DPO repo holds both
  safetensors and .bin copies, which is why its listing totals 54.9 GB; ~27 GB each
  resident in bf16, fits in 48 GB); not run.
- SFT is still the largest single stage everywhere; RLVR is small (4–7%) everywhere.
- Tulu CPU and MPS agree on the pattern (shares within 0.3 pp), as agreed: robustness check, not a gate.

## 2. ψ — below threshold on every delta, every region, every model

Max ψ anywhere: 0.50 (OLMo-7B DPO→Instruct, prob-weighted). Threshold 0.71–0.83. **Two-family replication.**

Preference delta (SFT→DPO):

| model | top-10 | top-50 | top-1k | plausible | prob-weighted |
|---|---|---|---|---|---|
| OLMo-1B (MPS) | −0.104 [−0.214, +0.009] | +0.030 | +0.001 | +0.007 | −0.028 |
| OLMo-7B | +0.087 [−0.034, +0.205] | +0.262 | +0.170 | +0.148 [0.108, 0.183] | +0.349 [0.249, 0.443] |
| Tulu-8B | +0.120 [+0.006, +0.232] | +0.278 | +0.234 | +0.221 [0.193, 0.247] | +0.400 [0.291, 0.495] |

At 7–8B the preference delta is **positively** correlated with reference log-prob (weak typicality reweighting,
well under threshold); at 1B it is ~0. So ψ's magnitude replicates; its sign/size is scale-dependent like the
entropy share. The OLMo-1B negative top-10 sign is hardware-fragile (CI excludes zero on 2 of 3 backends) — already
downgraded to "suggestive" in the draft.

## 3. Register diagnostic — NULL (report it, don't drop it)

Probability mass on special/reserved token IDs inside the shared vocab, prompt-final positions:

| model | base | SFT | DPO | final |
|---|---|---|---|---|
| OLMo-1B (IDs 100278–100351) | 5.1e-9 | 2.9e-8 | 4.5e-8 | 4.2e-8 |
| OLMo-7B (same) | 2.7e-9 | 9.3e-9 | 1.8e-8 | 1.7e-8 |
| Tulu-8B (IDs 128000–128255) | 7.0e-6 | 1.0e-5 | 6.3e-6 | 5.6e-6 |

Inconclusive. Mass is negligible at every stage; it rises after SFT in OLMo, but from ~1e-9 to ~1e-8, and in Tulu
it is non-monotonic and falls back below base. With raw prompts and no chat template, special tokens are not
expected at these positions, so this is a weak test, not a refutation. It was the only quantitative route to the
§2.4 learned-register claim tried so far; a chat-templated or continuation-position version would be the stronger test.

## 4. Port validation (OLMo-1B, tolerances fixed in advance)

- `analyse_psi_family.py` vs the original `analyse_psi.py` on the same dumps: max diff **0.00e+00**.
- MPS vs cloud: entropy within ±0.01 on 11/12 values (base→Instruct drop +0.015, acceptable); ψ within ±0.02 on
  22/24 cells. Both misses are on DPO→Instruct (top-10 +0.047, weighted +0.024), the smallest delta.
- Mac CPU vs cloud CPU drift is the same size (max 0.027) on the same cells, so this is bf16 cross-hardware noise,
  not the port. Pattern held on all three backends (SFT share 97.0 / 99.7 / ~99%).

## 5. Protocol notes and provenance

- **Protocol deviation (state as such):** OLMo needed no vocab restriction (all checkpoints 100,352 wide).
  Tulu does: the SFT/DPO/final heads are 128,264 wide (`<pad>` at 128256 plus 7 untokenisable padding rows);
  Llama-3.1-8B is 128,256. Tulu figures use the shared 128,256 IDs, renormalised. Mass on the excluded IDs
  is ≤7e-6 (mean ~4e-7) at every stage, so the restriction does not understate entropy change.
- **Tokeniser identity:** 0 ID-map differences over the shared range and identical encodings of all 36 prompts,
  for both families (`check_tokenizers.py`).
- **Precision:** every checkpoint is loaded bf16, asserted on every parameter after load. Stored formats:
  Tulu SFT/DPO/final and Llama base are bf16 (Tulu DPO is 16 GB bf16, *not* 32 GB fp32 as the brief assumed);
  OLMo-2-1124-7B base is **fp32** (29.2 GB) and its SFT/DPO are pickled `.bin` — all downcast to bf16 on load.
- **Base weights:** `meta-llama/Llama-3.1-8B` (Meta's repo, access granted). Side datum: the ungated
  `unsloth/Meta-Llama-3.1-8B` mirror's shards are sha256-identical to Meta's, and its log-probs are bit-identical
  under this protocol. The mirror does edit config/tokeniser metadata (pad token, `padding_side=left`), which
  matters for batched/padded use only.
- Results JSON: `results/results_psi_{olmo,olmocpu,tulu,tulucpu,olmo7b}.json`. Logs: `logs/`.

## 6. Not done

- Tasks 2–4 (Qwen, 1B sweep, Zephyr) not started.
- No CPU robustness run for OLMo-7B (Tulu CPU already agreed; not a gate).
- Entropy is still measured at prompt-final positions only (the handover's §5 reviewer point).
