# Pre-registration — cross-lineage register test

Written 2026-10-13 22:20:44 IST, before the test's code or data existed. Queued to run AFTER the OLMo-13B chain.

## Question
On final-model-written continuations, the SFT stage's entropy falls by more than the base model's
(extra -0.19 [-0.23,-0.16] nats at OLMo-1B; results/results_register_olmo{,@inst}.json). But the final model
descends from that SFT checkpoint, so this may be lineage self-familiarity rather than a shared register.

## Design
Continuations written by allenai/OLMo-2-1124-7B-Instruct (raw prompt; 6 x 32 tokens; T=1.0, top-p 0.95,
seed 20261012), teacher-forced through the four OLMo-2-0425-1B stages. 1B-SFT is no ancestor of the writer.
Tokeniser identity 1B <-> 7B-Instruct verified (0 ID-map diffs, identical prompt encodings).
Compared against the 1B base-written run (results_register_olmo.json), same 36 prompts.

## Statistic
X = [H_1B-SFT(7B-inst text) - H_1B-SFT(1B-base text)] - [H_1B-base(7B-inst text) - H_1B-base(1B-base text)]
(prompt-level means, bootstrap over the 36 prompts, 20k, seed 0). Same-lineage reference value: -0.19.

## Reading
- X < 0 with 95% CI excluding 0  -> cross-lineage register transfer: the SFT-installed concentration also
  applies to aligned text from a different-scale, different-release model of the same family.
- X CI includes 0                -> the same-lineage effect is self-familiarity; the entropy leg of the
  register claim collapses and is reported as such (rho_reg carries the register claim independently).
- Magnitude is reported against -0.19 either way; no threshold for "partial" is set.
Secondary, reported without prediction: DPO and final-stage changes; DPO share on this text.
