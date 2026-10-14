"""
Experiment 2: Measure psi on a real base/aligned pair.

Theory (exp 1d) says decontamination beats temperature only when the reference
model's log-probability explains most of the preference bias. Call that share
psi. This script estimates it empirically.

Method
------
For each token position over a corpus of prompts and continuations, compute:
    delta(v)   = log p_aligned(v | ctx) - log p_base(v | ctx)     [the alignment delta]
    ref(v)     = log p_base(v | ctx)                              [typicality]

If the alignment delta were purely a typicality reweighting, delta would be a
deterministic function of ref, and R^2 of delta ~ ref would be 1. If alignment
adds information orthogonal to typicality, R^2 falls.

psi_hat = sqrt(R^2) of the per-position regression of delta on ref.

This is the decisive measurement. It decides which operator is correct, and it
has never been reported.
"""
import os, json, math
os.environ.setdefault("HF_HOME", "/home/claude/hf")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(1)
torch.manual_seed(20260803)

BASE = "HuggingFaceTB/SmolLM2-135M"
ALIGN = "HuggingFaceTB/SmolLM2-135M-Instruct"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
algn = AutoModelForCausalLM.from_pretrained(ALIGN, dtype=torch.float32).eval()

# Text spanning several registers, so psi is not an artefact of one domain.
TEXTS = [
    "List some unusual uses for a brick. One unusual use is to",
    "List some unusual uses for a paperclip. You could use it to",
    "Unusual uses for an old newspaper include using it to",
    "The most surprising thing about deep sea creatures is that they",
    "Write the opening line of a short story about a lighthouse keeper who",
    "A good way to explain entropy to a child is to say that",
    "The main argument against the proposal is that it",
    "In my opinion, the best approach to this problem would be to",
    "Describe a colour to someone who has never been able to see. You might say",
    "The reason old bridges were built with arches is because",
    "An unconventional solution to urban traffic congestion would be to",
    "What people usually get wrong about learning a language is that they",
]


@torch.no_grad()
def logprobs(model, ids):
    out = model(ids).logits[0].float()
    return torch.log_softmax(out, dim=-1)


records = []
per_text = []

for t in TEXTS:
    ids = tok(t, return_tensors="pt")["input_ids"]
    lb = logprobs(base, ids)   # [T, V]
    la = logprobs(algn, ids)
    # use the final position's full next-token distribution
    for pos in range(max(1, lb.shape[0] - 6), lb.shape[0]):
        b = lb[pos].numpy()
        a = la[pos].numpy()
        d = a - b
        # restrict to the plausible set: tokens the base gives non-trivial mass.
        # Modelling the delta over 49k tokens is dominated by numerical noise
        # in the tail, which is not where generation happens.
        keep = b > np.log(1e-6)
        if keep.sum() < 50:
            continue
        bb, dd = b[keep], d[keep]
        r = np.corrcoef(bb, dd)[0, 1]
        records.append(dict(text=t[:40], pos=int(pos), n_tokens=int(keep.sum()),
                            r=float(r), r2=float(r ** 2)))
    per_text.append(t)

r2s = np.array([x["r2"] for x in records])
rs = np.array([x["r"] for x in records])
ns = np.array([x["n_tokens"] for x in records])

print("=" * 84)
print("EXPERIMENT 2 -- structure of the alignment delta on a real model pair")
print(f"  base    : {BASE}")
print(f"  aligned : {ALIGN}")
print("=" * 84)
print(f"\npositions analysed: {len(records)}")
print(f"plausible-set size: median {np.median(ns):.0f} tokens "
      f"(range {ns.min()}-{ns.max()})")
print(f"\ncorrelation of alignment delta with base log-prob:")
print(f"  mean r    = {rs.mean():+.4f}   (sd {rs.std():.4f})")
print(f"  mean R^2  = {r2s.mean():.4f}")
print(f"  median R^2= {np.median(r2s):.4f}")
print(f"  psi_hat   = sqrt(mean R^2) = {math.sqrt(r2s.mean()):.4f}")
print(f"  R^2 range = {r2s.min():.4f} .. {r2s.max():.4f}")

# ------------------------------------------------------ pooled regression
allb, alld = [], []
for t in TEXTS:
    ids = tok(t, return_tensors="pt")["input_ids"]
    lb = logprobs(base, ids)
    la = logprobs(algn, ids)
    pos = lb.shape[0] - 1
    b = lb[pos].numpy(); d = (la[pos] - lb[pos]).numpy()
    keep = b > np.log(1e-6)
    allb.append(b[keep]); alld.append(d[keep])
allb = np.concatenate(allb); alld = np.concatenate(alld)
rp = np.corrcoef(allb, alld)[0, 1]
print(f"\npooled across prompts (final position only):")
print(f"  n = {len(allb)},  r = {rp:+.4f},  R^2 = {rp**2:.4f}, "
      f"psi_hat = {abs(rp):.4f}")
slope, intercept = np.polyfit(allb, alld, 1)
print(f"  delta ~= {slope:+.4f} * log_p_base {intercept:+.4f}")

print("\n" + "=" * 84)
print("DECISION RULE (from exp 1d)")
print("=" * 84)
print("  Decontamination beats temperature only if psi > ~0.71-0.83.")
psi = abs(rp)
if psi > 0.83:
    verdict = "DECONTAMINATION favoured across the tested rho range."
elif psi > 0.71:
    verdict = "MARGINAL. Favoured only at higher quality/typicality correlation."
else:
    verdict = "TEMPERATURE favoured. The delta is largely orthogonal to typicality."
print(f"  measured psi_hat = {psi:.4f}")
print(f"  -> {verdict}")

json.dump(dict(records=records, pooled_r=float(rp), psi_hat=float(psi),
               slope=float(slope), intercept=float(intercept),
               mean_r2=float(r2s.mean())),
          open("/home/claude/results_exp2.json", "w"), indent=1)
print("\nsaved -> results_exp2.json")
