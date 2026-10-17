"""
Measure psi across each stage of a full post-training pipeline.

Why this matters
----------------
The pilot measured psi on a base->instruct (SFT) delta. Typicality bias is a
claim about *preference* data, so that was arguably the wrong delta. OLMo 2
publishes base -> SFT -> DPO -> Instruct as separate checkpoints, which lets us
isolate the preference delta (SFT->DPO) from the instruction delta (base->SFT).

psi is the correlation between the per-token alignment delta and the reference
model's own log-probability. If mode collapse were a reweighting toward
reference typicality, psi would be high (theory puts the threshold at ~0.71-0.83).

Memory strategy
---------------
Two 1.5B models at once would not fit in 7GB alongside overhead, so each
checkpoint is loaded alone, its log-prob vectors are written to disk, and the
model is freed before the next is loaded. All delta arithmetic happens afterward
on the saved arrays.

Usage:  python3 psi_pipeline.py <stage>        # stage in STAGES, caches logprobs
        python3 psi_pipeline.py analyse        # computes psi for every delta
"""
import os, sys, json, gc
os.environ.setdefault("HF_HOME", "/home/claude/hf")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(2)

FAMILY = "allenai/OLMo-2-0425-1B"
STAGES = {
    "base":     FAMILY,
    "sft":      FAMILY + "-SFT",
    "dpo":      FAMILY + "-DPO",
    "instruct": FAMILY + "-Instruct",
}
OUT = "/home/claude/scale/lp"
os.makedirs(OUT, exist_ok=True)

# 30 prompts across six task families. The pilot used 12 and was underpowered;
# psi is cheap (forward passes only) so there is no reason to economise here.
PROMPTS = [
    # divergent thinking (the pilot's friendliest case)
    "List some unusual uses for a brick. One unusual use is to",
    "List some unusual uses for a paperclip. You could use it to",
    "Unusual uses for an old newspaper include using it to",
    "Name an unexpected use for a bicycle wheel. You could",
    "An unconventional use for a glass bottle would be to",
    # narrative
    "Write the opening line of a short story about a lighthouse keeper who",
    "Begin a story about a woman who finds a letter addressed to",
    "The first sentence of a novel about a failing orchard reads:",
    "Start a short story that begins in the middle of an argument about",
    "Write an opening that introduces a character whose job is to",
    # explanatory
    "A good way to explain entropy to a child is to say that",
    "The reason old bridges were built with arches is because",
    "Explain why metal feels colder than wood at the same temperature:",
    "The simplest way to describe how vaccines work is that they",
    "Why does bread rise? The short answer is that",
    # argumentative
    "The main argument against the proposal is that it",
    "One serious objection to remote work is that it",
    "The strongest case for congestion pricing is that",
    "Critics of standardised testing argue mainly that",
    "A reasonable counterargument to that position would be that",
    # practical / instructional
    "An unconventional solution to urban traffic congestion would be to",
    "To fix a dripping tap, the first thing to check is",
    "If you want to learn a language quickly, the most effective approach is",
    "The best way to prepare for a difficult conversation is to",
    "When a project is running late, the first thing a manager should do is",
    # descriptive / open
    "Describe a colour to someone who has never been able to see. You might say",
    "What people usually get wrong about learning a language is that they",
    "The most surprising thing about deep sea creatures is that they",
    "The difference between a good question and a bad one is that",
    "What makes a city feel alive is usually",
]

POS_PER_PROMPT = 3          # last N positions, for robustness
PLAUSIBLE = np.log(1e-6)    # tail cutoff; below this is numerical noise


@torch.no_grad()
def cache_stage(stage):
    """Load one checkpoint, save log-prob vectors for all prompts, free it."""
    name = STAGES[stage]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

    rows, meta = [], []
    for pi, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt")["input_ids"]
        lg = model(ids).logits[0].float()
        lp = torch.log_softmax(lg, dim=-1)
        T = lp.shape[0]
        for pos in range(max(0, T - POS_PER_PROMPT), T):
            rows.append(lp[pos].numpy().astype(np.float32))
            meta.append(dict(prompt_idx=pi, pos=int(pos), n_tok=int(T)))
        if pi % 10 == 0:
            print(f"  {stage}: prompt {pi}/{len(PROMPTS)}", flush=True)

    arr = np.stack(rows)
    np.save(f"{OUT}/{stage}.npy", arr)
    json.dump(dict(meta=meta, vocab=int(arr.shape[1]), model=name,
                   tokenizer_vocab=len(tok)),
              open(f"{OUT}/{stage}.json", "w"))
    print(f"  saved {stage}: {arr.shape}  vocab={arr.shape[1]}", flush=True)

    del model, lp, lg
    gc.collect()


def corr(x, y):
    if x.size < 20 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def analyse():
    have = [s for s in STAGES if os.path.exists(f"{OUT}/{s}.npy")]
    L = {s: np.load(f"{OUT}/{s}.npy") for s in have}
    M = {s: json.load(open(f"{OUT}/{s}.json")) for s in have}

    print("=" * 94)
    print("PSI ACROSS A FULL POST-TRAINING PIPELINE --", FAMILY)
    print("=" * 94)
    for s in have:
        print(f"  {s:<9} vocab={L[s].shape[1]:<8} rows={L[s].shape[0]:<5} "
              f"tokenizer_len={M[s]['tokenizer_vocab']}")

    V = min(L[s].shape[1] for s in have)   # common vocab prefix
    for s in have:
        L[s] = L[s][:, :V]
    print(f"\n  using common vocab prefix V={V}")

    # entropy per stage
    print("\n" + "-" * 94)
    print("ENTROPY BY STAGE (nats, mean over positions)")
    print("-" * 94)
    ent = {}
    for s in have:
        p = np.exp(L[s])
        p = p / p.sum(1, keepdims=True)
        e = -(p * np.log(np.clip(p, 1e-30, None))).sum(1)
        ent[s] = e
        print(f"  {s:<9} entropy={e.mean():7.4f}   effective branching={np.exp(e).mean():8.1f}")

    order = [s for s in ["base", "sft", "dpo", "instruct"] if s in have]
    for a, b in zip(order, order[1:]):
        d = ent[a].mean() - ent[b].mean()
        print(f"  {a:>8} -> {b:<9} entropy change {-d:+.4f} nats   "
              f"branching {np.exp(ent[a]).mean():.1f} -> {np.exp(ent[b]).mean():.1f}")

    # psi for every meaningful delta
    print("\n" + "=" * 94)
    print("PSI = corr(delta, reference log-prob).  Theory threshold for")
    print("reference-subtraction to beat temperature: psi > ~0.71-0.83")
    print("=" * 94)
    pairs = [("base", "sft", "instruction delta (comparable to the 135M pilot)"),
             ("sft", "dpo", "PREFERENCE delta -- what typicality bias claims"),
             ("dpo", "instruct", "RLVR delta"),
             ("sft", "instruct", "full post-SFT delta"),
             ("base", "instruct", "end-to-end delta")]
    results = []
    print(f"{'delta':<22} {'top-10':>8} {'top-50':>8} {'top-1k':>8} "
          f"{'plaus':>8} {'pwgt':>8}  note")
    for ref, al, note in pairs:
        if ref not in have or al not in have:
            continue
        row = dict(ref=ref, aligned=al, note=note)
        cells = []
        for label, K in [("top-10", 10), ("top-50", 50), ("top-1k", 1000),
                         ("plaus", None), ("pwgt", "w")]:
            rs = []
            for i in range(L[ref].shape[0]):
                b = L[ref][i]
                dl = L[al][i] - b
                if K == "w":
                    w = np.exp(b)
                    w = w / w.sum()
                    mb = (w * b).sum()
                    md = (w * dl).sum()
                    cov = (w * (b - mb) * (dl - md)).sum()
                    den = np.sqrt((w * (b - mb) ** 2).sum() * (w * (dl - md) ** 2).sum())
                    rs.append(float(cov / den) if den > 0 else np.nan)
                else:
                    idx = np.where(b > PLAUSIBLE)[0] if K is None else np.argsort(-b)[:K]
                    rs.append(corr(b[idx], dl[idx]))
            v = float(np.nanmean(rs))
            row[label] = v
            cells.append(v)
        results.append(row)
        print(f"{ref+'->'+al:<22} " + " ".join(f"{c:>8.3f}" for c in cells) + f"  {note}")

    # pooled regression on the preference delta specifically
    if "sft" in have and "dpo" in have:
        print("\n" + "-" * 94)
        print("POOLED REGRESSION, PREFERENCE DELTA (sft -> dpo), plausible set")
        print("-" * 94)
        bs, ds = [], []
        for i in range(L["sft"].shape[0]):
            b = L["sft"][i]
            idx = np.where(b > PLAUSIBLE)[0]
            bs.append(b[idx]); ds.append((L["dpo"][i] - b)[idx])
        bs = np.concatenate(bs); ds = np.concatenate(ds)
        r = corr(bs, ds)
        sl, ic = np.polyfit(bs, ds, 1)
        print(f"  n={len(bs):,}  r={r:+.4f}  R^2={r**2:.4f}  psi_hat={abs(r):.4f}")
        print(f"  delta ~= {sl:+.4f} * log_p_sft {ic:+.4f}")

    json.dump(results, open("/home/claude/scale/results_psi_pipeline.json", "w"), indent=1)
    print("\nsaved -> results_psi_pipeline.json")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "analyse":
        analyse()
    else:
        cache_stage(a)
