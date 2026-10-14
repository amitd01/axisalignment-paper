"""Evaluate PREREG-13B (with its appended note): 13B vs 7B DPO share, paired over the 36 prompts."""
import json, numpy as np
D = "code/dumps/register"; S = ["base", "sft", "dpo", "inst"]
J = np.random.default_rng(0).integers(0, 36, (20000, 36))
def armB(fam):
    rows = json.load(open(f"{D}/{fam}_cont.json"))["rows"]
    p = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows])
    H = {s: np.load(f"{D}/{fam}_B_{s}_H.npy").astype(float) for s in S}
    return {s: np.array([H[s][p == i].mean() for i in range(36)]) for s in S}
def armA(fam):
    return {s: (lambda a: -(np.exp(a) * a).sum(1))(np.load(f"{D}/{fam}_A_{s}.npy").astype(float)) for s in S}
def pf(fam):
    return {s: (lambda a: -(np.exp(a) * a).sum(1))(np.load(f"code/dumps/{fam}_{s}.npy").astype(float)) for s in S}
def summ(E):
    tot = E["base"] - E["inst"]; dpo = E["sft"] - E["dpo"]
    return tot, dpo, dpo[J].mean(1) / tot[J].mean(1)
ci = lambda x, k=1: f"{k*x.mean():.3f} [{k*np.percentile(x,2.5):.3f}, {k*np.percentile(x,97.5):.3f}]"
out = {}
for label, f7, f13 in [("PRIMARY arm B, base-written", lambda: armB("olmo7b"), lambda: armB("olmo13b")),
                       ("secondary arm B, final-model-written", lambda: armB("olmo7b@inst"), lambda: armB("olmo13b@inst")),
                       ("secondary arm A, chat-templated", lambda: armA("olmo7b"), lambda: armA("olmo13b")),
                       ("ARTEFACT ARM raw prompt-final", lambda: pf("olmo7b"), lambda: pf("olmo13b"))]:
    t7, d7, s7 = summ(f7()); t13, d13, s13 = summ(f13())
    diff = s13 - s7
    print(f"\n{label}")
    print(f"   DPO share   7B {100*d7.mean()/t7.mean():5.1f}%   13B {100*d13.mean()/t13.mean():5.1f}%   "
          f"13B-7B {100*(d13.mean()/t13.mean()-d7.mean()/t7.mean()):+5.1f} pts  95% CI [{100*np.percentile(diff,2.5):+.1f}, {100*np.percentile(diff,97.5):+.1f}]")
    print(f"   total loss  7B {ci(t7[J].mean(1))}   13B {ci(t13[J].mean(1))}   13B-7B {ci((t13-t7)[J].mean(1))}")
    print(f"   DPO drop    7B {ci(d7[J].mean(1))}   13B {ci(d13[J].mean(1))}   13B-7B {ci((d13-d7)[J].mean(1))}")
