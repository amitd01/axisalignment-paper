"""EXPLORATORY (post hoc; run after the decompositions, to answer a framing question). Full-position version of the
400-sample R^2 table: share of each stage's shift variance over the plausible set explained by typicality.
  unw  = linear + piecewise-linear (hinges at 20/40/60/80th pct of ref), unweighted   (pairs with merged Shapley shares)
  w    = linear only, ref-probability-weighted                                        (pairs with the pre-registered weighted fit)
Median over positions; 95% CI by resampling prompts (positions pooled within resampled prompts), 2,000 draws."""
import json, math, numpy as np
D = "code/dumps/register"; T = math.log(1e-6)
def r2s(fam, a, t):
    rows = json.load(open(f"{D}/{fam}_cont.json"))["rows"]
    po = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows])
    R = np.load(f"{D}/{fam}_B_{a}.npy", mmap_mode="r"); X = np.load(f"{D}/{fam}_B_{t}.npy", mmap_mode="r")
    U, W = np.full(len(po), np.nan), np.full(len(po), np.nan)
    for i in range(len(po)):
        b = R[i].astype(float); d = X[i].astype(float) - b; S = b > T
        if S.sum() < 20: continue
        bs, ds = b[S], d[S]; v = ds.var()
        if v < 1e-12: continue
        k = np.percentile(bs, [20, 40, 60, 80])
        M = np.stack([np.ones_like(bs), bs] + [np.maximum(0, bs - q) for q in k], 1)
        U[i] = 1 - ((ds - M @ np.linalg.lstsq(M, ds, rcond=None)[0]) ** 2).mean() / v
        p = np.exp(bs - bs.max()); w = p / p.sum(); sw = np.sqrt(w); M2 = np.stack([np.ones_like(bs), bs], 1)
        c = np.linalg.lstsq(M2 * sw[:, None], ds * sw, rcond=None)[0]; m = (w * ds).sum()
        W[i] = 1 - (w * (ds - M2 @ c) ** 2).sum() / (w * (ds - m) ** 2).sum()
    rng = np.random.default_rng(0); groups = [np.where(po == j)[0] for j in range(36)]
    out = {}
    for name, A in [("unw", U), ("w", W)]:
        bs_ = [np.nanmedian(A[np.concatenate([groups[j] for j in rng.integers(0, 36, 36)])]) for _ in range(2000)]
        out[name] = (float(np.nanmedian(A)), float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5)), int(np.isfinite(A).sum()))
    return out
res = {}
for fam in ["olmo", "olmo7b", "tulu", "olmo13b"]:
    for a, t in [("base", "sft"), ("sft", "dpo")]:
        o = r2s(fam, a, t); res[f"{fam} {a}->{t}"] = o
        print(f"{fam:<8}{a}->{t:<4}  unw lin+nonlin R2 {100*o['unw'][0]:.1f}% [{100*o['unw'][1]:.1f}, {100*o['unw'][2]:.1f}]   "
              f"weighted linear R2 {100*o['w'][0]:.1f}% [{100*o['w'][1]:.1f}, {100*o['w'][2]:.1f}]   n={o['unw'][3]}", flush=True)
json.dump(res, open("results/results_r2_exploratory.json", "w"), indent=1)
