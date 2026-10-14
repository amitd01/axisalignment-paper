"""Extended entropy decomposition (pre-registered: results/PREREG-RESIDUAL.md, including its appended note).

Per position, over the plausible set:  d = a + beta*ref + f(ref) + [gamma*g_-p + sum_k c_k u_k] + eps
  f(ref)  piecewise-linear in ref: hinges at the 20/40/60/80th percentiles of ref in the plausible set
  u_k     top-8 principal directions of the shift, CROSS-FITTED (estimated on the other half of prompts)
Exact Shapley attribution of the entropy change over {linear typ, nonlinear typ, shared, eps, outside}.

Usage: python decomp_resid.py <family> [...]
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomp import D, RES, LOGP_PLAUS, shapley

PAIRS = [("base", "sft"), ("sft", "dpo")]
K_PC = 8
NAMES = ["lin_typ", "nonlin_typ", "shared", "eps", "outside"]
MERGE = os.environ.get("MERGE") == "1"          # if set, lin_typ holds the merged typicality share, nonlin_typ = 0


def run(family):
    rows = json.load(open(f"{D}/{family}_cont.json"))["rows"]
    prompt_of = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows]).astype(int)
    P = prompt_of.max() + 1
    half = prompt_of % 2
    out = {}
    for a, t in PAIRS:
        ref = np.load(f"{D}/{family}_B_{a}.npy", mmap_mode="r")
        tgt = np.load(f"{D}/{family}_B_{t}.npy", mmap_mode="r")
        N, V = ref.shape
        Gp = np.zeros((P, V)); cnt = np.zeros(P)
        for i in range(N):
            Gp[prompt_of[i]] += tgt[i].astype(np.float64) - ref[i]; cnt[prompt_of[i]] += 1
        Gt, Nt = Gp.sum(0), cnt.sum()
        # common token set for PCA: plausible under ref in >= 5% of positions
        plaus_frac = np.zeros(V)
        for s in range(0, N, 500):
            plaus_frac += (np.asarray(ref[s:s + 500]) > LOGP_PLAUS).sum(0)
        U = np.where(plaus_frac / N >= 0.05)[0]
        # cross-fitted principal directions of the shift on U: PCs from half h are applied to half 1-h
        PCs = {}
        for h in (0, 1):
            idx = np.where(half == h)[0]
            M = np.stack([(tgt[i, U].astype(np.float64) - ref[i, U]) for i in idx])
            M -= M.mean(0)
            _, _, Vt = np.linalg.svd(M, full_matrices=False)
            PCs[1 - h] = Vt[:K_PC]                      # used for positions in the OTHER half
        acc = {k: [[] for _ in range(P)] for k in NAMES + ["dH", "err"]}
        for i in range(N):
            p = prompt_of[i]
            b = ref[i].astype(np.float64); d = tgt[i].astype(np.float64) - b
            g = (Gt - Gp[p]) / (Nt - cnt[p])
            S = b > LOGP_PLAUS; outm = ~S
            bs = b[S]
            knots = np.percentile(bs, [20, 40, 60, 80]) if bs.size >= 10 else np.array([])
            hinges = [np.maximum(0.0, bs - k) for k in knots]
            u_full = np.zeros((K_PC, V)); u_full[:, U] = PCs[half[i]]
            cols = [np.ones_like(bs), bs] + hinges + [g[S]] + [u_full[k, S] for k in range(K_PC)]
            X = np.stack(cols, 1)
            c = np.linalg.lstsq(X, d[S], rcond=None)[0]
            nh = len(hinges)
            lin = np.zeros(V); non = np.zeros(V); sh = np.zeros(V); eps = np.zeros(V)
            lin[S] = c[1] * bs
            if nh:
                non[S] = X[:, 2:2 + nh] @ c[2:2 + nh]
            sh[S] = X[:, 2 + nh:] @ c[2 + nh:]
            eps[S] = d[S] - lin[S] - non[S] - sh[S]       # includes the intercept
            if MERGE:   # exploratory: linear + nonlinear typicality as ONE player (they are collinear)
                phi_m, dH = shapley([lin + non, sh, eps, d * outm], b)
                phi = np.array([phi_m[0], 0.0, phi_m[1], phi_m[2], phi_m[3]])
            else:
                phi, dH = shapley([lin, non, sh, eps, d * outm], b)
            for k, v in zip(NAMES, phi):
                acc[k][p].append(v)
            acc["dH"][p].append(dH); acc["err"][p].append(abs(phi.sum() - dH))
        pm = {k: np.array([np.mean(v) for v in vs]) for k, vs in acc.items()}
        J = np.random.default_rng(0).integers(0, P, (20000, P))
        loss = -pm["dH"]
        rec = dict(entropy_loss=float(loss.mean()), common_tokens=int(len(U)), max_err=float(pm["err"].max()))
        parts = []
        for k in NAMES:
            sh_ = (-pm[k]).mean() / loss.mean(); bs_ = (-pm[k])[J].mean(1) / loss[J].mean(1)
            rec[k] = dict(share=float(sh_), ci=[float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))])
            parts.append(f"{k} {100*sh_:6.1f}% [{100*rec[k]['ci'][0]:6.1f},{100*rec[k]['ci'][1]:6.1f}]")
        out[f"{a}->{t}"] = rec
        print(f"\n{family} {a}->{t}: loss {loss.mean():+.4f} nats, |U|={len(U)}, max Shapley err {rec['max_err']:.1e}")
        print("   " + "   ".join(parts), flush=True)
    tag = "_merged" if MERGE else ""
    json.dump(out, open(f"{RES}/results_decomp_resid{tag}_{family}.json", "w"), indent=1)


if __name__ == "__main__":
    for fam in sys.argv[1:]:
        run(fam)
