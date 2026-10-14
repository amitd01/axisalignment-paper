"""Where does each stage's entropy loss come from? (pre-registered: results/PREREG-DECOMP.md)

Counterfactual Shapley attribution of H(tgt) - H(ref) at every arm-B continuation position.
A counterfactual applies a subset of the shift's parts: q_S = softmax(ref + sum of parts in S).
Shapley values over the parts sum exactly to the real entropy change.

  REGION    head (ref-top tokens covering >= 50% ref mass) / body (rest of plausible) / outside
  COMPONENT typicality (beta*ref) / shared-shift (gamma*g_-p) / residual / outside, fitted per position
            over the plausible set, unweighted and ref-probability-weighted

Usage: python decomp.py <family> [<family> ...]      (families as in register_v2.py; base-written arm B)
"""
import os, sys, json, math, itertools
import numpy as np

D = os.path.dirname(os.path.abspath(__file__)) + "/dumps/register"
RES = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + "/../results")
PAIRS = [("base", "sft"), ("sft", "dpo"), ("dpo", "inst")]
LOGP_PLAUS = math.log(1e-6)


def ent(z):
    z = z - z.max()
    e = np.exp(z)
    Z = e.sum()
    return math.log(Z) - float((e * z).sum()) / Z


def shapley(parts, b):
    """parts: list of arrays summing to the shift. Returns Shapley values for H(softmax(b + sum S))."""
    n = len(parts)
    v = {}
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            v[S] = ent(b + sum((parts[k] for k in S), np.zeros_like(b)))
    phi = np.zeros(n)
    for k in range(n):
        others = [j for j in range(n) if j != k]
        for r in range(n):
            wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                phi[k] += wgt * (v[tuple(sorted(S + (k,)))] - v[S])
    return phi, v[tuple(range(n))] - v[()]


def fit(d, b, g, w=None):
    X = np.stack([np.ones_like(b), b, g], 1)
    if w is not None:
        sw = np.sqrt(w / w.sum())
        coef = np.linalg.lstsq(X * sw[:, None], d * sw, rcond=None)[0]
    else:
        coef = np.linalg.lstsq(X, d, rcond=None)[0]
    return coef


def run(family):
    rows = json.load(open(f"{D}/{family}_cont.json"))["rows"]
    prompt_of = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows]).astype(int)
    P = prompt_of.max() + 1
    out = {}
    for a, t in PAIRS:
        ref = np.load(f"{D}/{family}_B_{a}.npy", mmap_mode="r")
        tgt = np.load(f"{D}/{family}_B_{t}.npy", mmap_mode="r")
        V = ref.shape[1]
        Gp = np.zeros((P, V)); cnt = np.zeros(P)
        for i in range(ref.shape[0]):
            Gp[prompt_of[i]] += tgt[i].astype(np.float64) - ref[i]; cnt[prompt_of[i]] += 1
        Gt, Nt = Gp.sum(0), cnt.sum()
        acc = {k: [[] for _ in range(P)] for k in
               ["dH", "reg_head", "reg_body", "reg_out", "u_typ", "u_shared", "u_resid", "u_out",
                "w_typ", "w_shared", "w_resid", "w_out", "head_size", "max_err"]}
        for i in range(ref.shape[0]):
            p = prompt_of[i]
            b = ref[i].astype(np.float64); d = tgt[i].astype(np.float64) - b
            g = (Gt - Gp[p]) / (Nt - cnt[p])
            pr = np.exp(b - b.max()); pr /= pr.sum()
            S = b > LOGP_PLAUS
            order = np.argsort(-b)
            k = int(np.searchsorted(np.cumsum(pr[order]), 0.5) + 1)
            head = np.zeros(V, bool); head[order[:k]] = True
            body = S & ~head; outm = ~S
            # region decomposition
            phi_r, dH = shapley([d * head, d * body, d * outm], b)
            errs = [abs(phi_r.sum() - dH)]
            # component decompositions
            res = {}
            for tag, w in [("u", None), ("w", pr[S])]:
                c = fit(d[S], b[S], g[S], w)
                typ = np.zeros(V); sh = np.zeros(V); rs = np.zeros(V)
                typ[S] = c[1] * b[S]; sh[S] = c[2] * g[S]; rs[S] = d[S] - typ[S] - sh[S]
                phi_c, dH2 = shapley([typ, sh, rs, d * outm], b)
                errs.append(abs(phi_c.sum() - dH2)); res[tag] = phi_c
            for name, val in [("dH", dH), ("reg_head", phi_r[0]), ("reg_body", phi_r[1]), ("reg_out", phi_r[2]),
                              ("u_typ", res["u"][0]), ("u_shared", res["u"][1]), ("u_resid", res["u"][2]),
                              ("u_out", res["u"][3]), ("w_typ", res["w"][0]), ("w_shared", res["w"][1]),
                              ("w_resid", res["w"][2]), ("w_out", res["w"][3]), ("head_size", k),
                              ("max_err", max(errs))]:
                acc[name][p].append(val)
        pm = {k: np.array([np.mean(v) for v in vs]) for k, vs in acc.items()}
        rng = np.random.default_rng(0); J = rng.integers(0, P, (20000, P))
        loss = -pm["dH"]                          # entropy LOSS, positive when entropy falls
        rec = dict(entropy_loss=float(loss.mean()), median_head_size=float(np.median(np.concatenate(acc["head_size"]))),
                   max_shapley_error=float(pm["max_err"].max()))
        line = []
        for kname in ["reg_head", "reg_body", "reg_out", "u_typ", "u_shared", "u_resid", "u_out",
                      "w_typ", "w_shared", "w_resid", "w_out"]:
            share = (-pm[kname]).mean() / loss.mean()
            bs = (-pm[kname])[J].mean(1) / loss[J].mean(1)
            rec[kname] = dict(share=float(share), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            line.append(f"{kname} {100*share:6.1f}% [{100*rec[kname]['ci'][0]:6.1f},{100*rec[kname]['ci'][1]:6.1f}]")
        # pre-registered comparison: weighted typicality vs shared-shift, paired over prompts
        diff = ((-pm["w_typ"])[J].mean(1) - (-pm["w_shared"])[J].mean(1)) / loss[J].mean(1)
        rec["w_typ_minus_shared"] = dict(share=float(((-pm["w_typ"]).mean() - (-pm["w_shared"]).mean()) / loss.mean()),
                                         ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
        out[f"{a}->{t}"] = rec
        print(f"\n{family} {a}->{t}: entropy loss {loss.mean():+.4f} nats, median head size "
              f"{rec['median_head_size']:.0f} tokens, max Shapley sum error {rec['max_shapley_error']:.1e}")
        print("   REGION     " + "   ".join(line[:3]))
        print("   UNWEIGHTED " + "   ".join(line[3:7]))
        print("   WEIGHTED   " + "   ".join(line[7:]))
        print(f"   weighted typ - shared: {100*rec['w_typ_minus_shared']['share']:+.1f} pts "
              f"[{100*rec['w_typ_minus_shared']['ci'][0]:+.1f}, {100*rec['w_typ_minus_shared']['ci'][1]:+.1f}]", flush=True)
    json.dump(out, open(f"{RES}/results_decomp_{family}.json", "w"), indent=1)


if __name__ == "__main__":
    for fam in sys.argv[1:]:
        run(fam)
