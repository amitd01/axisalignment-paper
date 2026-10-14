"""Reproduces every paper number that was computed inline during the session (no saved script until now).

Each section mirrors the original inline code, including RNG seeds, so the sampled quantities reproduce exactly.
Needs the per-position arrays in code/dumps/ (regenerate with dump_logprobs_mps.py / register_v2.py; ~53 GB) and,
for section I/H, pilot-135m/results_stats.json and pilot-135m/results_exp3_scored.json.

Usage: python logs/inline_checks.py [project_root]      (default: the directory above this script)
"""
import os, sys, json, math
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/code/dumps/register"
S = ["base", "sft", "dpo", "inst"]
T = math.log(1e-6)
FAMS = ["olmo", "olmo7b", "tulu", "olmo13b"]
J = np.random.default_rng(0).integers(0, 36, (20000, 36))
ci = lambda x: f"{x.mean():+.3f} [{np.percentile(x, 2.5):+.3f}, {np.percentile(x, 97.5):+.3f}]"


def armB(fam):
    rows = json.load(open(f"{D}/{fam}_cont.json"))["rows"]
    p = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows])
    H = {s: np.load(f"{D}/{fam}_B_{s}_H.npy").astype(float) for s in S}
    return {s: np.array([H[s][p == i].mean() for i in range(36)]) for s in S}


def pf(fam):
    out = {}
    for s in S:
        a = np.load(f"{ROOT}/code/dumps/{fam}_{s}.npy").astype(float)
        if fam == "tulu":
            a = a[:, :128256]; a = a - np.logaddexp.reduce(a, 1, keepdims=True)
        out[s] = a
    return out


H_ = lambda a: -(np.exp(a) * a).sum(1)

# ---------------------------------------------------------------- A. sampled beta (paper §5.2)
print("A. sampled beta, 400 positions, rng seed 1 (1B/7B/Tulu, both stages)")
rng = np.random.default_rng(1)
for fam in ["olmo", "olmo7b", "tulu"]:
    for a, t in [("base", "sft"), ("sft", "dpo")]:
        R = np.load(f"{D}/{fam}_B_{a}.npy", mmap_mode="r"); X = np.load(f"{D}/{fam}_B_{t}.npy", mmap_mode="r")
        idx = rng.choice(R.shape[0], 400, replace=False)
        bu, bw, r2u, r2w = [], [], [], []
        for i in idx:
            b = R[i].astype(float); d = X[i].astype(float) - b; m = b > T
            p = np.exp(b - b.max()); p /= p.sum(); w = p[m] / p[m].sum()
            Xm = np.stack([np.ones(m.sum()), b[m]], 1)
            c = np.linalg.lstsq(Xm, d[m], rcond=None)[0]; e = d[m] - Xm @ c; bu.append(c[1])
            r2u.append(1 - e.var() / d[m].var() if d[m].var() > 0 else np.nan)
            sw = np.sqrt(w); c = np.linalg.lstsq(Xm * sw[:, None], d[m] * sw, rcond=None)[0]; e = d[m] - Xm @ c
            mm = (w * d[m]).sum(); bw.append(c[1])
            den = (w * (d[m] - mm) ** 2).sum(); r2w.append(1 - (w * e ** 2).sum() / den if den > 0 else np.nan)
        print(f"  {fam:<7}{a}->{t:<4} beta_unw median {np.median(bu):+.3f} (frac>0 {np.mean(np.array(bu) > 0):.2f})  "
              f"R2 {np.nanmedian(r2u):.3f} | beta_w median {np.median(bw):+.3f} (frac>0 {np.mean(np.array(bw) > 0):.2f})  "
              f"wR2 {np.nanmedian(r2w):.3f}")

# ---------------------------------------------------------------- B. 400-sample R^2 table (paper §5.1)
print("\nB. 400-sample median R^2, linear vs linear+piecewise, rng seed 2")
rng = np.random.default_rng(2)
for fam in FAMS:
    for a, t in [("base", "sft"), ("sft", "dpo")]:
        R = np.load(f"{D}/{fam}_B_{a}.npy", mmap_mode="r"); X = np.load(f"{D}/{fam}_B_{t}.npy", mmap_mode="r")
        lin, non = [], []
        for i in rng.choice(R.shape[0], 400, replace=False):
            b = R[i].astype(float); d = X[i].astype(float) - b; m = b > T
            if m.sum() < 20: continue
            bs, ds = b[m], d[m]; v = ds.var()
            if v < 1e-12: continue
            k = np.percentile(bs, [20, 40, 60, 80])
            for cols, acc in [([np.ones_like(bs), bs], lin),
                              ([np.ones_like(bs), bs] + [np.maximum(0, bs - q) for q in k], non)]:
                Xm = np.stack(cols, 1); c = np.linalg.lstsq(Xm, ds, rcond=None)[0]
                acc.append(1 - ((ds - Xm @ c) ** 2).mean() / v)
        print(f"  {fam:<8}{a}->{t:<4} linear {np.median(lin):.3f}   linear+nonlinear {np.median(non):.3f}")

# ---------------------------------------------------------------- C. precision, totals, branching (paper §2.3, §3.1, §7)
print("\nC. total-loss CI widths, arm-B totals, prompt-final branching ratios")
w_ = lambda x: np.percentile(x, 97.5) - np.percentile(x, 2.5)
tot_pf, tot_b = {}, {}
for fam in FAMS:
    P = {s: H_(v) for s, v in pf(fam).items()}; B = armB(fam)
    tot_pf[fam] = P["base"] - P["inst"]; tot_b[fam] = B["base"] - B["inst"]
    tp, tb = tot_pf[fam][J].mean(1), tot_b[fam][J].mean(1)
    print(f"  {fam:<8} pf width {w_(tp):.3f}  B width {w_(tb):.3f}  ratio {w_(tp) / w_(tb):.1f}x   "
          f"B total {ci(tb)}   pf branching ratio {np.exp(P['base']).mean() / np.exp(P['inst']).mean():.2f}")
for name, T_ in [("prompt-final", tot_pf), ("arm B", tot_b)]:
    print(f"  {name}: OLMo-7B - Tulu-8B total (paired by prompt) {ci((T_['olmo7b'] - T_['tulu'])[J].mean(1))}")
print(f"  arm B: 13B - 1B total (paired by prompt) {ci((tot_b['olmo13b'] - tot_b['olmo'])[J].mean(1))}")

# ---------------------------------------------------------------- D. whose-text, per stage (paper §2.3)
print("\nD. final-written minus base-written entropy, per stage")
for fam in FAMS:
    A_, B_ = armB(fam), armB(fam + "@inst")
    print(f"  {fam:<8}" + "  ".join(f"{s} {ci((B_[s] - A_[s])[J].mean(1))}" for s in S))

# ---------------------------------------------------------------- E. plausible-set size (paper §4.5)
print("\nE. plausible-set size |{v: p_ref > 1e-6}| per position, reference stage of each delta")
for fam in ["olmo", "olmo7b", "tulu"]:
    for s in ["base", "sft", "dpo"]:
        Bm = np.load(f"{D}/{fam}_B_{s}.npy", mmap_mode="r")
        c = np.concatenate([(np.asarray(Bm[i:i + 500]) > T).sum(1) for i in range(0, Bm.shape[0], 500)])
        print(f"  arm B {fam:<7}{s:<5} median {np.median(c):6.0f}  IQR [{np.percentile(c, 25):.0f}, {np.percentile(c, 75):.0f}]")
    for s in ["base", "sft"]:
        c = (np.load(f"{ROOT}/code/dumps/{fam}_{s}.npy") > T).sum(1)
        print(f"  prompt-final {fam:<7}{s:<5} median {np.median(c):6.0f}")

# ---------------------------------------------------------------- F. lineage statistic (paper §4.4)
print("\nF. cross-lineage X = change(stage) - change(base), 1B stages")
A_ = armB("olmo")
for label, fam in [("7B-Instruct text (cross-lineage)", "olmo@x7binst"), ("1B-Instruct text (same lineage)", "olmo@inst")]:
    B_ = armB(fam)
    print(f"  {label}: " + "  ".join(f"X[{s}] {ci(((B_[s] - A_[s]) - (B_['base'] - A_['base']))[J].mean(1))}" for s in ["sft", "dpo", "inst"]))

# ---------------------------------------------------------------- G/H/I. 135M (paper §6)
P135 = f"{ROOT}/pilot-135m"
if os.path.exists(f"{P135}/results_stats.json"):
    s = json.load(open(f"{P135}/results_stats.json"))
    r = np.log(np.array(s["pplA"]) / np.array(s["pplB"]))
    print(f"\nI. 135M lambda=-1.0 vs temperature, per-prompt matched: {np.exp(r.mean()):.2f}x, "
          f"CI [{np.exp(s['ci'][0]):.2f}, {np.exp(s['ci'][1]):.2f}], wins {(r > 0).sum()}/{len(r)}")
if os.path.exists(f"{P135}/results_exp3_scored.json"):
    d = json.load(open(f"{P135}/results_exp3_scored.json")); rows = d if isinstance(d, list) else list(d.values())
    Tp = sorted([(x["jac"], x["pplmed"]) for x in rows if x["op"] == "temp"], reverse=True)
    def front(j):
        for (j1, p1), (j2, p2) in zip(Tp, Tp[1:]):
            if j2 <= j <= j1:
                f = math.log(j1 / j) / math.log(j1 / j2); return math.exp(math.log(p1) + f * math.log(p2 / p1))
    print("H. published 135M points vs temperature frontier at matched Jaccard (ratio = ppl / frontier ppl):")
    for x in rows:
        if x["op"] in ("dec", "lam"):
            fp = front(x["jac"])
            if fp: print(f"  {x['tag']:<6} jac {x['jac']:.4f}  ratio {x['pplmed'] / fp:.2f}")
sweep = f"{ROOT}/code/dumps/135m/results_sweep_scored.json"
if os.path.exists(sweep):
    R = {x["tag"]: x for x in json.load(open(sweep))}
    Tt = ["T10", "T11", "T12", "T13", "T16", "T20"]
    pp = lambda tag, k: np.array(R[tag]["per_prompt"][k])
    def ratio(idx, g):
        tj = np.array([np.nanmean(pp(t, "jac")[idx]) for t in Tt]); tp = np.array([np.nanmean(pp(t, "pplmed")[idx]) for t in Tt])
        gj = np.nanmean(pp(g, "jac")[idx]); gp = np.nanmean(pp(g, "pplmed")[idx])
        o = np.argsort(tj); x, y = np.log(tj[o]), np.log(tp[o])
        if not (x[0] <= np.log(gj) <= x[-1]): return np.nan
        return gp / np.exp(np.interp(np.log(gj), x, y))
    Bs = np.random.default_rng(0).integers(0, 5, (20000, 5))
    print("G. pre-registered 135M gamma sweep, ratio to the new temperature frontier (bootstrap over 5 prompts)")
    for g in ["G005", "G010", "G020", "G030", "G050"]:
        bs = np.array([ratio(b, g) for b in Bs]); ok = ~np.isnan(bs)
        print(f"  {g}: ratio {ratio(np.arange(5), g):.2f}  CI [{np.percentile(bs[ok], 2.5):.2f}, {np.percentile(bs[ok], 97.5):.2f}]  in-range {ok.mean():.2f}")
