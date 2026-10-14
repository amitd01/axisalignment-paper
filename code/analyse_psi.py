"""Measure psi for each training-stage delta, across regions of the distribution.

psi is defined as the share of the alignment delta explained by the REFERENCE
model's own log-probability. Typicality bias predicts psi should be high: if
alignment reweights toward what the base model already found likely, the delta
is largely a function of reference log-prob.

OLMo-2-1B publishes base / SFT / DPO / Instruct separately, so we can isolate:
  base -> SFT        instruction tuning (what the 135M pilot actually measured)
  SFT  -> DPO        the TRUE preference delta -- the claim typicality bias makes
  DPO  -> Instruct   OLMo-2's RLVR stage
  base -> Instruct   the full pipeline end to end

Bootstrap CIs are over the 36 prompts.
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import FLAT

D = os.path.dirname(os.path.abspath(__file__)) + "/dumps"
load = lambda t: np.load(f"{D}/{t}.npy")

base, sft, dpo, inst = load("olmo_base"), load("olmo_sft"), load("olmo_dpo"), load("olmo_inst")
fams = [f for f, _ in FLAT]

PAIRS = [
    ("base -> SFT",       base, sft,  "instruction tuning"),
    ("SFT -> DPO",        sft,  dpo,  "TRUE preference delta"),
    ("DPO -> Instruct",   dpo,  inst, "RLVR stage"),
    ("base -> Instruct",  base, inst, "full pipeline"),
]

REGIONS = [("top-10", 10), ("top-50", 50), ("top-100", 100),
           ("top-1000", 1000), ("plausible", None)]


def ent(lp):
    p = np.exp(lp)
    return float(-(p * lp).sum())


def psi_region(ref, tgt, k):
    """Per-prompt correlation of delta with reference log-prob."""
    out = []
    for i in range(ref.shape[0]):
        b, d = ref[i], tgt[i] - ref[i]
        idx = np.argsort(-b)[:k] if k else np.where(b > math.log(1e-6))[0]
        if len(idx) < 5:          # was 20, which silently dropped the top-10 region
            continue
        bb, dd = b[idx], d[idx]
        if bb.std() < 1e-9 or dd.std() < 1e-9:
            continue
        out.append(np.corrcoef(bb, dd)[0, 1])
    return np.array(out)


def wpsi(ref, tgt):
    """Probability-weighted correlation over the full vocab."""
    out = []
    for i in range(ref.shape[0]):
        b, d = ref[i], tgt[i] - ref[i]
        w = np.exp(b); w = w / w.sum()
        mb = (w * b).sum(); md = (w * d).sum()
        cov = (w * (b - mb) * (d - md)).sum()
        den = math.sqrt((w * (b - mb) ** 2).sum() * (w * (d - md) ** 2).sum())
        out.append(cov / den if den > 0 else np.nan)
    return np.array(out)


def boot_ci(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    m = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


print("=" * 92)
print("SCALED psi MEASUREMENT -- OLMo-2-0425-1B, 36 prompts x 6 task families")
print("vocab 100,352   |   psi* threshold from theory: 0.71 - 0.83")
print("=" * 92)

results = {}
for name, ref, tgt, note in PAIRS:
    print(f"\n### {name}   ({note})")
    print(f"  {'region':<12} {'mean r':>9} {'sd':>7} {'95% CI':>20} {'verdict':>14}")
    rec = {}
    for rn, k in REGIONS:
        r = psi_region(ref, tgt, k)
        lo, hi = boot_ci(r)
        psi = abs(r.mean())
        verdict = "above psi*" if psi > 0.71 else "BELOW psi*"
        print(f"  {rn:<12} {r.mean():>+9.4f} {r.std():>7.4f} "
              f"[{lo:>+7.4f},{hi:>+7.4f}] {verdict:>14}")
        rec[rn] = dict(mean=float(r.mean()), sd=float(r.std()), ci=[lo, hi],
                       psi=float(psi), n=int(len(r)))
    w = wpsi(ref, tgt)
    lo, hi = boot_ci(w)
    print(f"  {'weighted':<12} {np.nanmean(w):>+9.4f} {np.nanstd(w):>7.4f} "
          f"[{lo:>+7.4f},{hi:>+7.4f}] "
          f"{'above psi*' if abs(np.nanmean(w))>0.71 else 'BELOW psi*':>14}")
    rec["weighted"] = dict(mean=float(np.nanmean(w)), ci=[lo, hi],
                           psi=float(abs(np.nanmean(w))))

    eref = np.array([ent(ref[i]) for i in range(ref.shape[0])])
    etgt = np.array([ent(tgt[i]) for i in range(tgt.shape[0])])
    drop = eref - etgt
    lo, hi = boot_ci(drop)
    print(f"  entropy: {eref.mean():.3f} -> {etgt.mean():.3f} nats  "
          f"(drop {drop.mean():+.3f}, 95% CI [{lo:+.3f},{hi:+.3f}])")
    print(f"  branching: {np.exp(eref).mean():.1f} -> {np.exp(etgt).mean():.1f} "
          f"({100*(np.exp(etgt).mean()/np.exp(eref).mean()-1):+.1f}%)")
    rec["entropy"] = dict(ref=float(eref.mean()), tgt=float(etgt.mean()),
                          drop=float(drop.mean()), drop_ci=[lo, hi],
                          branch_ref=float(np.exp(eref).mean()),
                          branch_tgt=float(np.exp(etgt).mean()))
    results[name] = rec

# ----------------------------------------------------- per-family breakdown
print("\n" + "=" * 92)
print("BY TASK FAMILY -- psi (plausible set) and entropy drop, per delta")
print("=" * 92)
ufam = sorted(set(fams), key=fams.index)
print(f"{'family':<17}" + "".join(f"{n.split(' -> ')[1][:9]:>11}" for n, *_ in PAIRS))
for fam in ufam:
    sel = [i for i, f in enumerate(fams) if f == fam]
    cells = []
    for name, ref, tgt, _ in PAIRS:
        rs = []
        for i in sel:
            b, d = ref[i], tgt[i] - ref[i]
            idx = np.where(b > math.log(1e-6))[0]
            rs.append(np.corrcoef(b[idx], d[idx])[0, 1])
        cells.append(f"{np.mean(rs):>+11.3f}")
    print(f"{fam:<17}" + "".join(cells))

print(f"\n{'family':<17}" + "".join(f"{'dH '+n.split(' -> ')[1][:6]:>11}" for n, *_ in PAIRS))
for fam in ufam:
    sel = [i for i, f in enumerate(fams) if f == fam]
    cells = []
    for name, ref, tgt, _ in PAIRS:
        d = [ent(ref[i]) - ent(tgt[i]) for i in sel]
        cells.append(f"{np.mean(d):>+11.3f}")
    print(f"{fam:<17}" + "".join(cells))

json.dump(results, open(f"{D}/../results_psi_1b.json", "w"), indent=1)
print("\nsaved -> results_psi_1b.json")
