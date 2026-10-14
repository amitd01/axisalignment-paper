"""Generalised analyse_psi.py: usage  python analyse_psi_family.py <olmo|tulu>

Measure psi for each training-stage delta, across regions of the distribution.

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

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "olmo"     # tag prefix: olmo | tulu
TAGS = dict(base=f"{FAMILY}_base", sft=f"{FAMILY}_sft", dpo=f"{FAMILY}_dpo", inst=f"{FAMILY}_inst")

# Tokeniser identity at the token-ID level is a hard requirement before any
# logit arithmetic. Compare the exact input IDs each model saw, and vocab width.
metas = {k: json.load(open(f"{D}/{t}.json")) for k, t in TAGS.items()}
ref_ids = metas["base"].get("input_ids")
for k, m in metas.items():
    if ref_ids is not None:
        assert m["input_ids"] == ref_ids, f"input token IDs differ between base and {k}"
dts = {k: m.get("param_dtype", "unrecorded") for k, m in metas.items()}
print(f"param dtypes: {dts}")
assert len(set(dts.values())) == 1, f"checkpoints loaded at different precisions: {dts}"
print(f"token-ID identity: OK across {list(TAGS)} "
      f"({'input IDs checked' if ref_ids else 'vocab width only'}; vocab={metas['base']['vocab']})")

# Common support. Tulu checkpoints widen the head to 128,264 (<pad> + padding) while
# base Llama 3.1 is 128,256. Log-probs normalised over different supports must not be
# subtracted, so every stage is restricted to the shared ID range and renormalised --
# identical to log_softmax of the truncated logits, and a no-op where widths match.
# check_tokenizers.py verifies the token->ID map is identical over that range.
def common(lp, V):
    if lp.shape[1] == V:                       # widths match: strict no-op
        return lp, np.zeros(lp.shape[0])
    x = lp.astype(np.float64)
    lse_in = np.logaddexp.reduce(x[:, :V], axis=1, keepdims=True)
    lse_all = np.logaddexp.reduce(x, axis=1, keepdims=True)
    return (x[:, :V] - lse_in).astype(np.float32), 1 - np.exp(lse_in - lse_all)[:, 0]
raw = {k: load(TAGS[k]) for k in ("base", "sft", "dpo", "inst")}
V = min(a.shape[1] for a in raw.values())
print(f"vocab widths: { {k: a.shape[1] for k, a in raw.items()} } -> common V={V:,}")
cut = {}
for k in raw:
    raw[k], dropped = common(raw[k], V)
    cut[k] = dropped
    print(f"  {k:<5} mass on IDs >= V: max {dropped.max():.2e}  mean {dropped.mean():.2e}")
base, sft, dpo, inst = (raw[k] for k in ("base", "sft", "dpo", "inst"))
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
print(f"psi MEASUREMENT -- {FAMILY}: {[m['model_id'] for m in metas.values()]}")
print(f"{base.shape[0]} prompts | vocab {base.shape[1]:,} | psi* threshold from theory: 0.71 - 0.83")
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

# ----------------------------------------- share of total entropy loss by stage
# Headline for cross-family comparison: absolute nats differ with scale, the share does not.
E = {k: np.array([ent(a[i]) for i in range(a.shape[0])]) for k, a in
     dict(base=base, sft=sft, dpo=dpo, inst=inst).items()}
steps = [("SFT", "base", "sft"), ("DPO", "sft", "dpo"), ("RLVR/final", "dpo", "inst")]
total = E["base"] - E["inst"]
rng = np.random.default_rng(0)
boots = []
for _ in range(20000):
    j = rng.integers(0, len(total), len(total))
    boots.append([(E[a][j] - E[b][j]).mean() / total[j].mean() for _, a, b in steps])
boots = np.array(boots)
print("\n" + "=" * 92)
print(f"SHARE OF TOTAL ENTROPY LOSS BY STAGE  (total {total.mean():+.3f} nats, "
      f"branching {np.exp(E['base']).mean():.1f} -> {np.exp(E['inst']).mean():.1f}, "
      f"{100*(np.exp(E['inst']).mean()/np.exp(E['base']).mean()-1):+.1f}%)")
print("=" * 92)
share = {}
for (name, a, b), col in zip(steps, boots.T):
    d = (E[a] - E[b]).mean()
    sh = d / total.mean()
    lo, hi = np.percentile(col, [2.5, 97.5])
    print(f"  {name:<11} drop {d:+.4f} nats   share {100*sh:6.1f}%   95% CI [{100*lo:6.1f}%, {100*hi:6.1f}%]")
    share[name] = dict(drop=float(d), share=float(sh), ci=[float(lo), float(hi)])
results["entropy_share"] = dict(total_drop=float(total.mean()), stages=share)

# ------------------------------------ register diagnostic: mass on special tokens
# (b) mass on IDs >= V that the common support excludes (zero at base by construction)
# (c) mass on special/reserved IDs inside the common support: [tokenizer base vocab, V).
#     For Llama 3.1 that is 128000-128255 (<|eot_id|>, <|start_header_id|>, ...),
#     measurable at every stage including base. Uses the un-renormalised full-width rows.
print("\n" + "=" * 92)
print("REGISTER DIAGNOSTIC -- probability mass on special tokens, per stage (mean over prompts)")
print("=" * 92)
sp0 = metas["base"]["tokenizer_vocab_size"]
reg = {}
for k in ("base", "sft", "dpo", "inst"):
    full = load(TAGS[k]).astype(np.float64)
    spm = np.exp(full[:, sp0:V]).sum(1)
    top = [int(sp0 + i) for i in np.argsort(-np.exp(full[:, sp0:V]).mean(0))[:3]]
    reg[k] = dict(excluded_mass_mean=float(cut[k].mean()), excluded_mass_max=float(cut[k].max()),
                  special_mass_mean=float(spm.mean()), special_mass_max=float(spm.max()),
                  special_id_range=[int(sp0), int(V)], top_special_ids=top)
    print(f"  {k:<5} (b) excluded IDs>={V}: mean {cut[k].mean():.2e} max {cut[k].max():.2e}   "
          f"(c) special IDs [{sp0},{V}): mean {spm.mean():.2e} max {spm.max():.2e}  top ids {top}")
results["register_diagnostic"] = reg

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

RES = os.path.dirname(D) + "/../results"
out = f"{RES}/results_psi_{FAMILY}.json"
json.dump(dict(models={k: m["model_id"] for k, m in metas.items()}, common_vocab=int(V),
               dropped_mass_max={k: float(v.max()) for k, v in cut.items()}, results=results),
          open(out, "w"), indent=1)
print(f"\nsaved -> {os.path.normpath(out)}")
