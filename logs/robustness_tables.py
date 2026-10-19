"""Reprints the §5.1 specification grid from the saved JSON and computes every range the paper cites from it.

Reads results/results_robustness{tag}_{family}.json written by code/robustness.py; needs no model dumps.
Writes nothing; redirect stdout to logs/robustness{tag}.txt.

Usage: python logs/robustness_tables.py [tag]      (default tag: _s4, the stride-4 run)
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code")
from robustness import report                       # same table layout as the original run

TAG = sys.argv[1] if len(sys.argv) > 1 else "_s4"
FAMS = ["olmo", "olmo7b", "tulu", "olmo13b"]
STAGES = {"base->sft": "instruction tuning", "sft->dpo": "preference stage"}   # the pre-registered pairs

R = {f: json.load(open(f"{ROOT}/results/results_robustness{TAG}_{f}.json")) for f in FAMS}
for f in FAMS:
    for pair, r in R[f].items():
        report(f, pair, r)

# ---- ranges cited in §5.1 (percent, over the four models and every specification in the slice)
pct = lambda xs: f"{100 * min(xs):.1f} to {100 * max(xs):.1f}"
print(f"\n==== CITED RANGES ({TAG}) ====")
for pair, name in STAGES.items():
    for wt in ["w", "u"]:
        cells = [(f, s, row) for f in FAMS for s, row in R[f][pair]["grid"].items() if s.startswith(wt + "/")]
        print(f"{name:<19} {wt}: typ share {pct([row['typ']['share'] for *_, row in cells])}   "
              f"R2_typ {pct([row['r2_typ']['median'] for *_, row in cells])}")
    fam = [v["typ"] for f in FAMS for v in R[f][pair]["by_family"].values()]
    lofo = [(f, v["typ"]) for f in FAMS for v in R[f][pair]["leave_one_family_out"].values()]
    prim = {f: R[f][pair]["grid"]["w/1e-6/prompt"]["typ"]["share"] for f in FAMS}
    shift = max(abs(v - prim[f]) for f, v in lofo)
    print(f"{name:<19} per-family typ (pre-registered spec) {pct(fam)}; "
          f"largest leave-one-family-out move {100 * shift:.1f} pts")

# held-out shared direction: largest change in any share when g excludes the prompt's own family
dg = max(abs(R[f][p]["grid"][s][k]["share"] - R[f][p]["grid"][s.replace("/prompt", "/family")][k]["share"])
         for f in FAMS for p in STAGES for s in R[f][p]["grid"] if s.endswith("/prompt")
         for k in ["typ", "shared", "resid", "out"])
print(f"largest change from holding the shared direction out by family: {100 * dg:.1f} pts")

# composition vs consequence: cells where the entropy share exceeds the variance share
cells = [R[f][p]["grid"][s] for f in FAMS for p in STAGES for s in R[f][p]["grid"]]
n = sum(c["typ"]["share"] > c["r2_typ"]["median"] for c in cells)
print(f"typ entropy share > its variance share in {n} of {len(cells)} cells")
bad = [f"{f} {p} {s}" for f in FAMS for p in STAGES for s, c in R[f][p]["grid"].items()
       if c["typ"]["share"] <= c["r2_typ"]["median"]]
print("  exceptions: " + "; ".join(bad))

# the final stage (dpo->inst), reported without prediction
print("final stage, pre-registered spec: " + "; ".join(
    f"{f} typ {100 * R[f]['dpo->inst']['grid']['w/1e-6/prompt']['typ']['share']:.1f} "
    f"{[round(100 * x, 1) for x in R[f]['dpo->inst']['grid']['w/1e-6/prompt']['typ']['ci']]} "
    f"shared {100 * R[f]['dpo->inst']['grid']['w/1e-6/prompt']['shared']['share']:.1f}"
    for f in FAMS if "dpo->inst" in R[f]))

# ---- the §5.1 table: per model and stage, range over the six specifications of each weighting
print("\n§5.1 table: typ share (min-max over 3 thresholds x 2 shared directions), weighted R2_typ")
for pair, name in STAGES.items():
    for f in FAMS:
        g = R[f][pair]["grid"]
        rng = lambda wt, k: [g[s][k]["share"] if k != "r2_typ" else g[s][k]["median"] for s in g if s.startswith(wt + "/")]
        w, u, r2 = rng("w", "typ"), rng("u", "typ"), rng("w", "r2_typ")
        print(f"{name:<19} {f:<8} weighted {pct(w):>16}   unweighted {pct(u):>16}   R2_w {pct(r2):>12}")

# ---- stride check: pre-registered cell of this run against decomp.py's full-position shares
print("\nstride check, w/1e-6/prompt typ share vs results_decomp_{family}.json w_typ (points):")
for f in FAMS:
    full = json.load(open(f"{ROOT}/results/results_decomp_{f}.json"))
    print(f"  {f:<8}" + "".join(f"  {p} {100 * R[f][p]['grid']['w/1e-6/prompt']['typ']['share']:6.1f} vs "
                                f"{100 * full[p]['w_typ']['share']:6.1f}" for p in STAGES))
