"""Ad-hoc check (not pre-registered): is base-written minus final-written DPO share a stable ~9pt offset?"""
import json, numpy as np
D = "code/dumps/register"; S = ["base", "sft", "dpo", "inst"]
def per_prompt(fam):
    rows = json.load(open(f"{D}/{fam}_cont.json"))["rows"]
    p = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows])
    H = {s: np.load(f"{D}/{fam}_B_{s}_H.npy").astype(float) for s in S}
    return {s: np.array([H[s][p == i].mean() for i in range(36)]) for s in S}
def stats(pm, J):
    tot = (pm["base"] - pm["inst"])[J].mean(-1)
    d = {k: (pm[a] - pm[b])[J].mean(-1) for k, a, b in [("SFT","base","sft"),("DPO","sft","dpo"),("RLVR","dpo","inst")]}
    return d, tot
rng = np.random.default_rng(0); J = rng.integers(0, 36, (20000, 36)); I = np.arange(36)
ci = lambda x: f"{x.mean():+.2f} [{np.percentile(x,2.5):+.2f},{np.percentile(x,97.5):+.2f}]"
offs = {}
for fam in ["olmo", "olmo7b", "tulu"]:
    A, B = per_prompt(fam), per_prompt(fam + "@inst")
    (da, ta), (db, tb) = stats(A, I), stats(B, I)
    (Da, Ta), (Db, Tb) = stats(A, J), stats(B, J)
    off = 100 * (Da["DPO"] / Ta - Db["DPO"] / Tb); offs[fam] = off
    print(f"\n{fam}: DPO share base-text {100*da['DPO']/ta:.1f}%  final-text {100*db['DPO']/tb:.1f}%  "
          f"offset {100*(da['DPO']/ta - db['DPO']/tb):+.1f} pts  95% CI [{np.percentile(off,2.5):+.1f}, {np.percentile(off,97.5):+.1f}]")
    for k in ["SFT", "DPO", "RLVR"]:
        print(f"   {k:<5} drop nats  base-text {da[k]:+.3f}  final-text {db[k]:+.3f}   diff(final-base) {ci(Db[k]-Da[k])}")
    print(f"   total drop    base-text {ta:+.3f}  final-text {tb:+.3f}   diff {ci(Tb-Ta)}")
    print(f"   base entropy  base-text {A['base'].mean():.3f}  final-text {B['base'].mean():.3f}")
print("\nbetween-model offset differences (independent bootstraps):")
for a, b in [("olmo","olmo7b"),("olmo7b","tulu"),("olmo","tulu")]:
    print(f"   {a} - {b}: {ci(offs[a]-offs[b])} pts")
