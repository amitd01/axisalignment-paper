"""Score all generated conditions. Run after generation completes.

Diversity metrics are computed across the N samples for a given prompt, then
averaged over prompts. Fluency is judged by GPT-2, which shares neither family
nor tokeniser with either model in the mixture, so it cannot be gamed by either.
"""
import os, json, glob, math, itertools
pass  # port: default HF cache
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(1)
jt = AutoTokenizer.from_pretrained("openai-community/gpt2")
jm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2",
                                          dtype=torch.float32).eval()


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(samples, n):
    g = []
    for s in samples:
        g += ngrams(s.split(), n)
    return len(set(g)) / max(1, len(g))


def pairwise_jaccard(samples):
    sets = [set(s.split()) for s in samples]
    v = []
    for i, j in itertools.combinations(range(len(sets)), 2):
        u = len(sets[i] | sets[j])
        v.append(len(sets[i] & sets[j]) / u if u else 0.0)
    return float(np.mean(v)) if v else 0.0


def self_rep(samples):
    v = []
    for s in samples:
        g = ngrams(s.split(), 3)
        v.append(1 - len(set(g)) / len(g) if g else 0.0)
    return float(np.mean(v))


def type_token(samples):
    v = []
    for s in samples:
        w = s.split()
        v.append(len(set(w)) / len(w) if w else 0.0)
    return float(np.mean(v))


@torch.no_grad()
def judge(samples):
    """Median GPT-2 perplexity. Median rather than mean: one degenerate
    sample at ppl 1e5 would otherwise dominate the condition."""
    out = []
    for s in samples:
        s = s.strip()
        if len(s) < 5:
            continue
        ids = jt(s, return_tensors="pt")["input_ids"]
        if ids.shape[1] < 2:
            continue
        loss = jm(ids, labels=ids).loss.item()
        if math.isfinite(loss):
            out.append(math.exp(min(loss, 15)))
    return (float(np.median(out)), float(np.mean(out))) if out else (float("nan"),) * 2


ORDER = ["T10", "T11", "T12", "T13", "T16", "T20", "G005", "G010", "G020", "G030", "G050"]
_PORT_LABELS = {"T10": "A  temperature  T=1.00", "T11": "A  temperature  T=1.10", "T12": "A  temperature  T=1.20",
                "T13": "A  temperature  T=1.30", "T16": "A  temperature  T=1.60", "T20": "A  temperature  T=2.00",
                "G005": "C  decontam     g=0.05", "G010": "C  decontam     g=0.10", "G020": "C  decontam     g=0.20",
                "G030": "C  decontam     g=0.30", "G050": "C  decontam     g=0.50"}
_ORIG_ORDER = ["T07", "T10", "T13", "T16", "T20",
         "L100", "L050", "L000", "Lm050", "Lm100", "Lm150", "Lm200",
         "G02", "G05", "G08", "G12", "X050", "X100"]
LABEL = {"T07": "A  temperature  T=0.70", "T10": "A  temperature  T=1.00",
         "T13": "A  temperature  T=1.30", "T16": "A  temperature  T=1.60",
         "T20": "A  temperature  T=2.00",
         "L100": "B  lambda       l=+1.00  (aligned)",
         "L050": "B  lambda       l=+0.50",
         "L000": "B  lambda       l= 0.00  (base)",
         "Lm050": "B  lambda      l=-0.50  (reversal)",
         "Lm100": "B  lambda      l=-1.00  (reversal)",
         "Lm150": "B  lambda      l=-1.50  (reversal)",
         "Lm200": "B  lambda      l=-2.00  (reversal)",
         "G02": "C  decontam     g=0.20", "G05": "C  decontam     g=0.50",
         "G08": "C  decontam     g=0.80", "G12": "C  decontam     g=1.20",
         "X050": "D  CONTROL shuffled delta  -0.50",
         "X100": "D  CONTROL shuffled delta  -1.00"}

LABEL.update(_PORT_LABELS)   # port: sweep conditions
rows = []
for tag in ORDER:
    f = f"/Users/amitdas/claude/repos/alignment-axis/code/dumps/135m/gen_{tag}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    m = dict(d2=[], d3=[], jac=[], rep=[], tt=[], pplmed=[], pplmean=[])
    for pr, samples in d["gens"].items():
        m["d2"].append(distinct_n(samples, 2))
        m["d3"].append(distinct_n(samples, 3))
        m["jac"].append(pairwise_jaccard(samples))
        m["rep"].append(self_rep(samples))
        m["tt"].append(type_token(samples))
        a, b = judge(samples)
        m["pplmed"].append(a)
        m["pplmean"].append(b)
    rows.append(dict(tag=tag, label=LABEL[tag], op=d["op"], param=d["param"],
                     **{k: float(np.nanmean(v)) for k, v in m.items()},
                     per_prompt={k: [float(x) for x in v] for k, v in m.items()}))   # port: for bootstrap

print("=" * 100)
print("EXPERIMENT 3 -- RESULTS")
print("SmolLM2-135M (base) / SmolLM2-135M-Instruct (aligned), 5 prompts x 8 samples x 32 tok")
print("Judge: GPT-2 (independent family and tokeniser). top-p=0.95 held constant.")
print("=" * 100)
print(f"{'condition':<36} {'dist-2':>8} {'jaccard':>8} {'ttr':>7} {'selfrep':>8} "
      f"{'GPT2ppl':>9}")
print("-" * 100)
for r in rows:
    print(f"{r['label']:<36} {r['d2']:>8.3f} {r['jac']:>8.3f} {r['tt']:>7.3f} "
          f"{r['rep']:>8.3f} {r['pplmed']:>9.1f}")

# ---------------------------------------------- matched-diversity comparison
print("\n" + "=" * 100)
print("MATCHED-DIVERSITY COMPARISON (interpolating each operator's curve)")
print("Lower Jaccard = more diverse. At equal Jaccard, lower ppl = better operator.")
print("=" * 100)


def curve(op):
    pts = [(r["jac"], r["pplmed"], r["param"]) for r in rows if r["op"] == op]
    return sorted(pts, key=lambda x: -x[0])


def interp(op, target_jac):
    c = curve(op)
    xs = [p[0] for p in c][::-1]
    ys = [p[1] for p in c][::-1]
    if not xs or target_jac < min(xs) or target_jac > max(xs):
        return None
    return float(np.interp(target_jac, xs, ys))


print(f"{'tgt jac':>9} {'A temp':>10} {'B lambda':>11} {'C decon':>10} {'D ctrl':>10} {'winner':>15}")
for tj in [0.10, 0.075, 0.06, 0.05, 0.040, 0.035, 0.025, 0.015]:
    v = {o: interp(o, tj) for o in ["temp", "lam", "dec", "ctrl"]}
    ok = {k: x for k, x in v.items() if x is not None}
    win = min(ok, key=ok.get) if ok else "-"
    nm = {"temp":"temperature","lam":"lambda","dec":"decontaminate","ctrl":"shuffled-ctrl"}
    fmt = lambda x: f"{x:.1f}" if x is not None else "  n/a"
    print(f"{tj:>9.3f} {fmt(v['temp']):>10} {fmt(v['lam']):>11} {fmt(v['dec']):>10} {fmt(v['ctrl']):>10} {nm.get(win,'-'):>15}")

json.dump(rows, open("/Users/amitdas/claude/repos/alignment-axis/code/dumps/135m/results_sweep_scored.json", "w"), indent=1)
print("\nsaved -> results_exp3_scored.json")

# ------------------------------------------------------------- sample output
print("\n" + "=" * 100)
print("SAMPLE OUTPUT (prompt 1, first 2 samples per condition)")
print("=" * 100)
for tag in ORDER:
    f = f"/Users/amitdas/claude/repos/alignment-axis/code/dumps/135m/gen_{tag}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    pr = list(d["gens"])[0]
    print(f"\n{LABEL[tag]}")
    for s in d["gens"][pr][:2]:
        print(f"   | {s.strip()[:110]}")
