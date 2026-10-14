"""Score the 1B generation sweep and emit per-condition and per-prompt metrics.

Diversity is computed across the N samples of a single prompt, then averaged
over prompts. Fluency is median GPT-2 perplexity, GPT-2 sharing neither family
nor tokeniser with either model in the mixture.
"""
import os, sys, json, glob, math, itertools
os.environ.setdefault("HF_HOME", "/home/claude/hf")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))

jt = AutoTokenizer.from_pretrained("openai-community/gpt2")
jm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2",
                                          dtype=torch.float32).eval()

LABEL = {
    "T10": ("temp", 1.0, r"A temperature $T{=}1.0$"),
    "T13": ("temp", 1.3, r"A temperature $T{=}1.3$"),
    "T16": ("temp", 1.6, r"A temperature $T{=}1.6$"),
    "L100": ("lam", 1.0, r"B $\lambda{=}{+}1.0$ (aligned)"),
    "L000": ("lam", 0.0, r"B $\lambda{=}0.0$ (base)"),
    "Lm050": ("lam", -0.5, r"B $\lambda{=}{-}0.5$"),
    "Lm100": ("lam", -1.0, r"B $\lambda{=}{-}1.0$"),
    "Lm150": ("lam", -1.5, r"B $\lambda{=}{-}1.5$"),
    "G02": ("dec", 0.2, r"C $\gamma{=}0.2$"),
    "G05": ("dec", 0.5, r"C $\gamma{=}0.5$"),
    "X100": ("ctrl", -1.0, r"D shuffled $-1.0$"),
}
ORDER = ["T10", "T13", "T16", "L100", "L000", "Lm050", "Lm100", "Lm150",
         "G02", "G05", "X100"]


def ngrams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


def distinct_n(ss, n):
    g = []
    for s in ss:
        g += ngrams(s.split(), n)
    return len(set(g)) / max(1, len(g))


def jaccard(ss):
    S = [set(s.split()) for s in ss]
    v = []
    for i, j in itertools.combinations(range(len(S)), 2):
        u = len(S[i] | S[j])
        v.append(len(S[i] & S[j]) / u if u else 0.0)
    return float(np.mean(v)) if v else 0.0


def ttr(ss):
    v = []
    for s in ss:
        w = s.split()
        v.append(len(set(w)) / len(w) if w else 0.0)
    return float(np.mean(v))


@torch.no_grad()
def ppl(ss):
    out = []
    for s in ss:
        s = s.strip()
        if len(s) < 5:
            continue
        ids = jt(s, return_tensors="pt")["input_ids"]
        if ids.shape[1] < 2:
            continue
        l = jm(ids, labels=ids).loss.item()
        if math.isfinite(l):
            out.append(math.exp(min(l, 15)))
    return float(np.median(out)) if out else float("nan")


rows, percell = [], {}
for tag in ORDER:
    fs = sorted(glob.glob(f"{HERE}/gen/{tag}__*.json"))
    if not fs:
        continue
    m = dict(d2=[], jac=[], tt=[], ppl=[])
    cells = {}
    for f in fs:
        d = json.load(open(f))
        s = d["gens"]
        a, b, c, e = distinct_n(s, 2), jaccard(s), ttr(s), ppl(s)
        m["d2"].append(a); m["jac"].append(b); m["tt"].append(c); m["ppl"].append(e)
        cells[d["pi"]] = dict(jac=b, ppl=e, family=d["family"])
    op, param, lab = LABEL[tag]
    rows.append(dict(tag=tag, op=op, param=param, label=lab, n=len(fs),
                     d2=float(np.nanmean(m["d2"])), jac=float(np.nanmean(m["jac"])),
                     ttr=float(np.nanmean(m["tt"])),
                     ppl=float(np.nanmean(m["ppl"])),
                     ppl_median=float(np.nanmedian(m["ppl"]))))
    percell[tag] = cells

print(f"{'condition':<30} {'n':>3} {'dist2':>7} {'jaccard':>8} {'ttr':>6} {'ppl':>10}")
for r in rows:
    print(f"{r['tag']:<30} {r['n']:>3} {r['d2']:>7.3f} {r['jac']:>8.4f} "
          f"{r['ttr']:>6.3f} {r['ppl']:>10.1f}")

json.dump(dict(rows=rows, percell=percell),
          open(f"{HERE}/results_gen_1b.json", "w"), indent=1)
print("\nsaved -> results_gen_1b.json")
