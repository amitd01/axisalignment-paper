"""Emit LaTeX table fragments from the result JSONs into the arXiv packet.

Tables that depend on the generation sweep emit a clearly-marked placeholder if
the sweep has not finished, so the document always compiles.
"""
import os, json, math, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/claude/arxiv-packet"
os.makedirs(OUT, exist_ok=True)

psi = json.load(open(f"{HERE}/results_psi_1b.json"))

PAIRS = ["base -> SFT", "SFT -> DPO", "DPO -> Instruct", "base -> Instruct"]
NICE = {"base -> SFT": r"base$\to$SFT", "SFT -> DPO": r"SFT$\to$DPO",
        "DPO -> Instruct": r"DPO$\to$Instruct",
        "base -> Instruct": r"base$\to$Instruct"}
NOTE = {"base -> SFT": "instruction tuning", "SFT -> DPO": "preference optimisation",
        "DPO -> Instruct": "verifiable reward", "base -> Instruct": "full pipeline"}


def w(name, s):
    open(f"{OUT}/{name}", "w").write(s)
    print("wrote", name)


# ---------------------------------------------------------------- stages
rows = []
for p in PAIRS:
    e = psi[p]["entropy"]
    rows.append(
        f"{NICE[p]} & {NOTE[p]} & {e['ref']:.3f} & {e['tgt']:.3f} & "
        f"${e['drop']:+.3f}$ & $[{e['drop_ci'][0]:+.3f}, {e['drop_ci'][1]:+.3f}]$ & "
        f"{e['branch_ref']:.1f} & {e['branch_tgt']:.1f} & "
        f"${100*(e['branch_tgt']/e['branch_ref']-1):+.1f}\\%$ \\\\")
w("tab_stages.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Entropy cost of each alignment stage. Next-token entropy in nats at the
final prompt position, averaged over 36 prompts, with bootstrap 95\% CIs over
prompts (20{,}000 resamples). Instruction tuning accounts for essentially the
entire collapse; preference optimisation does not.}
\label{tab:stages}
\begin{tabular}{llcccccc r}
\toprule
Stage & & $H_{\mathrm{ref}}$ & $H_{\mathrm{tgt}}$ & $\Delta H$ & 95\% CI &
  \multicolumn{2}{c}{branching} & change \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- psi
REG = [("top-10", "top-10"), ("top-50", "top-50"), ("top-100", "top-100"),
       ("top-1000", "top-1000"), ("plausible", "plausible set"),
       ("weighted", "prob.\\ weighted")]
lines = []
for rn, disp in REG:
    cells = []
    for p in PAIRS:
        d = psi[p][rn]
        ci = d["ci"]
        cells.append(f"${d['mean']:+.3f}$ & \\scriptsize$[{ci[0]:+.3f},{ci[1]:+.3f}]$")
    lines.append(f"{disp} & " + " & ".join(cells) + r" \\")
w("tab_psi.tex", r"""\begin{table}[htbp]
\centering\small
\caption{$\psi$ by region and stage: correlation of the per-token alignment
delta with reference log-probability, averaged over 36 prompts with bootstrap
95\% CIs. Typicality bias predicts values near 1; the decision threshold from
theory is $\psi^{*}\approx0.71$--$0.83$. Every cell falls far below it, and the
true preference delta (SFT$\to$DPO) is indistinguishable from zero throughout.}
\label{tab:psi}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l cc cc cc cc}
\toprule
& \multicolumn{2}{c}{base$\to$SFT} & \multicolumn{2}{c}{SFT$\to$DPO}
& \multicolumn{2}{c}{DPO$\to$Instruct} & \multicolumn{2}{c}{base$\to$Instruct} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Region & $r$ & 95\% CI & $r$ & 95\% CI & $r$ & 95\% CI & $r$ & 95\% CI \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- family
import sys
sys.path.insert(0, HERE)
from prompts import FLAT
D = f"{HERE}/dumps"
arr = {t: np.load(f"{D}/{t}.npy") for t in ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_inst"]}
MAP = {"base -> SFT": ("olmo_base", "olmo_sft"), "SFT -> DPO": ("olmo_sft", "olmo_dpo"),
       "DPO -> Instruct": ("olmo_dpo", "olmo_inst"),
       "base -> Instruct": ("olmo_base", "olmo_inst")}
fams = [f for f, _ in FLAT]
ufam = sorted(set(fams), key=fams.index)
ENT = lambda lp: float(-(np.exp(lp) * lp).sum())
flines = []
for fam in ufam:
    sel = [i for i, f in enumerate(fams) if f == fam]
    cs = []
    for p in PAIRS:
        a, b = MAP[p]
        cs.append(f"${np.mean([ENT(arr[a][i]) - ENT(arr[b][i]) for i in sel]):+.3f}$")
    flines.append(f"{fam.replace('_',' ')} & " + " & ".join(cs) + r" \\")
w("tab_family.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Entropy change ($\Delta H$, nats) per stage, broken out by task family,
six prompts each. Instruction tuning costs between 0.44 and 0.87 nats in every
register; the preference-optimisation column hovers near zero and changes sign.}
\label{tab:family}
\begin{tabular}{lcccc}
\toprule
Task family & base$\to$SFT & SFT$\to$DPO & DPO$\to$Instruct & base$\to$Instruct \\
\midrule
""" + "\n".join(flines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- generation
GEN = f"{HERE}/results_gen_1b.json"
PLACE = (r"""\begin{table}[htbp]\centering\small
\caption{PLACEHOLDER --- generation sweep still running.}
\begin{tabular}{l}\toprule pending \\ \bottomrule\end{tabular}\end{table}
""")

if not os.path.exists(GEN):
    for f in ["tab_gen.tex", "tab_matched.tex", "tab_control.tex"]:
        w(f, PLACE)
    print("\n(generation tables are placeholders -- sweep not finished)")
else:
    g = json.load(open(GEN))
    rows = g["rows"]
    byt = {r["tag"]: r for r in rows}

    main = [r for r in rows if r["op"] != "ctrl"]
    lines = [f"{r['label']} & {r['d2']:.3f} & {r['jac']:.4f} & {r['ttr']:.3f} & "
             f"{r['ppl']:.1f} \\\\" for r in main]
    w("tab_gen.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Generation sweep, OLMo-2-1B base$\leftrightarrow$Instruct. 12 prompts
$\times$ 6 samples $\times$ 32 tokens, top-$p=0.95$ held constant. Lower Jaccard
is more diverse; lower perplexity is more fluent.}
\label{tab:gen}
\begin{tabular}{lcccc}
\toprule
Condition & distinct-2 & Jaccard $\downarrow$ & TTR & GPT-2 ppl \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    def curve(op):
        pts = [(r["jac"], r["ppl"]) for r in rows if r["op"] == op]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    def interp(op, t):
        xs, ys = curve(op)
        if not xs or t < min(xs) or t > max(xs):
            return None
        return float(np.interp(t, xs, ys))

    jl = sorted([r["jac"] for r in rows if r["op"] == "lam"])
    targets = [round(x, 3) for x in np.linspace(max(0.005, jl[0]), jl[-1], 6)]
    ml = []
    for t in targets:
        v = {o: interp(o, t) for o in ["temp", "lam", "dec", "ctrl"]}
        ok = {k: x for k, x in v.items() if x is not None}
        win = min(ok, key=ok.get) if ok else "--"
        nm = {"temp": "temperature", "lam": r"$\lambda$", "dec": r"$\gamma$",
              "ctrl": "control"}
        f = lambda x: f"{x:.1f}" if x is not None else "--"
        ml.append(f"{t:.3f} & {f(v['temp'])} & {f(v['lam'])} & {f(v['dec'])} & "
                  f"{f(v['ctrl'])} & {nm.get(win,'--')} \\\\")
    w("tab_matched.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Matched-diversity comparison, interpolating each operator's frontier.
At a given diversity level, lower perplexity is the better operator.}
\label{tab:matched}
\begin{tabular}{lccccl}
\toprule
Target Jaccard & A temperature & B $\lambda$ & C $\gamma$ & D control & winner \\
\midrule
""" + "\n".join(ml) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    cl = []
    for tag in ["Lm100", "X100", "T13"]:
        if tag in byt:
            r = byt[tag]
            cl.append(f"{r['label']} & {r['jac']:.4f} & {r['ppl']:.1f} \\\\")
    w("tab_control.tex", r"""\begin{table}[htbp]
\centering\small
\caption{Direction control. The shuffled delta preserves the true delta's norm
and value distribution exactly, destroying only which token receives which
shift.}
\label{tab:control}
\begin{tabular}{lcc}
\toprule
Condition & Jaccard $\downarrow$ & GPT-2 ppl \\
\midrule
""" + "\n".join(cl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
