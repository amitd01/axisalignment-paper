"""Generate the numeric tables for results/TASK2-REPORT.md straight from result JSONs."""
import json, numpy as np
L = lambda p: json.load(open(p))
f = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
pc = lambda d: f"{100*d['share']:.1f}% [{100*d['ci'][0]:.1f}, {100*d['ci'][1]:.1f}]"
M = [("OLMo-2-0425-1B", "olmo"), ("OLMo-2-1124-7B", "olmo7b"), ("Tulu-3-8B", "tulu"), ("OLMo-2-1124-13B", "olmo13b")]
out = []
out.append("### T1. Stage shares with absolute drops and totals, by measurement convention\n")
out.append("| model | convention | SFT share | DPO share | DPO drop (nats) | RLVR share | total loss (nats) |")
out.append("|---|---|---|---|---|---|---|")
for name, k in M:
    pf = L(f"results/results_psi_{k}.json")["results"]["entropy_share"]
    s = pf["stages"]
    out.append(f"| {name} | raw prompt-final (artefact arm) | {pc(s['SFT'])} | {pc(s['DPO'])} | {s['DPO']['drop']:.3f} | {pc(s['RLVR/final'])} | {pf['total_drop']:.3f} |")
    for conv, fam, key in [("A chat-templated", k, "A_shares"), ("**B base-written (primary)**", k, "B_shares"), ("B final-model-written", k + "@inst", "B_shares")]:
        r = L(f"results/results_register_{fam}.json")[key]
        out.append(f"| {name} | {conv} | {pc(r['SFT'])} | {pc(r['DPO'])} | {r['DPO']['drop']:.3f} | {pc(r['RLVR/final'])} | {r['total']:.3f} |")
out.append("\n### T2. psi on the preference delta (SFT->DPO), all regions\n")
out.append("| model | convention | top-10 | top-50 | top-1000 | plausible | prob-weighted |")
out.append("|---|---|---|---|---|---|---|")
for name, k in M:
    r = L(f"results/results_psi_{k}.json")["results"]["SFT -> DPO"]
    g = lambda reg: f"{r[reg]['mean']:+.3f} [{r[reg]['ci'][0]:+.3f}, {r[reg]['ci'][1]:+.3f}]"
    out.append(f"| {name} | prompt-final | {g('top-10')} | {g('top-50')} | {g('top-1000')} | {g('plausible')} | {g('weighted')} |")
    for conv, fam in [("B base-written", k), ("B final-written", k + "@inst")]:
        G = L(f"results/results_register_{fam}.json")["B_geometry"]["sft->dpo"]
        out.append(f"| {name} | {conv} | -- | {f(G['psi_50'])} | -- | {f(G['psi_pl'])} | {f(G['psi_w'])} |")
mx = {}
for name, k in M:
    R = L(f"results/results_psi_{k}.json")["results"]
    for d in ["base -> SFT", "SFT -> DPO", "DPO -> Instruct", "base -> Instruct"]:
        mx.setdefault("prompt-final", []).append((R[d]["plausible"]["mean"], name, d)); mx.setdefault("pf_w", []).append((R[d]["weighted"]["mean"], name, d))
    for conv, fam in [("B base", k), ("B final", k + "@inst")]:
        for d, G in L(f"results/results_register_{fam}.json")["B_geometry"].items():
            if "basecpu" in d: continue
            mx.setdefault(conv, []).append((G["psi_pl"][0], name, d)); mx.setdefault(conv + "_w", []).append((G["psi_w"][0], name, d))
out.append("\nMax psi per convention (4 models x 4 deltas): " + "; ".join(f"{c}: {max(v)[0]:.3f} ({max(v)[1]} {max(v)[2]})" for c, v in mx.items()))
out.append("\n### T3. Register geometry, arm B base-written, plausible region (pre-registered test = base->SFT)\n")
out.append("| model | delta | psi | rho_reg | partial | rho - psi | psi_w | rho_w |")
out.append("|---|---|---|---|---|---|---|---|")
for name, k in M:
    G = L(f"results/results_register_{k}.json")["B_geometry"]
    for d in ["base->sft", "sft->dpo", "dpo->inst", "base->basecpu"]:
        if d in G:
            g = G[d]; lab = "NULL (MPS vs CPU)" if "cpu" in d else d
            out.append(f"| {name} | {lab} | {f(g['psi_pl'])} | {f(g['rho_pl'])} | {f(g['part_pl'])} | {f(g['rho_minus_psi_pl'])} | {f(g['psi_w'])} | {f(g['rho_w'])} |")
out.append("\n### T4. Special-token mass by stage (v1 register test, raw prompt-final) -- null\n")
out.append("| model | base | SFT | DPO | final |"); out.append("|---|---|---|---|---|")
for name, k in M:
    r = L(f"results/results_psi_{k}.json")["results"]["register_diagnostic"]
    out.append(f"| {name} | " + " | ".join(f"{r[s]['special_mass_mean']:.1e}" for s in ["base", "sft", "dpo", "inst"]) + " |")
out.append("\n### T5. Entropy-loss decomposition, arm B base-written (Shapley shares of each stage's loss)\n")
out.append("| model | delta | loss (nats) | head | body | typ (unw) | shared (unw) | resid (unw) | typ (w) | shared (w) | typ(w) - shared(w) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for name, k in M:
    R = L(f"results/results_decomp_{k}.json")
    for d in ["base->sft", "sft->dpo", "dpo->inst"]:
        r = R[d]; s = lambda x: f"{100*r[x]['share']:.0f}%"
        out.append(f"| {name} | {d} | {r['entropy_loss']:.3f} | {pc(r['reg_head'])} | {s('reg_body')} | {s('u_typ')} | {s('u_shared')} | {s('u_resid')} | {s('w_typ')} | {s('w_shared')} | {100*r['w_typ_minus_shared']['share']:+.0f} [{100*r['w_typ_minus_shared']['ci'][0]:+.0f}, {100*r['w_typ_minus_shared']['ci'][1]:+.0f}] |")
out.append("\n### T6. Residual test: base->SFT entropy loss with NONLINEAR typicality and cross-fitted shared directions\n")
out.append("| model | nonlinear typ (prereg stat) | linear typ | merged typ (exploratory) | shared | eps (merged run) | prereg verdict on 'cannot touch' |")
out.append("|---|---|---|---|---|---|---|")
for name, k in M:
    r = L(f"results/results_decomp_resid_{k}.json")["base->sft"]; m = L(f"results/results_decomp_resid_merged_{k}.json")["base->sft"]
    lo, hi = r["nonlin_typ"]["ci"]; v = "too strong" if lo > 0.5 else ("stands" if hi < 0.5 else "unresolved")
    out.append(f"| {name} | {pc(r['nonlin_typ'])} | {pc(r['lin_typ'])} | {pc(m['lin_typ'])} | {pc(m['shared'])} | {pc(m['eps'])} | {v} |")
print("\n".join(out))
