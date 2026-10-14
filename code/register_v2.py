"""Register test v2 (pre-registered in results/PREREG-REGISTER.md).

v1 measured special-token mass at raw-prompt-final positions. In OLMo-2 and Tulu-3 the chat-role markers
are ordinary text tokens, so v1 could not see the register at all. v2 tests the geometry instead:

  psi      = corr(delta, ref log-prob)            typicality: shift is a reweighting of what ref liked
  rho_reg  = corr(delta, g_-p)                    register: shift follows a shared, context-free direction
             g_-p = mean delta over positions of the OTHER prompts (leave-one-prompt-out)
  rho_part = rho_reg controlling for ref log-prob

Arm A: chat-templated prompts, assistant-start position (36 rows).
Arm B: base-model continuations (6 x 32 tokens per prompt) teacher-forced through every stage, so all
       stages are compared on identical contexts; every continuation position is a row.

Usage:
  python register_v2.py gen   <family>            # arm-B continuations from the family's base model
  python register_v2.py score <family> <stage>    # arm A + arm B log-probs for one stage
  python register_v2.py analyse <family>
"""
import os, sys, json, gc, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import FLAT

D = os.path.dirname(os.path.abspath(__file__)) + "/dumps/register"
os.makedirs(D, exist_ok=True)
SEED, N_SAMPLES, N_NEW, TOP_P = 20261012, 6, 32, 0.95

FAMILIES = {
    "olmo": dict(V=100352, template="allenai/OLMo-2-0425-1B-SFT", stages=dict(
        base="allenai/OLMo-2-0425-1B", sft="allenai/OLMo-2-0425-1B-SFT",
        dpo="allenai/OLMo-2-0425-1B-DPO", inst="allenai/OLMo-2-0425-1B-Instruct")),
    "olmo7b": dict(V=100352, template="allenai/OLMo-2-1124-7B-SFT", stages=dict(
        base="allenai/OLMo-2-1124-7B", sft="allenai/OLMo-2-1124-7B-SFT",
        dpo="allenai/OLMo-2-1124-7B-DPO", inst="allenai/OLMo-2-1124-7B-Instruct")),
    "olmo13b": dict(V=100352, template="allenai/OLMo-2-1124-13B-SFT", stages=dict(
        base="allenai/OLMo-2-1124-13B", sft="allenai/OLMo-2-1124-13B-SFT",
        dpo="allenai/OLMo-2-1124-13B-DPO", inst="allenai/OLMo-2-1124-13B-Instruct")),
    # Tulu heads are 128,264 wide vs base 128,256: every stage is restricted to the shared IDs and
    # renormalised, exactly as in analyse_psi_family.py.
    "tulu": dict(V=128256, template="allenai/Llama-3.1-Tulu-3-8B-SFT", stages=dict(
        base="meta-llama/Llama-3.1-8B", sft="allenai/Llama-3.1-Tulu-3-8B-SFT",
        dpo="allenai/Llama-3.1-Tulu-3-8B-DPO", inst="allenai/Llama-3.1-Tulu-3-8B")),
}
STAGES = ["base", "sft", "dpo", "inst"]
# Exploratory (not pre-registered): arm-B continuations written by the FINAL model instead of the base,
# to check that arm-B results are not an artefact of scoring aligned stages on base-style text.
for _f in ["olmo", "olmo7b", "olmo13b", "tulu"]:
    FAMILIES[f"{_f}@inst"] = dict(FAMILIES[_f], gen_src="inst")
# Cross-lineage test (results/PREREG-LINEAGE.md): 1B stages score text written by OLMo-2-1124-7B-Instruct,
# of which no 1B checkpoint is an ancestor. Shared tokeniser verified.
FAMILIES["olmo@x7binst"] = dict(FAMILIES["olmo"], gen_model="allenai/OLMo-2-1124-7B-Instruct")


def load_model(name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval().to(dev)
    dts = sorted({str(p.dtype) for p in model.parameters()})
    assert dts == ["torch.bfloat16"], f"{name} not uniformly bf16: {dts}"
    print(f"{name}: params {dts} on {dev}", flush=True)
    return tok, model, dev


def gen(family):
    import torch
    F = FAMILIES[family]
    src = F.get("gen_model") or F["stages"][F.get("gen_src", "base")]
    tok, model, dev = load_model(src)
    out = []
    with torch.no_grad():
        for pi, (fam, p) in enumerate(FLAT):
            ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
            torch.manual_seed(SEED + pi)
            g = model.generate(ids, do_sample=True, temperature=1.0, top_p=TOP_P, top_k=0,
                               max_new_tokens=N_NEW, num_return_sequences=N_SAMPLES,
                               pad_token_id=tok.eos_token_id)
            for s in range(N_SAMPLES):
                c = g[s, ids.shape[1]:].tolist()
                if tok.eos_token_id in c:              # keep through the first EOS, drop padding after it
                    c = c[:c.index(tok.eos_token_id) + 1]
                out.append(dict(prompt_idx=pi, sample=s, prompt_ids=ids[0].tolist(), cont_ids=c))
    json.dump(dict(source=src, seed=SEED, top_p=TOP_P, rows=out),
              open(f"{D}/{family}_cont.json", "w"))
    print(f"saved {len(out)} continuations, mean len {np.mean([len(r['cont_ids']) for r in out]):.1f}")


def score(family, stage):
    import torch
    F = FAMILIES[family]; V = F["V"]
    name = F["stages"][stage]
    stage = os.environ.get("OUT_STAGE", stage)      # e.g. basecpu: same weights, other backend (null control)
    ttok = __import__("transformers").AutoTokenizer.from_pretrained(F["template"])
    tok, model, dev = load_model(name)

    def lp_common(logits):                         # fp32 log-softmax over the shared ID range
        lg = logits.float()
        full = torch.logsumexp(lg, -1, keepdim=True)
        lp = torch.log_softmax(lg[..., :V], -1)
        excl = 1 - torch.exp(torch.logsumexp(lg[..., :V], -1) - full[..., 0])
        return lp, excl

    with torch.no_grad():
        # ---- arm A: chat-templated, assistant-start position
        A, A_ids, A_ex = [], [], []
        for fam, p in FLAT:
            ids = ttok.apply_chat_template([{"role": "user", "content": p}],
                                           add_generation_prompt=True, tokenize=True)
            ids = list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
            # The raw-prompt protocol feeds Llama a BOS; Tulu's template text omits it. Prepend so arm A
            # differs from the raw protocol only by the template. No-op for OLMo (template starts with BOS).
            if ttok.bos_token_id is not None and ids[0] != ttok.bos_token_id:
                ids = [ttok.bos_token_id] + ids
            lp, ex = lp_common(model(torch.tensor([ids], device=dev)).logits[0, -1])
            A.append(lp.cpu().numpy().astype(np.float32)); A_ids.append(ids); A_ex.append(float(ex))
        np.save(f"{D}/{family}_A_{stage}.npy", np.stack(A))

        # ---- arm B: teacher-forced base continuations, every continuation position
        rows = json.load(open(f"{D}/{family}_cont.json"))["rows"]
        n = sum(len(r["cont_ids"]) for r in rows)
        B = np.lib.format.open_memmap(f"{D}/{family}_B_{stage}.npy", mode="w+",
                                      dtype=np.float16, shape=(n, V))
        H = np.zeros(n, np.float32); EX = np.zeros(n, np.float32)
        k = 0
        for r in rows:
            seq = r["prompt_ids"] + r["cont_ids"]
            Lp, Lc = len(r["prompt_ids"]), len(r["cont_ids"])
            lg = model(torch.tensor([seq], device=dev)).logits[0, Lp - 1:Lp - 1 + Lc]
            lp, ex = lp_common(lg)
            H[k:k + Lc] = (-(lp.exp() * lp).sum(-1)).cpu().numpy()   # exact fp32 entropy
            EX[k:k + Lc] = ex.cpu().numpy()
            B[k:k + Lc] = lp.cpu().numpy().astype(np.float16)
            k += Lc
        B.flush()
    np.save(f"{D}/{family}_B_{stage}_H.npy", H)
    json.dump(dict(model_id=name, stage=stage, V=V, A_input_ids=A_ids, A_excluded_mass=A_ex,
                   B_rows=n, B_excluded_mass_max=float(EX.max()), B_excluded_mass_mean=float(EX.mean()),
                   eos_token_id=tok.eos_token_id),
              open(f"{D}/{family}_{stage}.json", "w"))
    print(f"saved {family}/{stage}: A {len(A)} rows, B {n} rows, excluded mass max {EX.max():.2e}")


# ----------------------------------------------------------------------------------- analysis
def wcorr(x, y, w):
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    den = math.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())
    return cov / den if den > 0 else np.nan


def corr(x, y):
    if len(x) < 5 or x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def partial(r_dg, r_db, r_gb):
    den = math.sqrt(max(1e-12, (1 - r_db ** 2) * (1 - r_gb ** 2)))
    return (r_dg - r_db * r_gb) / den


def boot(x, n=20000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def geometry(ref, tgt, prompt_of, label):
    """psi vs rho_reg per row, averaged within prompt, bootstrapped over prompts."""
    P = int(prompt_of.max()) + 1
    Gp = np.zeros((P, ref.shape[1])); cnt = np.zeros(P)
    for i in range(ref.shape[0]):
        Gp[prompt_of[i]] += tgt[i].astype(np.float64) - ref[i]
        cnt[prompt_of[i]] += 1
    Gt, Nt = Gp.sum(0), cnt.sum()
    per = {k: [[] for _ in range(P)] for k in
           ["psi_pl", "rho_pl", "part_pl", "psi_50", "rho_50", "part_50", "psi_w", "rho_w"]}
    for i in range(ref.shape[0]):
        p = prompt_of[i]
        b = ref[i].astype(np.float64); d = tgt[i].astype(np.float64) - b
        g = (Gt - Gp[p]) / (Nt - cnt[p])
        for tag, idx in [("pl", np.where(b > math.log(1e-6))[0]), ("50", np.argpartition(-b, 50)[:50])]:
            rdb, rdg, rgb = corr(b[idx], d[idx]), corr(g[idx], d[idx]), corr(g[idx], b[idx])
            per[f"psi_{tag}"][p].append(rdb); per[f"rho_{tag}"][p].append(rdg)
            per[f"part_{tag}"][p].append(partial(rdg, rdb, rgb))
        w = np.exp(b); w /= w.sum()
        per["psi_w"][p].append(wcorr(b, d, w)); per["rho_w"][p].append(wcorr(g, d, w))
    pm = {k: np.array([np.nanmean(v) for v in vs]) for k, vs in per.items()}
    res = {k: boot(v) for k, v in pm.items()}
    for tag in ["pl", "50", "w"]:
        res[f"rho_minus_psi_{tag}"] = boot(pm[f"rho_{tag}"] - pm[f"psi_{tag}"])
    print(f"  {label:<18}" + "  ".join(
        f"{k} {res[k][0]:+.3f} [{res[k][1]:+.3f},{res[k][2]:+.3f}]"
        for k in ["psi_pl", "rho_pl", "part_pl", "rho_minus_psi_pl"]))
    print(f"  {'':<18}" + "  ".join(
        f"{k} {res[k][0]:+.3f} [{res[k][1]:+.3f},{res[k][2]:+.3f}]"
        for k in ["psi_50", "rho_50", "psi_w", "rho_w"]))
    return res, Gt / Nt


def shares(Hs, prompt_of):
    P = int(prompt_of.max()) + 1
    pm = {s: np.array([Hs[s][prompt_of == p].mean() for p in range(P)]) for s in STAGES}
    total = pm["base"] - pm["inst"]
    rng = np.random.default_rng(0)
    steps = [("SFT", "base", "sft"), ("DPO", "sft", "dpo"), ("RLVR/final", "dpo", "inst")]
    out = dict(entropy={s: float(pm[s].mean()) for s in STAGES}, total=float(total.mean()))
    J = rng.integers(0, P, (20000, P))
    for name, a, b in steps:
        sh = (pm[a] - pm[b]).mean() / total.mean()
        bs = (pm[a][J] - pm[b][J]).mean(1) / total[J].mean(1)
        out[name] = dict(drop=float((pm[a] - pm[b]).mean()), share=float(sh),
                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        print(f"  {name:<11} drop {out[name]['drop']:+.4f}  share {100*sh:6.1f}%  "
              f"[{100*out[name]['ci'][0]:6.1f}%, {100*out[name]['ci'][1]:6.1f}%]")
    return out


def analyse(family):
    from transformers import AutoTokenizer
    F = FAMILIES[family]
    meta = {s: json.load(open(f"{D}/{family}_{s}.json")) for s in STAGES}
    a_ids = meta["base"]["A_input_ids"]
    assert all(meta[s]["A_input_ids"] == a_ids for s in STAGES), "arm-A input IDs differ across stages"
    tok = AutoTokenizer.from_pretrained(F["template"])
    rows = json.load(open(f"{D}/{family}_cont.json"))["rows"]
    prompt_B = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows]).astype(int)
    pairs = [("base", "sft"), ("sft", "dpo"), ("dpo", "inst"), ("base", "inst")]
    bpairs = pairs + ([("base", "basecpu")] if os.path.exists(f"{D}/{family}_B_basecpu.npy") else [])
    R = dict(family=family, models={s: meta[s]["model_id"] for s in STAGES},
             B_rows=int(len(prompt_B)), B_mean_cont_len=float(np.mean([len(r["cont_ids"]) for r in rows])),
             excluded_mass_max={s: max(max(meta[s]["A_excluded_mass"]), meta[s]["B_excluded_mass_max"])
                                for s in STAGES})
    print("=" * 100); print(f"REGISTER TEST v2 -- {family}: {list(R['models'].values())}"); print("=" * 100)

    # ---------------- arm A
    A = {s: np.load(f"{D}/{family}_A_{s}.npy") for s in STAGES}
    pA = np.arange(A["base"].shape[0])
    eos = meta["base"]["eos_token_id"]
    print("\nARM A -- chat-templated, assistant-start position (36 rows)")
    HA = {s: -(np.exp(A[s].astype(np.float64)) * A[s]).sum(1) for s in STAGES}
    R["A_shares"] = shares(HA, pA)
    R["A_eos_mass"] = {s: float(np.exp(A[s][:, eos]).mean()) for s in STAGES}
    print("  EOS mass:", {s: f"{v:.2e}" for s, v in R["A_eos_mass"].items()})
    R["A_geometry"] = {}
    for a, b in pairs:
        R["A_geometry"][f"{a}->{b}"], _ = geometry(A[a], A[b], pA, f"{a}->{b}")
    ctrl = os.path.exists(f"{D}/{family}_A_basecpu.npy")
    if ctrl:   # null control: same weights, other backend -- pure numerical noise, no register
        R["A_geometry"]["NULL base->basecpu"], _ = geometry(
            A["base"], np.load(f"{D}/{family}_A_basecpu.npy"), pA, "NULL base->cpu")

    # ---------------- arm B
    src = json.load(open(f"{D}/{family}_cont.json"))["source"]
    print(f"\nARM B -- continuations written by {src}, teacher-forced, {len(prompt_B)} positions")
    HB = {s: np.load(f"{D}/{family}_B_{s}_H.npy").astype(np.float64) for s in STAGES}
    R["B_shares"] = shares(HB, prompt_B)
    R["B_geometry"] = {}; R["B_top_register_tokens"] = {}
    for a, b in bpairs:
        ra = np.load(f"{D}/{family}_B_{a}.npy", mmap_mode="r"); rb = np.load(f"{D}/{family}_B_{b}.npy", mmap_mode="r")
        res, g = geometry(ra, rb, prompt_B, f"{a}->{b}")
        R["B_geometry"][f"{a}->{b}"] = res
        # lexical readout: largest shared shifts among tokens plausible under ref in >= 5% of positions
        common = (np.asarray(ra) > math.log(1e-4)).mean(0) >= 0.05
        idx = np.where(common)[0]
        up = idx[np.argsort(-g[idx])[:15]]; dn = idx[np.argsort(g[idx])[:15]]
        R["B_top_register_tokens"][f"{a}->{b}"] = dict(
            promoted=[(tok.decode([int(i)]), round(float(g[i]), 3)) for i in up],
            demoted=[(tok.decode([int(i)]), round(float(g[i]), 3)) for i in dn])
        print(f"  {'':<18}promoted: {[t for t, _ in R['B_top_register_tokens'][f'{a}->{b}']['promoted']]}")
        print(f"  {'':<18}demoted:  {[t for t, _ in R['B_top_register_tokens'][f'{a}->{b}']['demoted']]}")
        del ra, rb; gc.collect()

    out = os.path.normpath(os.path.dirname(D) + f"/../../results/results_register_{family}.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    cmd, fam = sys.argv[1], sys.argv[2]
    {"gen": lambda: gen(fam), "score": lambda: score(fam, sys.argv[3]), "analyse": lambda: analyse(fam)}[cmd]()
