"""Specification grid for the §5.1 attribution (v1.0 review response; exploratory, not pre-registered).

Asks whether "a small part of the shift carries nearly all of the entropy loss" survives the
analyst's choices, and reports both halves of that sentence under the SAME specification:

  R2_typ   share of the shift's variance over the plausible set explained by a linear fit on
           reference log-prob (weighted R2 when the fit is weighted)
  typ/shared/resid/out   Shapley shares of the entropy loss, exactly as decomp.py computes them

Grid (12 specifications; the pre-registered one is w / 1e-6 / prompt):
  weighting  u (unweighted) | w (reference-probability weighted)
  plausible  p_ref > 1e-5 | 1e-6 | 1e-7
  g          prompt: shared direction = mean shift on all OTHER prompts (decomp.py, leave-one-prompt-out)
             family: mean shift on prompts of OTHER task families only (held-out direction; a
                     same-family formatting habit cannot leak into it)

Uncertainty: bootstrap over prompts, stratified by task family (each family's six prompts are
resampled within the family). Positions are averaged within a prompt first, so prompts weigh equally.
The fitted components are re-estimated at every position, so they need no refitting per replicate;
g is computed once on the full sample (see the paper's §5 note).

Also reports, for the pre-registered specification, the shares within each task family and with
each family left out.

Usage: python robustness.py <family> [<family> ...] [--dumps DIR] [--stride K]
       --stride K scores every K-th position (quick look); default 1 = all positions.
"""
import os, sys, json, math, argparse
import numpy as np

from decomp import D as DEFAULT_DUMPS, RES, PAIRS, ent, shapley, fit
from prompts import FLAT

WEIGHTS = ["u", "w"]
THRESH = {"1e-5": 1e-5, "1e-6": 1e-6, "1e-7": 1e-7}
GMODE = ["prompt", "family"]
PRIMARY = ("w", "1e-6", "prompt")
PARTS = ["typ", "shared", "resid", "out"]
FAM_NAMES = sorted({f for f, _ in FLAT})


def r2_typ(d, b, w=None):
    """Variance share of d explained by a + c*b, weighted when w is given."""
    X = np.stack([np.ones_like(b), b], 1)
    w = np.ones_like(b) if w is None else w / w.sum()
    sw = np.sqrt(w)
    c = np.linalg.lstsq(X * sw[:, None], d * sw, rcond=None)[0]
    mu = (w * d).sum() / w.sum()
    ss_tot = (w * (d - mu) ** 2).sum()
    return 1.0 - (w * (d - X @ c) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0


def shares(per_prompt_loss, per_prompt_part, idx):
    """Share of the mean entropy loss carried by a part, over the prompts in idx (any shape [..., n])."""
    return per_prompt_part[idx].mean(-1) / per_prompt_loss[idx].mean(-1)


def stratified_draws(fam_of_prompt, n_boot, seed=0):
    """[n_boot, P] prompt indices, resampled within each task family."""
    rng = np.random.default_rng(seed)
    cols = []
    for f in np.unique(fam_of_prompt):
        members = np.flatnonzero(fam_of_prompt == f)
        cols.append(rng.choice(members, (n_boot, len(members))))
    return np.concatenate(cols, 1)


def run_pair(ref, tgt, prompt_of, fam_of_prompt, stride):
    P, V = len(fam_of_prompt), ref.shape[1]
    # per-prompt summed shift and counts, for the two held-out shared directions
    Gp = np.zeros((P, V)); cnt = np.zeros(P)
    for i in range(ref.shape[0]):
        Gp[prompt_of[i]] += tgt[i].astype(np.float64) - ref[i]; cnt[prompt_of[i]] += 1
    Gt, Nt = Gp.sum(0), cnt.sum()
    Gf = {f: Gp[fam_of_prompt == f].sum(0) for f in np.unique(fam_of_prompt)}
    Nf = {f: cnt[fam_of_prompt == f].sum() for f in np.unique(fam_of_prompt)}

    specs = [(wt, th, gm) for wt in WEIGHTS for th in THRESH for gm in GMODE]
    acc = {s: {k: [[] for _ in range(P)] for k in ["r2"] + PARTS} for s in specs}
    dH_acc = [[] for _ in range(P)]
    max_err = 0.0
    for i in range(0, ref.shape[0], stride):
        p = prompt_of[i]; f = fam_of_prompt[p]
        b = ref[i].astype(np.float64); d = tgt[i].astype(np.float64) - b
        g_dir = {"prompt": (Gt - Gp[p]) / (Nt - cnt[p]),
                 "family": (Gt - Gf[f]) / (Nt - Nf[f])}
        pr = np.exp(b - b.max()); pr /= pr.sum()
        dH_acc[p].append(ent(b + d) - ent(b))
        for wt, th, gm in specs:
            S = b > math.log(THRESH[th])
            w = pr[S] if wt == "w" else None
            g = g_dir[gm]
            c = fit(d[S], b[S], g[S], w)
            typ = np.zeros(V); sh = np.zeros(V)
            typ[S] = c[1] * b[S]; sh[S] = c[2] * g[S]
            rs = np.where(S, d - typ - sh, 0.0)
            phi, dH = shapley([typ, sh, rs, np.where(S, 0.0, d)], b)
            max_err = max(max_err, abs(phi.sum() - dH))
            a = acc[(wt, th, gm)]
            a["r2"][p].append(r2_typ(d[S], b[S], w))
            for k, v in zip(PARTS, phi):
                a[k][p].append(v)

    keep = np.array([len(x) > 0 for x in dH_acc])
    loss = -np.array([np.mean(x) if x else np.nan for x in dH_acc])
    pm = {s: {k: np.array([np.mean(x) if x else np.nan for x in a[k]]) for k in a} for s, a in acc.items()}
    return loss, pm, keep, max_err


def summarise(loss, pm, keep, fam_of_prompt, n_boot):
    idx_all = np.flatnonzero(keep)
    J = stratified_draws(fam_of_prompt[keep], n_boot)
    J = idx_all[J]                                    # map back to prompt ids
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    grid = {}
    for (wt, th, gm), m in pm.items():
        row = {"r2_typ": dict(median=float(np.nanmedian(m["r2"][keep])),
                              ci=ci(np.nanmedian(m["r2"][J], 1)))}
        for k in PARTS:
            row[k] = dict(share=float(shares(loss, -m[k], idx_all)), ci=ci(shares(loss, -m[k], J)))
        grid[f"{wt}/{th}/{gm}"] = row
    # per-family and leave-one-family-out, pre-registered specification only
    m = pm[PRIMARY]
    by_fam, lofo = {}, {}
    for f in FAM_NAMES:
        inside = idx_all[fam_of_prompt[idx_all] == f]
        outside = idx_all[fam_of_prompt[idx_all] != f]
        by_fam[f] = {k: float(shares(loss, -m[k], inside)) for k in PARTS}
        by_fam[f]["loss_nats"] = float(loss[inside].mean())
        lofo[f] = {k: float(shares(loss, -m[k], outside)) for k in PARTS}
    return grid, by_fam, lofo


def run(family, dumps, stride, n_boot):
    rows = json.load(open(f"{dumps}/{family}_cont.json"))["rows"]
    prompt_of = np.concatenate([[r["prompt_idx"]] * len(r["cont_ids"]) for r in rows]).astype(int)
    fam_of_prompt = np.array([FLAT[p][0] for p in range(prompt_of.max() + 1)])
    out = {}
    for a, t in PAIRS:
        fa, ft = f"{dumps}/{family}_B_{a}.npy", f"{dumps}/{family}_B_{t}.npy"
        if not (os.path.exists(fa) and os.path.exists(ft)):
            continue                                  # e.g. families without an RLVR stage
        ref, tgt = np.load(fa, mmap_mode="r"), np.load(ft, mmap_mode="r")
        loss, pm, keep, max_err = run_pair(ref, tgt, prompt_of, fam_of_prompt, stride)
        grid, by_fam, lofo = summarise(loss, pm, keep, fam_of_prompt, n_boot)
        out[f"{a}->{t}"] = dict(entropy_loss=float(np.nanmean(loss)), max_shapley_error=max_err,
                                stride=stride, grid=grid, by_family=by_fam, leave_one_family_out=lofo)
        report(family, f"{a}->{t}", out[f"{a}->{t}"])
    tag = "" if stride == 1 else f"_s{stride}"
    json.dump(out, open(f"{RES}/results_robustness{tag}_{family}.json", "w"), indent=1)


def report(family, pair, r):
    pct = lambda x: f"{100 * x:7.1f}"
    print(f"\n{family} {pair}: entropy loss {r['entropy_loss']:+.4f} nats, "
          f"max Shapley error {r['max_shapley_error']:.1e}")
    print(f"  {'spec':<18}{'R2_typ':>8}{'typ':>8}{'  [95% CI]':<18}{'shared':>8}{'resid':>8}{'out':>8}")
    for spec, row in r["grid"].items():
        mark = " *" if tuple(spec.split("/")) == PRIMARY else "  "
        lo, hi = row["typ"]["ci"]
        print(f"{mark}{spec:<18}{pct(row['r2_typ']['median'])}{pct(row['typ']['share'])}"
              f"  [{100*lo:5.1f},{100*hi:6.1f}]  {pct(row['shared']['share'])}"
              f"{pct(row['resid']['share'])}{pct(row['out']['share'])}")
    print(f"  {'(pre-registered spec)':<20}{'typ':>8}{'shared':>8}{'| LOFO typ':>11}{'loss nats':>11}")
    for f in FAM_NAMES:
        bf, lf = r["by_family"][f], r["leave_one_family_out"][f]
        print(f"  {f:<20}{pct(bf['typ'])}{pct(bf['shared'])}  {pct(lf['typ'])}  {bf['loss_nats']:+9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="+")
    ap.add_argument("--dumps", default=DEFAULT_DUMPS)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()
    for fam in args.families:
        run(fam, args.dumps, args.stride, args.boot)
