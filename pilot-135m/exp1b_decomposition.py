"""
Experiment 1b: Why naive delta reversal fails, and what to do instead.

Experiment 1 found that lowering lambda below 0 raises entropy but LOWERS
expected quality, and is dominated by simple temperature on the
diversity/quality frontier. The reason is structural:

    r(k) = w_q * quality(k) + w_t * typicality(k)

Reversing lambda reverses the WHOLE reward. It suppresses the quality signal
along with the typicality signal. You cannot subtract the bias by subtracting
the vector that contains it.

The alternative is decomposition rather than reversal. The typicality component
is not latent -- it is proportional to the reference model's own log-prob, which
is directly observable. So define a decontamination operator:

    p_gamma(k)  propto  p_align(k) * p_ref(k)^(-gamma)

This removes typicality while leaving the quality component of the reward
intact. Note that this is exactly the shape of contrastive decoding
(arXiv:2210.15097): expert log-prob minus amateur log-prob.

This script compares three operators on identical distributions:
  (A) temperature on the aligned model
  (B) lambda reversal  (DeRa / proxy-tuning, negative direction)
  (C) typicality decontamination  (contrastive-decoding shape)
"""
import numpy as np
import json

rng = np.random.default_rng(20260803)
K, BETA = 2000, 1.0

ranks = np.arange(1, K + 1)
p_ref = 1.0 / ranks ** 1.1
p_ref /= p_ref.sum()
log_p_ref = np.log(p_ref)

q = rng.normal(0.0, 1.0, K)
qz = (q - q.mean()) / q.std()
tz = (log_p_ref - log_p_ref.mean()) / log_p_ref.std()


def norm(s):
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def eff(p):
    return float(np.exp(ent(p)))


def Eq(p):
    return float((p * qz).sum())


def topq(p, m=100):
    return float(p[np.argsort(-qz)[:m]].sum())


def make_aligned(w_q, w_t, lam=1.0):
    return norm(log_p_ref + lam * (w_q * qz + w_t * tz) / BETA)


# ------------------------------------------------------- the three operators
def op_temperature(w_q, w_t, T):
    p = make_aligned(w_q, w_t, lam=1.0)
    return norm(np.log(p) / T)


def op_lambda(w_q, w_t, lam):
    return make_aligned(w_q, w_t, lam=lam)


def op_decontaminate(w_q, w_t, gamma):
    """p_align * p_ref^(-gamma), i.e. contrastive-decoding shape."""
    p = make_aligned(w_q, w_t, lam=1.0)
    return norm(np.log(p) - gamma * log_p_ref)


# ------------------------------------------------------------------- sweeps
def frontier(fn, grid, w_q, w_t):
    out = []
    for g in grid:
        p = fn(w_q, w_t, g)
        out.append((float(g), eff(p), Eq(p), topq(p)))
    return out


scenarios = {
    "mild   (w_q=1.0, w_t=0.5)":  (1.0, 0.5),
    "strong (w_q=1.0, w_t=1.5)":  (1.0, 1.5),
    "severe (w_q=0.5, w_t=2.0)":  (0.5, 2.0),
}

T_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
L_grid = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0]
G_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

print("=" * 84)
print("EXPERIMENT 1b -- three operators on the diversity/quality frontier")
print("=" * 84)

all_res = {}
for name, (w_q, w_t) in scenarios.items():
    print(f"\n################  {name}  ################")
    base = make_aligned(w_q, w_t, 1.0)
    print(f"aligned model at rest: eff.supp={eff(base):.1f}  "
          f"E[q]={Eq(base):.4f}  topQcov={topq(base):.4f}")

    res = {}
    for label, fn, grid in [("A: temperature", op_temperature, T_grid),
                            ("B: lambda reversal", op_lambda, L_grid),
                            ("C: decontamination", op_decontaminate, G_grid)]:
        rows = frontier(fn, grid, w_q, w_t)
        res[label] = rows
        print(f"\n  {label}")
        print(f"  {'param':>8} {'eff.supp':>10} {'E[quality]':>12} {'topQcov':>10}")
        for g, e, eq, tq in rows:
            print(f"  {g:>8.2f} {e:>10.1f} {eq:>12.4f} {tq:>10.4f}")
    all_res[name] = res

# ------------------------------------------- matched-diversity head-to-head
print("\n" + "=" * 84)
print("MATCHED-DIVERSITY COMPARISON")
print("At equal effective support, which operator retains the most quality?")
print("=" * 84)


def quality_at_diversity(fn, w_q, w_t, target_eff, lo, hi, increasing):
    """Bisect the operator parameter to hit a target effective support."""
    for _ in range(80):
        mid = (lo + hi) / 2
        e = eff(fn(w_q, w_t, mid))
        if (e < target_eff) == increasing:
            lo = mid
        else:
            hi = mid
    p = fn(w_q, w_t, (lo + hi) / 2)
    return (lo + hi) / 2, eff(p), Eq(p), topq(p)


for name, (w_q, w_t) in scenarios.items():
    print(f"\n--- {name} ---")
    for target in [50, 150, 400]:
        print(f"  target effective support = {target}")
        specs = [
            ("A temperature   ", op_temperature, 1.0, 30.0, True),
            ("B lambda        ", op_lambda, 1.0, -6.0, False),
            ("C decontaminate ", op_decontaminate, 0.0, 3.0, True),
        ]
        best = None
        for lbl, fn, lo, hi, inc in specs:
            try:
                g, e, eq, tq = quality_at_diversity(fn, w_q, w_t, target, lo, hi, inc)
                mark = ""
                if best is None or eq > best[1]:
                    best = (lbl, eq)
                print(f"    {lbl} param={g:>7.3f}  eff={e:>7.1f}  "
                      f"E[q]={eq:>8.4f}  topQcov={tq:.4f}")
            except Exception as ex:
                print(f"    {lbl} failed: {ex}")
        print(f"    -> best quality at matched diversity: {best[0].strip()} "
              f"(E[q]={best[1]:.4f})")

# ------------------------------------------------ where does reversal ever win
print("\n" + "=" * 84)
print("WHEN DOES LAMBDA REVERSAL BEAT TEMPERATURE?")
print("Sweeping the bias ratio w_t/w_q, matched at effective support = 150")
print("=" * 84)
print(f"{'w_t/w_q':>9} {'A temp E[q]':>13} {'B lam E[q]':>12} {'C decon E[q]':>14} {'winner':>16}")

ratio_rows = []
for ratio in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e6]:
    if ratio > 1e5:
        w_q, w_t = 0.0, 2.0
        lbl = "inf"
    else:
        w_q, w_t = 1.0, ratio
        lbl = f"{ratio:.2f}"
    vals = {}
    for name, fn, lo, hi, inc in [
            ("A", op_temperature, 1.0, 60.0, True),
            ("B", op_lambda, 1.0, -8.0, False),
            ("C", op_decontaminate, 0.0, 4.0, True)]:
        try:
            _, _, eq, _ = quality_at_diversity(fn, w_q, w_t, 150, lo, hi, inc)
            vals[name] = eq
        except Exception:
            vals[name] = float("nan")
    win = max(vals, key=lambda k: vals[k])
    names = {"A": "temperature", "B": "lambda", "C": "decontaminate"}
    print(f"{lbl:>9} {vals['A']:>13.4f} {vals['B']:>12.4f} {vals['C']:>14.4f} "
          f"{names[win]:>16}")
    ratio_rows.append(dict(ratio=lbl, **vals, winner=names[win]))

with open("/home/claude/results_exp1b.json", "w") as f:
    json.dump({"frontiers": all_res, "ratio_sweep": ratio_rows}, f, indent=1)
print("\nsaved -> results_exp1b.json")
