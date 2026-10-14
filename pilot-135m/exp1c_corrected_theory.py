"""
Experiment 1c: The corrected theory model. Supersedes 1 and 1b.

Fix applied
-----------
Experiments 1 and 1b assumed quality and typicality are INDEPENDENT. That is
too generous to the decontamination operator: it implies you can push
arbitrarily hard against typicality with no quality cost, and the optimum
runs off to infinity.

In real text they are positively correlated. Common phrasings are common partly
because they are fluent and coherent. The tail is not a reservoir of neglected
brilliance; it is mostly noise, with some brilliance in it.

So introduce rho = corr(quality, typicality), and treat it as the parameter
that decides everything.

    quality(k) = rho * typicality(k) + sqrt(1-rho^2) * eps(k)

Under this model all three operators have an interior optimum, and the question
becomes empirical: what is rho for a real model on a real task?
"""
import numpy as np
import json

rng = np.random.default_rng(20260803)
K, BETA = 4000, 1.0

ranks = np.arange(1, K + 1)
p_ref = 1.0 / ranks ** 1.1
p_ref /= p_ref.sum()
log_p_ref = np.log(p_ref)
tz = (log_p_ref - log_p_ref.mean()) / log_p_ref.std()


def make_quality(rho, seed=0):
    eps = np.random.default_rng(seed).normal(0, 1, K)
    eps = (eps - eps.mean()) / eps.std()
    q = rho * tz + np.sqrt(max(0.0, 1 - rho ** 2)) * eps
    return (q - q.mean()) / q.std()


def norm(s):
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def eff(p):
    return float(np.exp(ent(p)))


def Eq(p, qz):
    return float((p * qz).sum())


def aligned(qz, w_q, w_t, lam=1.0):
    return norm(log_p_ref + lam * (w_q * qz + w_t * tz) / BETA)


def op_temp(qz, w_q, w_t, T):
    return norm(np.log(aligned(qz, w_q, w_t)) / T)


def op_lam(qz, w_q, w_t, lam):
    return aligned(qz, w_q, w_t, lam=lam)


def op_dec(qz, w_q, w_t, g):
    return norm(np.log(aligned(qz, w_q, w_t)) - g * log_p_ref)


def qad(fn, qz, w_q, w_t, target, lo, hi, inc):
    """Bisect operator parameter to a target effective support."""
    for _ in range(140):
        mid = (lo + hi) / 2
        if (eff(fn(qz, w_q, w_t, mid)) < target) == inc:
            lo = mid
        else:
            hi = mid
    m = (lo + hi) / 2
    p = fn(qz, w_q, w_t, m)
    return m, eff(p), Eq(p, qz)


print("=" * 86)
print("EXPERIMENT 1c -- operators under quality/typicality correlation rho")
print("Matched at effective support = 200. w_q=1.0, w_t=1.0 unless stated.")
print("=" * 86)
print(f"{'rho':>6} {'temp par':>9} {'temp E[q]':>10} {'lam par':>9} {'lam E[q]':>10} "
      f"{'dec par':>9} {'dec E[q]':>10} {'winner':>14}")

rows = []
for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    qz = make_quality(rho, seed=7)
    r = {}
    pA, _, eA = qad(op_temp, qz, 1.0, 1.0, 200, 1.0, 120.0, True)
    pB, _, eB = qad(op_lam, qz, 1.0, 1.0, 200, 1.0, -12.0, False)
    pC, _, eC = qad(op_dec, qz, 1.0, 1.0, 200, 0.0, 40.0, True)
    win = max([("temperature", eA), ("lambda", eB), ("decontaminate", eC)],
              key=lambda x: x[1])[0]
    print(f"{rho:>6.1f} {pA:>9.3f} {eA:>10.4f} {pB:>9.3f} {eB:>10.4f} "
          f"{pC:>9.3f} {eC:>10.4f} {win:>14}")
    rows.append(dict(rho=rho, temp_param=pA, temp_Eq=eA, lam_param=pB, lam_Eq=eB,
                     dec_param=pC, dec_Eq=eC, winner=win))

# ------------------------------------------------ where is the crossover
print("\n" + "=" * 86)
print("CROSSOVER: the rho at which decontamination stops beating temperature")
print("=" * 86)
lo, hi = 0.0, 0.99
for _ in range(60):
    mid = (lo + hi) / 2
    qz = make_quality(mid, seed=7)
    _, _, eA = qad(op_temp, qz, 1.0, 1.0, 200, 1.0, 120.0, True)
    _, _, eC = qad(op_dec, qz, 1.0, 1.0, 200, 0.0, 40.0, True)
    if eC > eA:
        lo = mid
    else:
        hi = mid
print(f"  crossover rho* = {(lo+hi)/2:.3f}")
print("  Below this, decontamination wins. Above it, plain temperature wins.")
print("  => The entire practical question reduces to measuring rho.")

# ------------------------------------------------ optimal gamma is finite now
print("\n" + "=" * 86)
print("OPTIMAL DECONTAMINATION STRENGTH (unconstrained, maximising E[quality])")
print("=" * 86)
print(f"{'rho':>6} {'gamma*':>9} {'E[q] at gamma*':>16} {'eff.supp there':>16}")
gstar = []
for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
    qz = make_quality(rho, seed=7)
    grid = np.arange(0, 8.001, 0.02)
    vals = [(Eq(op_dec(qz, 1.0, 1.0, g), qz), g) for g in grid]
    bq, bg = max(vals)
    print(f"{rho:>6.1f} {bg:>9.2f} {bq:>16.4f} {eff(op_dec(qz,1.0,1.0,bg)):>16.1f}")
    gstar.append(dict(rho=rho, gamma_star=float(bg), Eq=float(bq)))

# ------------------------------------------------ sensitivity to bias weight
print("\n" + "=" * 86)
print("SENSITIVITY: gamma* against the true bias weight w_t (rho=0.4)")
print("Does the optimum track w_t, as the theory predicts?")
print("=" * 86)
qz = make_quality(0.4, seed=7)
print(f"{'w_t':>6} {'gamma*':>9} {'ratio g*/w_t':>14}")
for w_t in [0.25, 0.5, 1.0, 2.0, 4.0]:
    grid = np.arange(0, 10.001, 0.02)
    vals = [(Eq(op_dec(qz, 1.0, w_t, g), qz), g) for g in grid]
    bq, bg = max(vals)
    print(f"{w_t:>6.2f} {bg:>9.2f} {bg/w_t:>14.3f}")

with open("/home/claude/results_exp1c.json", "w") as f:
    json.dump({"matched_frontier": rows, "gamma_star": gstar,
               "crossover_rho": (lo + hi) / 2}, f, indent=1)
print("\nsaved -> results_exp1c.json")
