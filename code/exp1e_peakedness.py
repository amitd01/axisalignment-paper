"""Experiment 1e: does the psi* threshold survive a realistic reference shape?

Why this exists
---------------
exp1d derives psi* = 0.71-0.83 on a Zipfian reference with exponent 1.1 over
4000 items. That distribution has effective branching ~218 and needs 18 tokens
to cover 50% of its mass.

The measured continuation-position distributions are far peakier: effective
branching 20-80, and a MEDIAN OF 2 TOKENS to cover 50% of base mass.

So psi* was derived off-regime. If the crossover moves materially under
realistic peakedness, the paper's central comparison needs restating. This
sweeps the Zipf exponent and the matched-diversity target, and reports psi*
on the same binary search exp1d used.

Everything else -- the operators, the bias construction, the matching
procedure -- is exp1d unchanged, so any movement is attributable to shape.
"""
import numpy as np, json

K = 4000


def make_ref(a):
    """Zipfian reference with exponent a, plus its shape diagnostics."""
    r = np.arange(1, K + 1)
    p = 1.0 / r ** a
    p /= p.sum()
    lp = np.log(p)
    tz = (lp - lp.mean()) / lp.std()
    H = float(-(p * np.log(p)).sum())
    head = int(np.searchsorted(np.cumsum(p), 0.5) + 1)
    return p, lp, tz, float(np.exp(H)), head


def z(x):
    return (x - x.mean()) / x.std()


def norm(s):
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def eff(p):
    return float(np.exp(ent(p)))


def build(tz, rho, psi, seed=11):
    r1 = np.random.default_rng(seed)
    r2 = np.random.default_rng(seed + 1)
    qz = z(rho * tz + np.sqrt(max(0, 1 - rho ** 2)) * z(r1.normal(0, 1, K)))
    b = z(psi * tz + np.sqrt(max(0, 1 - psi ** 2)) * z(r2.normal(0, 1, K)))
    return qz, b


def aligned(lp, qz, b, w_q, w_t):
    return norm(lp + w_q * qz + w_t * b)


def op_temp(lp, qz, b, T):
    return norm(np.log(aligned(lp, qz, b, 1.0, 1.0)) / T)


def op_dec(lp, qz, b, g):
    return norm(np.log(aligned(lp, qz, b, 1.0, 1.0)) - g * lp)


def qad(fn, lp, qz, b, target, lo, hi):
    """Binary-search the operator parameter to hit a matched effective support."""
    for _ in range(140):
        m = (lo + hi) / 2
        if eff(fn(lp, qz, b, m)) < target:
            lo = m
        else:
            hi = m
    m = (lo + hi) / 2
    return float((fn(lp, qz, b, m) * qz).sum())


def crossover(a, rho, target):
    """psi at which decontamination stops beating temperature."""
    p, lp, tz, _, _ = make_ref(a)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        qz, b = build(tz, rho, mid)
        eA = qad(op_temp, lp, qz, b, target, 1.0, 150.0)
        eC = qad(op_dec, lp, qz, b, target, 0.0, 60.0)
        if eC > eA:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


print("=" * 74)
print("Reference shape by Zipf exponent (K=4000)")
print("=" * 74)
print(f"{'exponent':>9} {'eff branching':>14} {'tokens to 50% mass':>20}")
shapes = {}
for a in [1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0]:
    _, _, _, br, head = make_ref(a)
    shapes[a] = (br, head)
    print(f"{a:>9.1f} {br:>14.1f} {head:>20d}")

print()
print("  measured continuation positions: branching 20-80, median 2 tokens to 50%")
print("  exp1d used exponent 1.1 -> branching 218, 18 tokens. Off-regime.")

# pick the exponents whose shape brackets the measured regime
real = [a for a in shapes if 15 <= shapes[a][0] <= 100]
print(f"\n  exponents matching the measured regime: {real}")

print()
print("=" * 74)
print("psi* under realistic peakedness")
print("matched-diversity target set to each reference's own branching / 2")
print("=" * 74)
print(f"{'exponent':>9} {'rho':>5} {'branching':>10} {'target':>8} {'psi*':>8}")
rows = []
for a in [1.1] + real:
    br, head = shapes[a]
    target = max(5.0, br / 2)
    for rho in [0.2, 0.4, 0.6]:
        ps = crossover(a, rho, target)
        rows.append(dict(exponent=a, rho=rho, branching=br, head=head,
                         target=target, psi_star=ps))
        print(f"{a:>9.1f} {rho:>5.1f} {br:>10.1f} {target:>8.1f} {ps:>8.3f}")
    print()

base = [r["psi_star"] for r in rows if r["exponent"] == 1.1]
realistic = [r["psi_star"] for r in rows if r["exponent"] != 1.1]
print("=" * 74)
print(f"exp1d regime (exponent 1.1):   psi* in [{min(base):.3f}, {max(base):.3f}]")
print(f"measured regime:               psi* in [{min(realistic):.3f}, {max(realistic):.3f}]")
print()
print(f"Max measured psi on the PLAUSIBLE set, any delta, any model: 0.234")
print("=" * 74)

json.dump({"shapes": {str(k): v for k, v in shapes.items()}, "rows": rows},
          open("/home/claude/scale/results_exp1e.json", "w"), indent=1)
print("saved -> results_exp1e.json")
