"""
Experiment 1: Analytical model of typicality bias and delta reversal.

Setup
-----
A response space of K items. Each has:
  - base log-probability  log p_ref(k)   (what the pretrained model believes)
  - a latent quality      q(k)           (what we actually want)

Typicality bias: human preference raters reward typicality, not just quality.
    r(k) = w_q * q(k) + w_t * log p_ref(k)
with w_t > 0 encoding the bias documented in arXiv:2510.01171.

KL-regularised alignment (the standard RLHF objective) has closed form:
    p_align(k)  propto  p_ref(k) * exp(r(k)/beta)

DeRa (arXiv:2402.02992) proves aligned models at varying KL strength are
geometric mixtures of reference and aligned, differing only in mixing weight:
    p_lam(k)  propto  p_ref(k)^(1-lam) * p_align(k)^lam

Substituting gives the identity this whole experiment turns on:
    p_lam(k)  propto  p_ref(k) * exp(lam * r(k) / beta)

So lambda is exactly a rescaling of the reward's strength. lam=1 recovers the
released aligned model. lam=0 recovers the reference. lam<0 applies the reward
with reversed sign -- it actively promotes what alignment suppressed.

This script computes, exactly, what that does to entropy, tail mass, and
expected quality.
"""
import numpy as np
import json

rng = np.random.default_rng(20260803)

K = 2000          # response space size
BETA = 1.0        # KL regularisation strength

# ---------------------------------------------------------------- construct
# Reference distribution: Zipfian. A few very typical responses, a long tail.
ranks = np.arange(1, K + 1)
p_ref = 1.0 / ranks ** 1.1
p_ref /= p_ref.sum()
log_p_ref = np.log(p_ref)

# Latent quality: deliberately INDEPENDENT of typicality.
# Good ideas are not more likely to be common ones. This is the crux.
q = rng.normal(0.0, 1.0, K)

# Standardise both so the weights w_q / w_t are directly comparable
qz = (q - q.mean()) / q.std()
tz = (log_p_ref - log_p_ref.mean()) / log_p_ref.std()


def aligned_logp(w_q, w_t, beta=BETA, lam=1.0):
    """log p_lam up to a constant."""
    r = w_q * qz + w_t * tz
    s = log_p_ref + lam * r / beta
    s -= s.max()
    p = np.exp(s)
    return p / p.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def tail_mass(p, frac=0.5):
    """Probability mass sitting outside the most-typical `frac` of the ref dist."""
    cut = int(K * frac)
    return float(p[cut:].sum())


def eff_support(p):
    """Perplexity of the distribution = effective number of distinct answers."""
    return float(np.exp(entropy(p)))


def exp_quality(p):
    return float((p * qz).sum())


def coverage(p, topm=100):
    """Mass on the m highest-quality items, regardless of typicality."""
    idx = np.argsort(-qz)[:topm]
    return float(p[idx].sum())


# ---------------------------------------------------------------- scenarios
scenarios = {
    "no_typicality_bias":   dict(w_q=1.0, w_t=0.0),
    "mild_typicality_bias": dict(w_q=1.0, w_t=0.5),
    "strong_typicality_bias": dict(w_q=1.0, w_t=1.5),
    "pure_typicality_bias": dict(w_q=0.0, w_t=1.5),
}

lams = np.round(np.arange(-1.0, 2.01, 0.125), 4)

results = {}
for name, w in scenarios.items():
    rows = []
    for lam in lams:
        p = aligned_logp(lam=lam, **w)
        rows.append(dict(
            lam=float(lam),
            entropy=entropy(p),
            eff_support=eff_support(p),
            tail_mass=tail_mass(p),
            exp_quality=exp_quality(p),
            top_quality_coverage=coverage(p),
        ))
    results[name] = rows

# ---------------------------------------------------------------- reporting
print("=" * 78)
print("EXPERIMENT 1 -- Typicality bias and the lambda operator (exact, K=%d)" % K)
print("=" * 78)

ref_ent = entropy(p_ref)
print(f"\nReference distribution: entropy={ref_ent:.3f} nats, "
      f"effective support={np.exp(ref_ent):.1f} of {K}")

for name, rows in results.items():
    w = scenarios[name]
    print(f"\n--- {name}  (w_q={w['w_q']}, w_t={w['w_t']}) ---")
    print(f"{'lambda':>8} {'entropy':>9} {'eff.supp':>10} {'tailmass':>10} "
          f"{'E[quality]':>11} {'topQcov':>9}")
    for r in rows:
        if abs(r["lam"] * 4 - round(r["lam"] * 4)) < 1e-9 and \
           abs(r["lam"] - round(r["lam"] * 4) / 4) < 1e-9 and \
           (round(r["lam"] * 4) % 1 == 0) and (abs(r["lam"]) * 4 % 1 == 0) and \
           (r["lam"] in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]):
            print(f"{r['lam']:>8.2f} {r['entropy']:>9.3f} {r['eff_support']:>10.1f} "
                  f"{r['tail_mass']:>10.4f} {r['exp_quality']:>11.4f} "
                  f"{r['top_quality_coverage']:>9.4f}")

# ---------------------------------------------------------------- key result
print("\n" + "=" * 78)
print("KEY QUANTITIES")
print("=" * 78)

for name, rows in results.items():
    d = {r["lam"]: r for r in rows}
    a, r0 = d[1.0], d[0.0]
    print(f"\n{name}:")
    print(f"  alignment (lam 0 -> 1) changes effective support "
          f"{r0['eff_support']:.1f} -> {a['eff_support']:.1f} "
          f"({100*(a['eff_support']/r0['eff_support']-1):+.1f}%)")
    print(f"  alignment changes E[quality] "
          f"{r0['exp_quality']:.4f} -> {a['exp_quality']:.4f}")
    # what does reversal buy
    for lam in [-0.5, -0.25, 0.25]:
        x = d[lam]
        print(f"  lam={lam:>5}: eff.supp={x['eff_support']:>7.1f}  "
              f"E[q]={x['exp_quality']:>7.4f}  topQcov={x['top_quality_coverage']:.4f}")

# Where does quality peak vs where does diversity peak?
print("\n" + "=" * 78)
print("THE FRONTIER: does lambda trade diversity against quality differently")
print("than temperature does?")
print("=" * 78)


def temp_dist(p, T):
    s = np.log(p) / T
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()


for name in ["mild_typicality_bias", "strong_typicality_bias"]:
    w = scenarios[name]
    p_al = aligned_logp(lam=1.0, **w)
    print(f"\n--- {name} ---")
    print("  lambda path (from the aligned model, T=1):")
    for lam in [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5]:
        p = aligned_logp(lam=lam, **w)
        print(f"    lam={lam:>5.2f}  eff.supp={eff_support(p):>8.1f}  "
              f"E[q]={exp_quality(p):>8.4f}  topQcov={coverage(p):.4f}")
    print("  temperature path (from the same aligned model):")
    for T in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        p = temp_dist(p_al, T)
        print(f"    T  ={T:>5.2f}  eff.supp={eff_support(p):>8.1f}  "
              f"E[q]={exp_quality(p):>8.4f}  topQcov={coverage(p):.4f}")

with open("/home/claude/results_exp1.json", "w") as f:
    json.dump({"scenarios": scenarios, "results": results,
               "ref_entropy": ref_ent, "K": K}, f, indent=1)
print("\nsaved -> results_exp1.json")
