"""
Experiment 1d: Robustness. What if the typicality bias is only PARTIALLY
observable through the reference model's log-probability?

In 1c the bias term was exactly proportional to log p_ref, so decontamination
could cancel it exactly. That is the best possible case for the operator.
Real preference data rewards a bundle -- fluency, familiarity, formatting,
politeness -- of which reference log-prob is only a proxy.

Model: bias signal b = psi * tz + sqrt(1-psi^2) * noise
psi = 1 is the 1c case (perfectly observable). psi = 0 means the reference
model tells you nothing about the bias.
"""
import numpy as np, json
K = 4000
ranks = np.arange(1, K+1)
p_ref = 1.0/ranks**1.1; p_ref /= p_ref.sum()
log_p_ref = np.log(p_ref)
tz = (log_p_ref-log_p_ref.mean())/log_p_ref.std()
def z(x): return (x-x.mean())/x.std()
def norm(s):
    s=s-s.max(); e=np.exp(s); return e/e.sum()
def ent(p):
    p=p[p>0]; return float(-(p*np.log(p)).sum())
def eff(p): return float(np.exp(ent(p)))

def build(rho, psi, seed=11):
    r1=np.random.default_rng(seed); r2=np.random.default_rng(seed+1)
    qz = z(rho*tz + np.sqrt(max(0,1-rho**2))*z(r1.normal(0,1,K)))
    b  = z(psi*tz + np.sqrt(max(0,1-psi**2))*z(r2.normal(0,1,K)))
    return qz, b

def aligned(qz,b,w_q,w_t): return norm(log_p_ref + w_q*qz + w_t*b)
def op_temp(qz,b,w_q,w_t,T): return norm(np.log(aligned(qz,b,w_q,w_t))/T)
def op_dec(qz,b,w_q,w_t,g):  return norm(np.log(aligned(qz,b,w_q,w_t)) - g*log_p_ref)
def Eq(p,qz): return float((p*qz).sum())

def qad(fn,qz,b,w_q,w_t,target,lo,hi,inc):
    for _ in range(140):
        m=(lo+hi)/2
        if (eff(fn(qz,b,w_q,w_t,m))<target)==inc: lo=m
        else: hi=m
    m=(lo+hi)/2; p=fn(qz,b,w_q,w_t,m); return m, Eq(p,qz)

print("="*88)
print("EXPERIMENT 1d -- decontamination under bias mis-specification")
print("psi = how well reference log-prob proxies the actual preference bias")
print("Matched at effective support 200. w_q=1.0, w_t=1.0.")
print("="*88)
print(f"{'rho':>5} {'psi':>5} {'temp E[q]':>11} {'decon E[q]':>12} {'advantage':>11} {'winner':>15}")
rows=[]
for rho in [0.2,0.4,0.6]:
    for psi in [1.0,0.8,0.6,0.4,0.2,0.0]:
        qz,b=build(rho,psi)
        _,eA=qad(op_temp,qz,b,1.0,1.0,200,1.0,150.0,True)
        _,eC=qad(op_dec ,qz,b,1.0,1.0,200,0.0,60.0,True)
        w = "decontaminate" if eC>eA else "temperature"
        print(f"{rho:>5.1f} {psi:>5.1f} {eA:>11.4f} {eC:>12.4f} {eC-eA:>+11.4f} {w:>15}")
        rows.append(dict(rho=rho,psi=psi,temp=eA,dec=eC,adv=eC-eA,winner=w))
    print()

print("="*88)
print("CROSSOVER psi* per rho (below this psi, decontamination stops helping)")
print("="*88)
cross={}
for rho in [0.2,0.4,0.6]:
    lo,hi=0.0,1.0
    for _ in range(50):
        mid=(lo+hi)/2
        qz,b=build(rho,mid)
        _,eA=qad(op_temp,qz,b,1.0,1.0,200,1.0,150.0,True)
        _,eC=qad(op_dec ,qz,b,1.0,1.0,200,0.0,60.0,True)
        if eC>eA: hi=mid
        else: lo=mid
    cross[rho]=(lo+hi)/2
    print(f"  rho={rho}:  psi* = {(lo+hi)/2:.3f}")
print("\n  Interpretation: the reference model must explain at least this much")
print("  of the preference bias for decontamination to beat plain temperature.")
json.dump({"rows":rows,"crossover_psi":cross}, open("results_exp1d.json","w"), indent=1)
print("\nsaved -> results_exp1d.json")
