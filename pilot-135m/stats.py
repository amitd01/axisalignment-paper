"""Per-prompt bootstrap on the headline comparison, plus entropy accounting."""
import os,json,math,itertools
os.environ.setdefault("HF_HOME","/home/claude/hf")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(1)
jt=AutoTokenizer.from_pretrained("openai-community/gpt2")
jm=AutoModelForCausalLM.from_pretrained("openai-community/gpt2",dtype=torch.float32).eval()
def jac(ss):
    S=[set(s.split()) for s in ss]; v=[]
    for i,j in itertools.combinations(range(len(S)),2):
        u=len(S[i]|S[j]); v.append(len(S[i]&S[j])/u if u else 0)
    return float(np.mean(v))
@torch.no_grad()
def ppl(ss):
    o=[]
    for s in ss:
        s=s.strip()
        if len(s)<5: continue
        ids=jt(s,return_tensors="pt")["input_ids"]
        if ids.shape[1]<2: continue
        l=jm(ids,labels=ids).loss.item()
        if math.isfinite(l): o.append(math.exp(min(l,15)))
    return float(np.median(o)) if o else np.nan
def load(t):
    d=json.load(open(f"gen_{t}.json")); return d["gens"]
# Pair conditions with comparable diversity: T=1.3 (jac .032) vs lam=-1.0 (jac .033)
print("="*80)
print("PAIRED COMPARISON at matched diversity: temp T=1.30  vs  lambda=-1.00")
print("="*80)
A,B=load("T13"),load("Lm100")
print(f"{'prompt':<46} {'jacA':>6} {'jacB':>6} {'pplA':>9} {'pplB':>9}")
dA,dB,pA,pB=[],[],[],[]
for pr in A:
    ja,jb=jac(A[pr]),jac(B[pr]); pa,pb=ppl(A[pr]),ppl(B[pr])
    dA.append(ja); dB.append(jb); pA.append(pa); pB.append(pb)
    print(f"{pr[:44]:<46} {ja:>6.3f} {jb:>6.3f} {pa:>9.1f} {pb:>9.1f}")
dA,dB,pA,pB=map(np.array,(dA,dB,pA,pB))
print(f"\nmean jaccard: temp={dA.mean():.4f}  lambda={dB.mean():.4f}  (matched: diff={dA.mean()-dB.mean():+.4f})")
print(f"median ppl  : temp={np.median(pA):.1f}   lambda={np.median(pB):.1f}")
print(f"log-ppl ratio per prompt: {np.round(np.log(pA/pB),2)}")
rng=np.random.default_rng(0); n=len(pA)
boot=[np.mean(np.log(pA[i]/pB[i])) for i in [rng.integers(0,n,n) for _ in range(20000)]]
lo,hi=np.percentile(boot,[2.5,97.5])
print(f"\nbootstrap mean log-ppl advantage of lambda: {np.mean(np.log(pA/pB)):.3f}")
print(f"  95% CI [{lo:.3f}, {hi:.3f}]   -> ratio {math.exp(np.mean(np.log(pA/pB))):.2f}x "
      f"[{math.exp(lo):.2f}, {math.exp(hi):.2f}]")
print(f"  prompts where lambda wins: {(pB<pA).sum()}/{n}")
try:
    from scipy.stats import wilcoxon
    print(f"  Wilcoxon signed-rank p = {wilcoxon(pA,pB).pvalue:.4f}  (n={n}, min possible p=0.0625)")
except Exception as e: print(" scipy:",e)

print("\n"+"="*80); print("ENTROPY ACCOUNTING: what alignment costs, in nats"); print("="*80)
B_="HuggingFaceTB/SmolLM2-135M"; A_="HuggingFaceTB/SmolLM2-135M-Instruct"
tok=AutoTokenizer.from_pretrained(B_)
mb=AutoModelForCausalLM.from_pretrained(B_,dtype=torch.float32).eval()
ma=AutoModelForCausalLM.from_pretrained(A_,dtype=torch.float32).eval()
PR=list(A.keys())
eb,ea=[],[]
with torch.no_grad():
    for p in PR:
        ids=tok(p,return_tensors="pt")["input_ids"]
        lb=torch.log_softmax(mb(ids).logits[0,-1].float(),-1)
        la=torch.log_softmax(ma(ids).logits[0,-1].float(),-1)
        eb.append(float(-(lb.exp()*lb).sum())); ea.append(float(-(la.exp()*la).sum()))
eb,ea=np.array(eb),np.array(ea)
for p,x,y in zip(PR,eb,ea): print(f"  {p[:46]:<48} base={x:.3f}  aligned={y:.3f}  drop={x-y:+.3f}")
print(f"\n  mean entropy: base={eb.mean():.3f}  aligned={ea.mean():.3f}  mean drop={(eb-ea).mean():.3f} nats")
print(f"  effective branching factor: base={np.exp(eb).mean():.1f} -> aligned={np.exp(ea).mean():.1f} "
      f"({100*(np.exp(ea).mean()/np.exp(eb).mean()-1):+.1f}%)")
json.dump(dict(pplA=pA.tolist(),pplB=pB.tolist(),jacA=dA.tolist(),jacB=dB.tolist(),
   ent_base=eb.tolist(),ent_align=ea.tolist(),ci=[lo,hi]),open("results_stats.json","w"),indent=1)
print("\nsaved -> results_stats.json")
