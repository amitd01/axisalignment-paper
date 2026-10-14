"""Score the 6-condition core on prompts complete across ALL of those conditions.
Adds a paired bootstrap of lambda vs temperature at matched diversity."""
import os,json,glob,math,itertools,re
os.environ.setdefault("HF_HOME","/home/claude/hf")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(1)   # leave a core for the grinder
jt=AutoTokenizer.from_pretrained("openai-community/gpt2")
jm=AutoModelForCausalLM.from_pretrained("openai-community/gpt2",dtype=torch.float32).eval()
CORE=["T13","T16","Lm050","Lm100","Lm150","X100"]
LBL={"T13":"A temp T=1.3","T16":"A temp T=1.6","Lm050":"B lam -0.5",
     "Lm100":"B lam -1.0","Lm150":"B lam -1.5","X100":"D CONTROL shuf -1.0"}
cells={}
for f in glob.glob("gen/*.json"):
    m=re.match(r"(.+)__(\d+)\.json",os.path.basename(f))
    cells.setdefault(m.group(1),{})[int(m.group(2))]=f
common=sorted(set.intersection(*[set(cells[c]) for c in CORE]))
print("n =",len(common),"prompts:",common)
def jac(ss):
    S=[set(s.split()) for s in ss]; v=[]
    for i,j in itertools.combinations(range(len(S)),2):
        u=len(S[i]|S[j]); v.append(len(S[i]&S[j])/u if u else 0)
    return float(np.mean(v)) if v else 0.
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
    return float(np.median(o)) if o else float('nan')
res={}
print("\n%-22s %8s %9s"%("condition","jaccard","GPT2 ppl"))
for t in CORE:
    js,ps=[],[]
    for i in common:
        g=json.load(open(cells[t][i]))
        s=g["gens"] if isinstance(g.get("gens"),list) else list(g["gens"].values())[0]
        js.append(jac(s)); ps.append(ppl(s))
    res[t]=dict(jac_per=js,ppl_per=ps,jac=float(np.mean(js)),ppl=float(np.mean(ps)))
    print("%-22s %8.4f %9.1f"%(LBL[t],res[t]['jac'],res[t]['ppl']))
# closest matched pair: T13 vs Lm100
A,B=res["T13"],res["Lm100"]
pA,pB=np.array(A["ppl_per"]),np.array(B["ppl_per"])
jA,jB=np.array(A["jac_per"]),np.array(B["jac_per"])
print("\nPAIRED: temp T=1.3 vs lambda=-1.0   (diversity matched to %.4f)"%abs(jA.mean()-jB.mean()))
print("%-10s %10s %10s %8s"%("prompt","ppl temp","ppl lam","log ratio"))
for k,i in enumerate(common):
    print("%-10d %10.1f %10.1f %8.2f"%(i,pA[k],pB[k],math.log(pA[k]/pB[k])))
lr=np.log(pA/pB)
rng=np.random.default_rng(0); n=len(lr)
boot=[np.mean(lr[rng.integers(0,n,n)]) for _ in range(20000)]
lo,hi=np.percentile(boot,[2.5,97.5])
print("\nmean log ratio %.3f -> %.2fx   95%% CI [%.2f, %.2f]"%(lr.mean(),math.exp(lr.mean()),math.exp(lo),math.exp(hi)))
print("lambda wins on %d/%d prompts"%((pB<pA).sum(),n))
try:
    from scipy.stats import wilcoxon
    print("Wilcoxon p = %.4f  (floor at n=%d is %.4f)"%(wilcoxon(pA,pB).pvalue,n,2**-(n-1)))
except Exception as e: print("scipy:",e)
# control check
C=res["X100"]
print("\nCONTROL: shuffled delta jac=%.4f ppl=%.1f  vs  temp T=1.3 jac=%.4f ppl=%.1f"%(
      C['jac'],C['ppl'],A['jac'],A['ppl']))
json.dump({k:{kk:vv for kk,vv in v.items()} for k,v in res.items()},
          open("results_core_n%d.json"%len(common),"w"),indent=1)
print("\nsaved -> results_core_n%d.json"%len(common))
