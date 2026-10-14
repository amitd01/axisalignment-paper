"""Score only the prompt indices present in EVERY condition, so the
cross-condition comparison is not confounded by prompt coverage."""
import os,json,glob,math,itertools,re
os.environ.setdefault("HF_HOME","/home/claude/hf")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(2)
jt=AutoTokenizer.from_pretrained("openai-community/gpt2")
jm=AutoModelForCausalLM.from_pretrained("openai-community/gpt2",dtype=torch.float32).eval()
cells={}
for f in glob.glob("gen/*.json"):
    m=re.match(r"(.+)__(\d+)\.json", os.path.basename(f))
    cells.setdefault(m.group(1),{})[int(m.group(2))]=f
common=set.intersection(*[set(v) for v in cells.values()])
print("conditions:",len(cells)," common prompt indices:",sorted(common))
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
ORDER=["T10","T13","T16","L100","L000","Lm050","Lm100","Lm150","G02","G05","X100"]
LBL={"T10":"A temp T=1.0","T13":"A temp T=1.3","T16":"A temp T=1.6",
 "L100":"B lam +1.0 (aligned)","L000":"B lam  0.0 (base)","Lm050":"B lam -0.5",
 "Lm100":"B lam -1.0","Lm150":"B lam -1.5","G02":"C dec g=0.2","G05":"C dec g=0.5",
 "X100":"D CONTROL shuffled -1.0"}
rows=[]
print("\n%-26s %8s %9s   (n=%d common prompts)"%("condition","jaccard","GPT2 ppl",len(common)))
for t in ORDER:
    if t not in cells: continue
    js,ps=[],[]
    for i in sorted(common):
        g=json.load(open(cells[t][i]))
        s=g["gens"] if isinstance(g.get("gens"),list) else list(g["gens"].values())[0]
        js.append(jac(s)); ps.append(ppl(s))
    r=dict(tag=t,label=LBL[t],jac=float(np.mean(js)),ppl=float(np.mean(ps)),
           jac_per=js,ppl_per=ps)
    rows.append(r)
    print("%-26s %8.4f %9.1f"%(LBL[t],r['jac'],r['ppl']))
json.dump(rows,open("results_matched_1b.json","w"),indent=1)
# interpolated frontier
def interp(op,target):
    pts=sorted([(r['jac'],r['ppl']) for r in rows if r['tag'].startswith(op)])
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    if not xs or target<min(xs) or target>max(xs): return None
    return float(np.interp(target,xs,ys))
print("\nMATCHED-DIVERSITY FRONTIER (interpolated)")
print("%9s %10s %10s %10s %10s"%("tgt jac","A temp","B lambda","C dec","D ctrl"))
for tj in [0.07,0.06,0.05,0.04,0.03,0.02]:
    v=[interp("T",tj),interp("L",tj),interp("G",tj),interp("X",tj)]
    f=lambda x: f"{x:.1f}" if x else "  n/a"
    print("%9.3f %10s %10s %10s %10s"%(tj,f(v[0]),f(v[1]),f(v[2]),f(v[3])))
