"""Exp 2b: psi restricted to the region where sampling actually happens."""
import os, json, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(1)
B="HuggingFaceTB/SmolLM2-135M"; A="HuggingFaceTB/SmolLM2-135M-Instruct"
tok=AutoTokenizer.from_pretrained(B)
mb=AutoModelForCausalLM.from_pretrained(B,dtype=torch.float32).eval()
ma=AutoModelForCausalLM.from_pretrained(A,dtype=torch.float32).eval()
TEXTS=["List some unusual uses for a brick. One unusual use is to",
"List some unusual uses for a paperclip. You could use it to",
"Unusual uses for an old newspaper include using it to",
"The most surprising thing about deep sea creatures is that they",
"Write the opening line of a short story about a lighthouse keeper who",
"A good way to explain entropy to a child is to say that",
"The main argument against the proposal is that it",
"In my opinion, the best approach to this problem would be to",
"Describe a colour to someone who has never seen. You might say",
"The reason old bridges were built with arches is because",
"An unconventional solution to urban traffic congestion would be to",
"What people usually get wrong about learning a language is that they"]
@torch.no_grad()
def lp(m,ids): return torch.log_softmax(m(ids).logits[0].float(),-1)

def wcorr(x,y,w):
    w=w/w.sum(); mx=(w*x).sum(); my=(w*y).sum()
    cov=(w*(x-mx)*(y-my)).sum()
    return float(cov/np.sqrt((w*(x-mx)**2).sum()*(w*(y-my)**2).sum()))

print("="*80); print("EXP 2b -- psi by region of the distribution"); print("="*80)
res={}
for K in [10,25,50,100,250,1000,None]:
    rs=[]
    for t in TEXTS:
        ids=tok(t,return_tensors="pt")["input_ids"]
        b=lp(mb,ids)[-1].numpy(); a=lp(ma,ids)[-1].numpy(); d=a-b
        if K is None: idx=np.where(b>np.log(1e-6))[0]
        else: idx=np.argsort(-b)[:K]
        rs.append(np.corrcoef(b[idx],d[idx])[0,1])
    rs=np.array(rs); lbl=f"top-{K}" if K else "plausible set"
    res[lbl]=dict(mean_r=float(rs.mean()),psi=float(abs(rs.mean())),sd=float(rs.std()))
    print(f"  {lbl:>14}:  mean r = {rs.mean():+.4f}  (sd {rs.std():.4f})   psi_hat = {abs(rs.mean()):.4f}")

print("\n  probability-weighted (weights = p_base, full vocab):")
ws=[]
for t in TEXTS:
    ids=tok(t,return_tensors="pt")["input_ids"]
    b=lp(mb,ids)[-1].numpy(); d=(lp(ma,ids)[-1]-lp(mb,ids)[-1]).numpy()
    ws.append(wcorr(b,d,np.exp(b)))
ws=np.array(ws)
print(f"  {'weighted':>14}:  mean r = {ws.mean():+.4f}  (sd {ws.std():.4f})   psi_hat = {abs(ws.mean()):.4f}")
res["weighted"]=dict(mean_r=float(ws.mean()),psi=float(abs(ws.mean())))

print("\n"+"="*80); print("What does the delta actually do to the top of the distribution?"); print("="*80)
for t in TEXTS[:4]:
    ids=tok(t,return_tensors="pt")["input_ids"]
    b=lp(mb,ids)[-1]; a=lp(ma,ids)[-1]; d=a-b
    tb=torch.topk(b,5).indices; ta=torch.topk(a,5).indices
    ov=len(set(tb.tolist())&set(ta.tolist()))
    print(f"\n  '{t[-42:]}'")
    print(f"    base top5   : {[tok.decode([i]).strip() for i in tb]}")
    print(f"    aligned top5: {[tok.decode([i]).strip() for i in ta]}")
    print(f"    top-5 overlap: {ov}/5   delta on base-top5: {[round(float(d[i]),2) for i in tb]}")
    print(f"    entropy base={float(-(b.exp()*b).sum()):.3f}  aligned={float(-(a.exp()*a).sum()):.3f}")
json.dump(res,open("results_exp2b.json","w"),indent=1)
print("\nsaved -> results_exp2b.json")
