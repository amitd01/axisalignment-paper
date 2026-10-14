"""Step 0 of PREREG-135M-GAMMA: beta for SmolLM2-135M -> -Instruct (prob-weighted fit of shift on base log-prob,
plausible set), at prompt-final positions and at base-written continuation positions (8 x 32 per prompt)."""
import json, math, numpy as np, torch, importlib.util
from transformers import AutoTokenizer, AutoModelForCausalLM
spec = importlib.util.spec_from_file_location("g", __file__.replace("beta135m.py", "gen_one.py"))
src = open(spec.origin).read(); PROMPTS = eval(src.split("PROMPTS = ")[1].split("]")[0] + "]")
B, A = "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(B)
mb = AutoModelForCausalLM.from_pretrained(B, dtype=torch.float32).eval()
ma = AutoModelForCausalLM.from_pretrained(A, dtype=torch.float32).eval()
def beta(b, d):
    S = b > math.log(1e-6); p = np.exp(b - b.max()); p /= p.sum(); w = p[S] / p[S].sum()
    X = np.stack([np.ones(S.sum()), b[S]], 1); sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], d[S] * sw, rcond=None)[0][1]
res = {"prompt_final": [], "continuation": []}
torch.manual_seed(20260803)
with torch.no_grad():
    for pr in PROMPTS:
        ids = tok(pr, return_tensors="pt")["input_ids"]
        g = mb.generate(ids, do_sample=True, top_p=0.95, top_k=0, temperature=1.0, max_new_tokens=32,
                        num_return_sequences=8, pad_token_id=tok.eos_token_id)
        for s in range(8):
            lb = torch.log_softmax(mb(g[s:s+1]).logits[0].float(), -1).numpy().astype(float)
            la = torch.log_softmax(ma(g[s:s+1]).logits[0].float(), -1).numpy().astype(float)
            for t in range(ids.shape[1] - 1, g.shape[1] - 1):
                key = "prompt_final" if t == ids.shape[1] - 1 else "continuation"
                if key == "prompt_final" and s > 0: continue
                res[key].append(beta(lb[t], la[t] - lb[t]))
for k, v in res.items():
    v = np.array(v); print(f"{k:<13} n={len(v):4d}  beta_w median {np.median(v):+.3f}  IQR [{np.percentile(v,25):+.3f}, {np.percentile(v,75):+.3f}]  frac>0 {np.mean(v>0):.2f}")
json.dump({k: [float(x) for x in v] for k, v in res.items()}, open(__file__.replace("code/pilot135m_port/beta135m.py", "code/dumps/135m/beta135m.json"), "w"))
