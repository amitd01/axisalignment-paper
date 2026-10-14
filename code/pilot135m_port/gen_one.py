"""Generate one condition. Run as a separate process so memory is released.

Usage: python3 gen_one.py <op> <param> <tag>
  op in {temp, lam, dec}
"""
import os, sys, json
pass  # port: default HF cache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(1)
op, param, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]

SEED, N, MAXTOK, TOPP = 20260803, 8, 32, 0.95
B = "HuggingFaceTB/SmolLM2-135M"
A = "HuggingFaceTB/SmolLM2-135M-Instruct"

tok = AutoTokenizer.from_pretrained(B)
ma = AutoModelForCausalLM.from_pretrained(A, dtype=torch.float32).eval()
mb = (AutoModelForCausalLM.from_pretrained(B, dtype=torch.float32).eval()
      if op != "temp" else None)

PROMPTS = [
    "List some unusual uses for a brick. One unusual use is to",
    "List some unusual uses for a paperclip. You could use it to",
    "Unusual uses for an old newspaper include using it to",
    "Write the opening line of a short story about a lighthouse keeper who",
    "An unconventional solution to urban traffic congestion would be to",
]


@torch.no_grad()
def gen(prompt, seed):
    g = torch.Generator().manual_seed(seed)
    plen = len(tok(prompt)["input_ids"])
    ids = tok(prompt, return_tensors="pt")["input_ids"].repeat(N, 1)
    ca = cb = None
    cur = ids
    for _ in range(MAXTOK):
        oa = ma(cur, past_key_values=ca, use_cache=True)
        ca = oa.past_key_values
        za = oa.logits[:, -1, :].float()
        if mb is not None:
            ob = mb(cur, past_key_values=cb, use_cache=True)
            cb = ob.past_key_values
            zb = ob.logits[:, -1, :].float()
        if op == "temp":
            z = za / param
        elif op == "lam":
            z = zb + param * (za - zb)
        elif op == "dec":
            z = za - param * torch.log_softmax(zb, -1)
        else:
            raise ValueError(op)
        p = torch.softmax(z, -1)
        sp, si = torch.sort(p, descending=True, dim=-1)
        cum = sp.cumsum(-1)
        sp[cum - sp > TOPP] = 0.0
        sp = sp / sp.sum(-1, keepdim=True)
        nxt = si.gather(-1, torch.multinomial(sp, 1, generator=g))
        ids = torch.cat([ids, nxt], 1)
        cur = nxt
    return [tok.decode(r[plen:], skip_special_tokens=True) for r in ids]


out = {pr: gen(pr, SEED + 1000 * i) for i, pr in enumerate(PROMPTS)}
json.dump(dict(op=op, param=param, tag=tag, gens=out),
          open(f"/Users/amitdas/claude/repos/alignment-axis/code/dumps/135m/gen_{tag}.json", "w"), indent=1)
print("done", tag, flush=True)
