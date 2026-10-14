"""Scaled generation sweep at 1B.

Differences from the 135M pilot:
  - OLMo-2-0425-1B base <-> Instruct (7.4x the parameters)
  - 12 prompts, 2 from each of 6 task families (pilot: 5 prompts, 3 of them
    alternate-uses variants). With n=12 the paired Wilcoxon can actually reach
    significance; at n=5 its floor was p=0.0625.
  - Both models held resident for the whole sweep; one process, per-cell
    checkpointing so the run is resumable.

Operators (identical forms to the pilot):
  temp  z = z_a / T
  lam   z = z_b + lam (z_a - z_b)
  dec   z = z_a - gamma * log_softmax(z_b)
  ctrl  z = z_a + (lam-1) * shuffle(z_a - z_b)      direction control
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/home/claude/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS

torch.set_num_threads(2)
SEED, N, MAXTOK, TOPP = 20261012, 6, 32, 0.95
BASE = "allenai/OLMo-2-0425-1B"
ALGN = "allenai/OLMo-2-0425-1B-Instruct"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/gen"
os.makedirs(OUT, exist_ok=True)

# two prompts per family -> 12 total, balanced across registers
SEL = [(fam, p) for fam, ps in PROMPTS.items() for p in ps[:2]]

CONDS = [
    ("temp", 1.0,  "T10"), ("temp", 1.3,  "T13"), ("temp", 1.6,  "T16"),
    ("lam",  1.0,  "L100"), ("lam", 0.0,  "L000"), ("lam", -0.5, "Lm050"),
    ("lam", -1.0, "Lm100"), ("lam", -1.5, "Lm150"),
    ("dec",  0.2,  "G02"), ("dec", 0.5,  "G05"),
    ("ctrl", -1.0, "X100"),
]

tok = AutoTokenizer.from_pretrained(BASE)
ma = AutoModelForCausalLM.from_pretrained(ALGN, dtype=torch.bfloat16).eval()
mb = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).eval()
print("models loaded", flush=True)


@torch.no_grad()
def gen(op, param, prompt, seed):
    g = torch.Generator().manual_seed(seed)
    gp = torch.Generator().manual_seed(seed + 7)
    plen = len(tok(prompt)["input_ids"])
    ids = tok(prompt, return_tensors="pt")["input_ids"].repeat(N, 1)
    ca = cb = None
    cur = ids
    need_base = op != "temp"
    for _ in range(MAXTOK):
        oa = ma(cur, past_key_values=ca, use_cache=True)
        ca = oa.past_key_values
        za = oa.logits[:, -1, :].float()
        if need_base:
            ob = mb(cur, past_key_values=cb, use_cache=True)
            cb = ob.past_key_values
            zb = ob.logits[:, -1, :].float()
        if op == "temp":
            z = za / param
        elif op == "lam":
            z = zb + param * (za - zb)
        elif op == "dec":
            z = za - param * torch.log_softmax(zb, -1)
        elif op == "ctrl":
            d = za - zb
            perm = torch.stack([torch.randperm(d.shape[1], generator=gp)
                                for _ in range(d.shape[0])])
            z = za + (param - 1.0) * torch.gather(d, 1, perm)
        p = torch.softmax(z, -1)
        sp, si = torch.sort(p, descending=True, dim=-1)
        cum = sp.cumsum(-1)
        sp[cum - sp > TOPP] = 0.0
        sp = sp / sp.sum(-1, keepdim=True)
        nxt = si.gather(-1, torch.multinomial(sp, 1, generator=g))
        ids = torch.cat([ids, nxt], 1)
        cur = nxt
    return [tok.decode(r[plen:], skip_special_tokens=True) for r in ids]


t0 = time.time()
total = len(CONDS) * len(SEL)
done = 0
# Prompt-major: every prompt completes ALL conditions before the next prompt
# starts. Any stopping point therefore yields a balanced design (k prompts x all
# conditions) rather than a truncated one.
for pi, (fam, pr) in enumerate(SEL):
    for op, param, tag in CONDS:
        done += 1
        f = f"{OUT}/{tag}__{pi}.json"
        if os.path.exists(f):
            continue
        t = time.time()
        s = gen(op, param, pr, SEED + 1000 * pi)
        json.dump(dict(op=op, param=param, tag=tag, family=fam, prompt=pr,
                       pi=pi, gens=s), open(f, "w"), indent=1)
        print(f"[{done}/{total}] {tag} p{pi} ({fam}) {time.time()-t:.0f}s "
              f"elapsed {(time.time()-t0)/60:.1f}m", flush=True)
print("ALLDONE", flush=True)
