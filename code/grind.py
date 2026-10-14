"""Deadline-aware foreground grinder for the 1B sweep.

Why this exists
---------------
Background runs die in idle gaps between user turns (observed twice), and each
Bash call caps at 2 minutes. So cells must be produced inside foreground calls.
This script loads both models once, then produces cells until a wall-clock
deadline and exits cleanly — never killed mid-cell, so no work is wasted.

Priority order
--------------
Only the conditions needed to settle lambda-vs-temperature are run for the
remaining prompts:

    T13, T16      temperature frontier
    Lm050, Lm100, Lm150   lambda frontier
    X100          direction control

Dropped for new prompts (already characterised on prompts 0-3, and not part of
the matched-diversity comparison):
    T10 / L100    identical by construction; consistency check already passed
    L000          base-model reference point
    G02 / G05     decontamination; already answered at this scale

This yields an unbalanced overall design but a BALANCED 6-condition core across
all prompts, which is what the lambda comparison needs.

Usage:  python3 grind.py [deadline_seconds]
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "/home/claude/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS

torch.set_num_threads(2)
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
T0 = time.time()

SEED, N, MAXTOK, TOPP = 20261012, 6, 32, 0.95
BASE = "allenai/OLMo-2-0425-1B"
ALGN = "allenai/OLMo-2-0425-1B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + "/gen"

SEL = [(fam, p) for fam, ps in PROMPTS.items() for p in ps[:2]]

# the 6-condition core, in the order they should be filled
CORE = [("temp", 1.3, "T13"), ("temp", 1.6, "T16"),
        ("lam", -0.5, "Lm050"), ("lam", -1.0, "Lm100"),
        ("lam", -1.5, "Lm150"), ("ctrl", -1.0, "X100")]

# build the work queue: prompt-major, so stopping anywhere leaves a balanced set
todo = [(pi, fam, pr, op, param, tag)
        for pi, (fam, pr) in enumerate(SEL)
        for op, param, tag in CORE
        if not os.path.exists(f"{OUT}/{tag}__{pi}.json")]

if not todo:
    print("CORE COMPLETE — nothing to grind", flush=True)
    sys.exit(0)

print(f"queue: {len(todo)} cells remaining in the 6-condition core", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
ma = AutoModelForCausalLM.from_pretrained(ALGN, dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True).eval()
mb = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True).eval()
load_t = time.time() - T0
print(f"models loaded in {load_t:.0f}s", flush=True)


@torch.no_grad()
def gen(op, param, prompt, seed):
    g = torch.Generator().manual_seed(seed)
    gp = torch.Generator().manual_seed(seed + 7)
    plen = len(tok(prompt)["input_ids"])
    ids = tok(prompt, return_tensors="pt")["input_ids"].repeat(N, 1)
    ca = cb = None
    cur = ids
    for _ in range(MAXTOK):
        oa = ma(cur, past_key_values=ca, use_cache=True)
        ca = oa.past_key_values
        za = oa.logits[:, -1, :].float()
        if op != "temp":
            ob = mb(cur, past_key_values=cb, use_cache=True)
            cb = ob.past_key_values
            zb = ob.logits[:, -1, :].float()
        if op == "temp":
            z = za / param
        elif op == "lam":
            z = zb + param * (za - zb)
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


# estimate per-cell cost so we never start a cell we cannot finish
est = {"temp": 46.0, "lam": 90.0, "ctrl": 95.0}
done = 0
for pi, fam, pr, op, param, tag in todo:
    left = DEADLINE - (time.time() - T0)
    if left < est[op]:
        print(f"stopping: {left:.0f}s left, next cell needs ~{est[op]:.0f}s",
              flush=True)
        break
    t = time.time()
    s = gen(op, param, pr, SEED + 1000 * pi)
    json.dump(dict(op=op, param=param, tag=tag, family=fam, prompt=pr,
                   pi=pi, gens=s), open(f"{OUT}/{tag}__{pi}.json", "w"), indent=1)
    done += 1
    took = time.time() - t
    est[op] = took * 1.1          # adapt the estimate to this machine
    print(f"  + {tag} p{pi} ({fam}) {took:.0f}s", flush=True)

print(f"GRIND DONE: {done} cells this call, {time.time()-T0:.0f}s total", flush=True)
