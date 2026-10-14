"""Dump final-position next-token log-probs for one model over the prompt set.

Loads ONE model, writes a float32 array [n_prompts, vocab], frees it. This keeps
peak memory at a single model and lets us compare any number of checkpoints
pairwise offline. Required because psi is a property of a *pair*, and we want
several pairs (base->SFT, SFT->DPO, DPO->Instruct) from four checkpoints.

Usage: python3 dump_logprobs.py <model_id> <out_tag>
"""
import os, sys, json
os.environ.setdefault("HF_HOME", "/home/claude/hf")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import FLAT

torch.set_num_threads(2)
model_id, tag = sys.argv[1], sys.argv[2]
OUT = os.path.dirname(os.path.abspath(__file__)) + "/dumps"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(model_id)
# bfloat16 halves resident memory; we cast logits back to fp32 before softmax so
# the log-probs themselves carry full precision.
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()

rows = []
ids_used = []
with torch.no_grad():
    for fam, p in FLAT:
        ids = tok(p, return_tensors="pt")["input_ids"]
        lg = model(ids).logits[0, -1].float()
        rows.append(torch.log_softmax(lg, -1).numpy().astype(np.float32))
        ids_used.append(ids.shape[1])

arr = np.stack(rows)
np.save(f"{OUT}/{tag}.npy", arr)
meta = dict(model_id=model_id, tag=tag, shape=list(arr.shape),
            vocab=int(arr.shape[1]), n_prompts=int(arr.shape[0]),
            prompt_token_lens=ids_used,
            tokenizer_vocab_size=int(tok.vocab_size))
json.dump(meta, open(f"{OUT}/{tag}.json", "w"), indent=1)
print(f"saved {tag}: {arr.shape} vocab={arr.shape[1]}")
