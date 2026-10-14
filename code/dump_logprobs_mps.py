"""MPS port of dump_logprobs.py, for running natively on Apple Silicon.

Identical measurement: final-position next-token log-probs over prompts.FLAT,
raw prompt text (no chat template), bf16 weights, fp32 softmax. Loads ONE model,
writes [n_prompts, vocab] float32, frees it.

Also saves the exact input token IDs per prompt, so tokeniser identity can be
checked at the token-ID level across a pair before any logit arithmetic.

Usage: python dump_logprobs_mps.py <model_id> <out_tag>
"""
import os, sys, json, gc
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import FLAT

model_id, tag = sys.argv[1], sys.argv[2]
OUT = os.path.dirname(os.path.abspath(__file__)) + "/dumps"
os.makedirs(OUT, exist_ok=True)
dev = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(model_id)
# Every checkpoint is loaded as bf16 regardless of its stored precision (Tulu-3-8B-DPO
# ships fp32). Otherwise part of a measured delta would be precision, not training.
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval().to(dev)
param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
assert param_dtypes == ["torch.bfloat16"], f"{model_id} not uniformly bf16: {param_dtypes}"
print(f"{model_id}: params {param_dtypes} on {dev}")

rows, ids_used = [], []
with torch.no_grad():
    for fam, p in FLAT:
        ids = tok(p, return_tensors="pt")["input_ids"]
        lg = model(ids.to(dev)).logits[0, -1].float().cpu()
        rows.append(torch.log_softmax(lg, -1).numpy().astype(np.float32))
        ids_used.append(ids[0].tolist())

arr = np.stack(rows)
np.save(f"{OUT}/{tag}.npy", arr)
meta = dict(model_id=model_id, tag=tag, device=dev, param_dtype=param_dtypes[0], shape=list(arr.shape),
            vocab=int(arr.shape[1]), n_prompts=int(arr.shape[0]),
            prompt_token_lens=[len(x) for x in ids_used],
            input_ids=ids_used,
            tokenizer_vocab_size=int(tok.vocab_size), tokenizer_len=len(tok),
            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
json.dump(meta, open(f"{OUT}/{tag}.json", "w"), indent=1)
print(f"saved {tag}: {arr.shape} vocab={arr.shape[1]} on {dev}")
del model; gc.collect()
if dev == "mps":
    torch.mps.empty_cache()
