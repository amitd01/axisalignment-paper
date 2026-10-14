"""Token-ID identity across checkpoints of one family, over the shared ID range.

Usage: python check_tokenizers.py <model_id> <model_id> ...   (first is the reference)
Exits non-zero if any token string maps to a different ID below the common width.
"""
import sys
from transformers import AutoTokenizer, AutoConfig
from prompts import FLAT

ids = sys.argv[1:]
toks = {m: AutoTokenizer.from_pretrained(m) for m in ids}
heads = {m: AutoConfig.from_pretrained(m).vocab_size for m in ids}
V = min(heads.values())
inv = lambda t: {i: s for s, i in t.get_vocab().items() if i < V}
ref = inv(toks[ids[0]])
ok = True
for m in ids:
    cur = inv(toks[m])
    diff = [i for i in range(V) if ref.get(i) != cur.get(i)]
    enc = all(toks[m](p)["input_ids"] == toks[ids[0]](p)["input_ids"] for _, p in FLAT)
    print(f"{m:<45} head={heads[m]:<7} len(tok)={len(toks[m]):<7} "
          f"id-map diffs<{V}: {len(diff):<4} prompt encodings equal: {enc}")
    if diff:
        print("   first diffs:", [(i, ref.get(i), cur.get(i)) for i in diff[:5]])
    ok &= not diff and enc
print("TOKENISER IDENTITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
