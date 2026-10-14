set -u
lock()   { until mkdir logs/gpu.lock 2>/dev/null; do sleep 15; done; }
unlock() { rmdir logs/gpu.lock; }
for s in base::safetensors sft:-SFT:bin dpo:-DPO:safetensors inst:-Instruct:safetensors; do
  tag=${s%%:*}; rest=${s#*:}; suf=${rest%%:*}; fmt=${rest#*:}
  m=allenai/OLMo-2-1124-13B$suf
  echo "== dl $m ($fmt) $(date)"
  .venv/bin/hf download $m --include "*.json" --include "*.$fmt" --include "*.txt" --include "tokenizer*" > /dev/null \
    || { echo "FAILED dl $m"; continue; }
  echo "OK dl $m $(date)"
  lock; HF_HUB_OFFLINE=1 .venv/bin/python code/dump_logprobs_mps.py $m olmo13b_$tag || echo "FAILED dump $m"; unlock
done
echo OLMO13B_DONE
