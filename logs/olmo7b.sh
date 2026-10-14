set -u
while ! grep -q ALLDONE logs/download.log; do sleep 20; done
D() { HF_HUB_OFFLINE=1 .venv/bin/python code/dump_logprobs_mps.py "$@"; }
for s in base: sft:-SFT dpo:-DPO inst:-Instruct; do
  m=allenai/OLMo-2-1124-7B${s#*:}
  echo "== dl $m $(date)"
  .venv/bin/hf download $m --include "*.json" --include "*.safetensors" --include "*.bin" --include "*.txt" --include "tokenizer*" > /dev/null || { echo "FAILED dl $m"; continue; }
  echo "OK dl $m $(date)"
  D $m olmo7b_${s%%:*} || echo "FAILED dump $m"
done
echo OLMO7B_DONE
