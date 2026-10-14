for m in allenai/Llama-3.1-Tulu-3-8B-SFT unsloth/Meta-Llama-3.1-8B allenai/Llama-3.1-Tulu-3-8B-DPO allenai/Llama-3.1-Tulu-3-8B; do
  echo "== $m $(date)"; .venv/bin/hf download $m --include "*.json" --include "*.safetensors" --include "tokenizer*" > /dev/null && echo "OK $m $(date)" || echo "FAILED $m"
done
echo ALLDONE
