set -u
D() { .venv/bin/python code/dump_logprobs_mps.py "$@"; }
export HF_HUB_OFFLINE=1
waitfor() { while ! grep -q "^OK $1" logs/download.log; do sleep 20; done; }
D allenai/Llama-3.1-Tulu-3-8B-SFT tulu_sft
D unsloth/Meta-Llama-3.1-8B tulu_mirrorbase
: # meta download already complete
D meta-llama/Llama-3.1-8B tulu_base
waitfor allenai/Llama-3.1-Tulu-3-8B-DPO
D allenai/Llama-3.1-Tulu-3-8B-DPO tulu_dpo
waitfor "allenai/Llama-3.1-Tulu-3-8B "
D allenai/Llama-3.1-Tulu-3-8B tulu_inst
echo TULU_MPS_DONE
for s in base:meta-llama/Llama-3.1-8B sft:allenai/Llama-3.1-Tulu-3-8B-SFT dpo:allenai/Llama-3.1-Tulu-3-8B-DPO inst:allenai/Llama-3.1-Tulu-3-8B; do
  DEVICE=cpu D ${s#*:} tulucpu_${s%%:*}
done
echo TULU_CPU_DONE
