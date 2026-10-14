while pgrep -f "dump_logprobs_mps.py allenai/OLMo-2-0425-1B-SFT" >/dev/null || [ ! -f code/dumps/olmo_sft.npy ]; do sleep 10; done
.venv/bin/python code/dump_logprobs_mps.py allenai/OLMo-2-0425-1B olmo_base
.venv/bin/python code/dump_logprobs_mps.py allenai/OLMo-2-0425-1B-SFT olmo_sft
.venv/bin/python code/dump_logprobs_mps.py allenai/OLMo-2-0425-1B-DPO olmo_dpo
.venv/bin/python code/dump_logprobs_mps.py allenai/OLMo-2-0425-1B-Instruct olmo_inst
echo OLMO_ALL_DONE
