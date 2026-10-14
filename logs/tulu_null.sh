until mkdir logs/gpu.lock 2>/dev/null; do sleep 15; done
DEVICE=cpu OUT_STAGE=basecpu HF_HUB_OFFLINE=1 .venv/bin/python code/register_v2.py score tulu base || echo "FAILED tulu null"
rmdir logs/gpu.lock; echo TULU_NULL_DONE
