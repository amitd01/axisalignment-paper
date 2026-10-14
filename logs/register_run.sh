set -u
lock()   { until mkdir logs/gpu.lock 2>/dev/null; do sleep 15; done; }
unlock() { rmdir logs/gpu.lock; }
R() { HF_HUB_OFFLINE=1 .venv/bin/python code/register_v2.py "$@"; }
for fam in "$@"; do
  lock; R gen $fam || echo "FAILED gen $fam"; unlock
  for s in base sft dpo inst; do lock; R score $fam $s || echo "FAILED score $fam $s"; unlock; done
  echo "SCORED $fam"
done
echo REGISTER_DONE
