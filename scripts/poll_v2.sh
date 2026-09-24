echo "=== procs ==="
ps aux | grep -E 'mia_suppress_readout_v2|run_mia_suppress_v2' | grep -v grep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== log ==="
tail -25 /home/research/e85_scratch/mia_suppress_v2.log 2>/dev/null || echo "(no log)"
