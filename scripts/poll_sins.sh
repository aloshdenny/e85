echo "=== procs ==="
ps aux | grep -E 'sins_suppress|mia_suppress_readout_v2' | grep -v grep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== log ==="
tail -30 /home/research/e85_scratch/sins_suppress_v2.log 2>/dev/null || echo "(no log)"
