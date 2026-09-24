echo "=== procs ==="
ps aux | grep -E 'part_occlusion|run_occlusion' | grep -v grep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== log ==="
tail -20 /home/research/e85_scratch/occlusion_suppressed.log
