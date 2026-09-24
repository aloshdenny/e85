echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== idle ==="
ps aux | grep 'python scripts' | grep -v grep || echo none
echo "=== sins zip ==="
ls -lh /home/research/e85/target/sins.zip
echo "=== fairface cache ==="
ls -lh /home/research/e85_scratch/v2_bottlenecks.npz
