echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== e85 root ==="
ls -la /home/research/e85/ | head -30
echo "=== fairface ==="
ls -lh /home/research/e85/"fairface + ffhq"/ 2>/dev/null | head
ls /home/research/e85/"fairface + ffhq preds" 2>/dev/null | wc -l
echo "=== git ==="
cd /home/research/e85 && git status -sb && git log -1 --oneline
echo "=== existing residuals ==="
ls -lh /home/research/e85/abliterated/*.npz 2>/dev/null
echo "=== mia zip ==="
ls -lh /home/research/e85/target/mia.zip
echo "=== idle python ==="
ps aux | grep 'python scripts' | grep -v grep || echo none
echo "=== disk ==="
df -h /home | tail -1
