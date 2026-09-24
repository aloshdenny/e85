echo "=== processes ==="
ps aux | grep -E 'mia_suppress|run_mia' | grep -v grep || true
echo "=== scratch ==="
ls -la /home/research/e85_scratch/ 2>&1 || true
echo "=== script exists ==="
ls -la /home/research/e85/scripts/mia_suppress_readout.py /home/research/e85/scripts/run_mia_suppress.sh 2>&1
echo "=== try manual head ==="
head -5 /home/research/e85/scripts/run_mia_suppress.sh 2>&1
