echo "=== log size ==="
wc -c /home/research/e85_scratch/mia_suppress.log 2>&1 || true
echo "=== log ==="
cat /home/research/e85_scratch/mia_suppress.log 2>&1 | tail -20
echo "=== procs ==="
ps aux | grep -E 'mia_suppress|run_mia' | grep -v grep || true
echo "=== relaunch ==="
pkill -f mia_suppress_readout.py 2>/dev/null || true
sleep 1
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
cd /home/research/e85
: > /home/research/e85_scratch/mia_suppress.log
setsid bash /home/research/e85/scripts/run_mia_suppress.sh >> /home/research/e85_scratch/mia_suppress.log 2>&1 < /dev/null &
sleep 5
tail -8 /home/research/e85_scratch/mia_suppress.log
ps aux | grep mia_suppress_readout | grep -v grep || true
