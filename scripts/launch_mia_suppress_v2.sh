cp /mnt/c/Users/research/w4090/mia_suppress_readout_v2.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_mia_suppress_v2.sh /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/mia_suppress_readout.py /home/research/e85/scripts/ 2>/dev/null || true
chmod +x /home/research/e85/scripts/run_mia_suppress_v2.sh
bash /mnt/c/Users/research/w4090/place_fairface.sh
LOG=/home/research/e85_scratch/mia_suppress_v2.log
mkdir -p /home/research/e85_scratch
: > "$LOG"
setsid bash /home/research/e85/scripts/run_mia_suppress_v2.sh >> "$LOG" 2>&1 < /dev/null &
echo STARTED_PID=$!
sleep 4
tail -12 "$LOG"
