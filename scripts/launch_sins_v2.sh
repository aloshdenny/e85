cp /mnt/c/Users/research/w4090/sins.zip /home/research/e85/target/sins.zip
cp /mnt/c/Users/research/w4090/mia_suppress_readout.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/mia_suppress_readout_v2.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_sins_suppress_v2.sh /home/research/e85/scripts/
chmod +x /home/research/e85/scripts/run_sins_suppress_v2.sh
mkdir -p /home/research/e85/target_preds /home/research/e85_scratch
LOG=/home/research/e85_scratch/sins_suppress_v2.log
: > "$LOG"
setsid bash /home/research/e85/scripts/run_sins_suppress_v2.sh >> "$LOG" 2>&1 < /dev/null &
echo STARTED_PID=$!
sleep 5
tail -15 "$LOG"
