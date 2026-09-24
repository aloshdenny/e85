cp /mnt/c/Users/research/w4090/mia.zip /mnt/c/Users/research/w4090/sins.zip /home/research/e85/target/
cp /mnt/c/Users/research/w4090/mia_suppress_readout_v2.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_mia_suppress_v2.sh /mnt/c/Users/research/w4090/run_sins_suppress_v2.sh /mnt/c/Users/research/w4090/run_both_suppress_v2.sh /home/research/e85/scripts/
chmod +x /home/research/e85/scripts/run_both_suppress_v2.sh
LOG=/home/research/e85_scratch/both_suppress_v2_rerun.log
: > "$LOG"
setsid bash /home/research/e85/scripts/run_both_suppress_v2.sh >> "$LOG" 2>&1 < /dev/null &
echo LAUNCHED_PID=$!
sleep 3
tail -8 "$LOG"
