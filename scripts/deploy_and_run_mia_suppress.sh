cp /mnt/c/Users/research/w4090/run_mia_suppress.sh /home/research/e85/scripts/
chmod +x /home/research/e85/scripts/run_mia_suppress.sh
mkdir -p /home/research/e85_scratch
LOG=/home/research/e85_scratch/mia_suppress.log
: > "$LOG"
setsid bash /home/research/e85/scripts/run_mia_suppress.sh >> "$LOG" 2>&1 < /dev/null &
echo STARTED_PID=$!
sleep 2
tail -5 "$LOG"
