cp /mnt/c/Users/research/w4090/lookalikes.zip /home/research/e85/target/lookalikes.zip
cp /mnt/c/Users/research/w4090/apply_suppress_eval.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/part_occlusion_map.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/viz_suppress.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_followup_all3.sh /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/mia_suppress_readout.py /home/research/e85/scripts/
chmod +x /home/research/e85/scripts/run_followup_all3.sh
mkdir -p /home/research/e85_scratch
LOG=/home/research/e85_scratch/followup_all3.log
: > "$LOG"
setsid bash /home/research/e85/scripts/run_followup_all3.sh >> "$LOG" 2>&1 < /dev/null &
echo STARTED_PID=$!
sleep 3
tail -8 "$LOG"
