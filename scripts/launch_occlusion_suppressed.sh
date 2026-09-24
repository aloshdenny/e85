cp /mnt/c/Users/research/w4090/occlusion_saliency.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/part_occlusion_map.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/part_occlusion_report.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_occlusion_suppressed.sh /home/research/e85/scripts/
chmod +x /home/research/e85/scripts/run_occlusion_suppressed.sh
mkdir -p /home/research/e85_scratch
LOG=/home/research/e85_scratch/occlusion_suppressed.log
: > "$LOG"
setsid bash /home/research/e85/scripts/run_occlusion_suppressed.sh >> "$LOG" 2>&1 < /dev/null &
echo STARTED_PID=$!
sleep 4
tail -15 "$LOG"
