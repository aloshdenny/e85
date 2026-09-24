cp /mnt/c/Users/research/w4090/mia.zip /home/research/e85/target/
cp /mnt/c/Users/research/w4090/mia_suppress_readout_v2.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_mia_suppress_v2_inwild.sh /mnt/c/Users/research/w4090/run_mia_suppress_v2_face_top.sh /mnt/c/Users/research/w4090/run_mia_split_compare.sh /home/research/e85/scripts/
chmod +x /home/research/e85/scripts/run_mia_suppress_v2_inwild.sh /home/research/e85/scripts/run_mia_suppress_v2_face_top.sh /home/research/e85/scripts/run_mia_split_compare.sh
LOG=/home/research/e85_scratch/mia_split_compare.log
: > "$LOG"
setsid bash /home/research/e85/scripts/run_mia_split_compare.sh >> "$LOG" 2>&1 < /dev/null &
echo LAUNCHED_PID=$!
sleep 4
tail -6 "$LOG"
