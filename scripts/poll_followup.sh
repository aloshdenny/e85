echo "=== log tail ==="
tail -20 /home/research/e85_scratch/followup_all3.log 2>&1 || echo "(no log)"
echo "=== procs ==="
ps aux | grep -E 'apply_suppress|part_occlusion|viz_suppress|run_followup' | grep -v grep || echo none
