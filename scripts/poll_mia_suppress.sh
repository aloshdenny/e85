echo "=== log tail ==="
tail -15 /home/research/e85_scratch/mia_suppress.log 2>&1 || echo "(no log yet)"
echo "=== procs ==="
ps aux | grep mia_suppress | grep -v grep || echo "(no process)"
