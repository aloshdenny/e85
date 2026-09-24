set -x
mkdir -p /home/research/e85_scratch
bash /home/research/e85/scripts/run_mia_suppress.sh 2>&1 | head -50
