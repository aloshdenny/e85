#!/usr/bin/env bash
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo "=== procs ==="
pgrep -af "mia_suppress_readout_v2" || echo none
echo "=== mia log tail ==="
tail -35 /home/research/e85_scratch/mia_suppress_v2_rerun.log 2>/dev/null || echo missing
echo "=== sins log tail ==="
tail -35 /home/research/e85_scratch/sins_suppress_v2_rerun.log 2>/dev/null || echo missing
echo "=== both log tail ==="
tail -8 /home/research/e85_scratch/both_suppress_v2_rerun.log 2>/dev/null || echo missing
