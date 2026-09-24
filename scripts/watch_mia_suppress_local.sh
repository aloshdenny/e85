#!/usr/bin/env bash
# Poll Mia suppress readout job on 4090; scp results when DONE.
set -euo pipefail
HOST=research@100.86.165.70
LOG=/home/research/e85_scratch/mia_suppress.log
LOCAL_LOG=/Users/aoxo/vscode/e85/target_preds/mia_suppress.log
LOCAL_OUT=/Users/aoxo/vscode/e85/abliterated/mia_suppress_readout.npz

while true; do
  ssh "$HOST" "wsl bash -c 'tail -3 $LOG 2>/dev/null; pgrep -af mia_suppress_readout || true; grep -q DONE $LOG 2>/dev/null && echo WATCH_DONE'" \
    > /tmp/mia_sup_poll.txt 2>&1 || true
  if grep -q WATCH_DONE /tmp/mia_sup_poll.txt; then
    echo "Job finished — pulling artifacts"
    scp "$HOST:/home/research/e85/abliterated/mia_suppress_readout.npz" "$LOCAL_OUT" 2>/dev/null || true
    scp "$HOST:$LOG" "$LOCAL_LOG" 2>/dev/null || true
    exit 0
  fi
  if ! grep -q mia_suppress_readout /tmp/mia_sup_poll.txt; then
    if grep -q "Saved ->" /tmp/mia_sup_poll.txt 2>/dev/null; then
      scp "$HOST:/home/research/e85/abliterated/mia_suppress_readout.npz" "$LOCAL_OUT" || true
      scp "$HOST:$LOG" "$LOCAL_LOG" || true
      exit 0
    fi
  fi
  sleep 120
done
