#!/usr/bin/env bash
set -euo pipefail
cd /home/research/e85
SCRATCH=/home/research/e85_scratch
mkdir -p "$SCRATCH"
echo "=== MIA split compare: in_the_wild ==="
bash scripts/run_mia_suppress_v2_inwild.sh 2>&1 | tee "$SCRATCH/mia_suppress_v2_inwild.log"
echo "=== MIA split compare: face_top ==="
bash scripts/run_mia_suppress_v2_face_top.sh 2>&1 | tee "$SCRATCH/mia_suppress_v2_face_top.log"
python3 - <<'PY'
import re
from pathlib import Path

def parse_log(path):
    t = Path(path).read_text(errors="replace")
    m = re.search(r"holdout FACE drop ([+-]?\d+\.\d+)", t)
    r = re.search(r"random FairFace drop\s+([+-]?\d+\.\d+)", t)
    v = re.search(r"(OK: identity-selective|FAIL: [^\n]+)", t)
    ho = re.search(r"HOLDOUT  n=(\d+)", t)
    return {
        "holdout_drop": float(m.group(1)) if m else None,
        "rand_drop": float(r.group(1)) if r else None,
        "verdict": v.group(1) if v else "?",
        "holdout_n": int(ho.group(1)) if ho else None,
    }

rows = {}
for name, log in [("in_the_wild", "/home/research/e85_scratch/mia_suppress_v2_inwild.log"),
                  ("face_top", "/home/research/e85_scratch/mia_suppress_v2_face_top.log")]:
    rows[name] = parse_log(log)

def score(d):
    if d["holdout_drop"] is None:
        return -1
    ok = 1 if d["verdict"].startswith("OK") else 0
    sel = abs(d["rand_drop"] or 0)
    return (ok * 1000 + d["holdout_drop"] * 100 - sel * 400)

print("\n" + "=" * 60)
print("MIA SPLIT COMPARISON (600-image cleaned gallery)")
print(f"{'mode':<14} {'ho_n':>5} {'ho_drop':>9} {'rand_d':>9}  verdict")
print("-" * 60)
for name, d in rows.items():
    print(f"{name:<14} {d['holdout_n'] or '?':>5} {d['holdout_drop']:>+9.5f} "
          f"{d['rand_drop']:>+9.5f}  {d['verdict']}")
winner = max(rows, key=lambda k: score(rows[k]))
print("-" * 60)
print(f"WINNER: {winner}  (score={score(rows[winner]):.1f})")
print("=" * 60)
PY
echo "DONE mia_split_compare"
