python3 - <<'PY'
from pathlib import Path
t = Path("/home/research/e85_scratch/sins_suppress_v2.log").read_text(errors="replace")
for line in t.splitlines():
    s = line.strip()
    if any(k in s for k in ("Sins:", "reusing", "Capturing", "bottleneck", "TRIBE",
                             "splits:", "FACE means", "Training", "ep ",
                             "SINS", "HOLDOUT", "VERDICT", "Saved", "DONE",
                             "Error", "Traceback", "FAIL", "OK:")):
        if "Encoding" in s or "Loading weights" in s:
            continue
        print(s)
print("---")
print("alive" if "DONE sins" not in t else "finished")
PY
ps aux | grep mia_suppress_readout_v2 | grep -v grep || echo none
