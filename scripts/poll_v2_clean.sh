python3 - <<'PY'
from pathlib import Path
t = Path("/home/research/e85_scratch/mia_suppress_v2.log").read_text(errors="replace")
for line in t.splitlines():
    s = line.strip()
    if any(k in s for k in ("Mia:", "general portraits", "bottleneck", "TRIBE bottleneck",
                             "splits:", "FACE means", "Training rank", "ep ",
                             "STUDIO", "WILD", "NEIGHBOR", "RANDOM", "VERDICT",
                             "Saved ->", "DONE", "Error", "Traceback", "aligned")):
        if "Encoding" in s or "Loading weights" in s:
            continue
        print(s)
print("--- last bottleneck ---")
for line in reversed(t.splitlines()):
    if "bottleneck" in line:
        print(line.strip())
        break
PY
ps aux | grep mia_suppress_readout_v2 | grep -v grep || echo DONE_OR_DEAD
