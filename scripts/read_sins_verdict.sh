python3 - <<'PY'
from pathlib import Path
t = Path("/home/research/e85_scratch/sins_suppress_v2.log").read_text(errors="replace")
i = t.find("FACE means")
print(t[i:] if i>=0 else t[-3000:])
PY
