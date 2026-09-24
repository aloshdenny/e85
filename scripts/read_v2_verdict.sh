python3 - <<'PY'
from pathlib import Path
t = Path("/home/research/e85_scratch/mia_suppress_v2.log").read_text(errors="replace")
idx = t.find("TRIBE bottleneck similarity")
print(t[idx:] if idx>=0 else t[-4000:])
PY
ls -lh /home/research/e85/abliterated/mia_suppress_readout.npz /home/research/e85/abliterated/mia_suppress_readout_v2.npz
