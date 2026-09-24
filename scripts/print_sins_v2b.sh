#!/usr/bin/env bash
# Extract ROI tables + verdict from Sins v2b log (no pipes for Windows OpenSSH).
python3 - <<'PY'
from pathlib import Path
p = Path("/home/research/e85_scratch/sins_suppress_v2b.log")
text = p.read_text(errors="replace")
# keep from FACE means onward, drop tqdm noise already gone at this point
start = text.find("TRIBE bottleneck")
if start < 0:
    start = text.find("FACE means")
chunk = text[start:]
# strip leftover carriage junk
lines = []
for line in chunk.splitlines():
    if "\r" in line:
        line = line.split("\r")[-1]
    lines.append(line)
print("\n".join(lines))
PY
