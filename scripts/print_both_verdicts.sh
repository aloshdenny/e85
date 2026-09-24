#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
for name in ("mia", "sins"):
    p = Path(f"/home/research/e85_scratch/{name}_suppress_v2_rerun.log")
    t = p.read_text(errors="replace")
    print("=" * 22, name.upper(), "=" * 22)
    for block in ("FACE means", "MIA HOLDOUT", "SINS HOLDOUT", "TRIBE-NEIGHBOR", "RANDOM FAIRFACE"):
        i = t.find(block)
        if i < 0:
            continue
        chunk = t[i:i+600].split("\n\n")[0]
        print(chunk)
        print()
    i = t.rfind("V2 VERDICT")
    print(t[i:i+450])
    print()
PY
