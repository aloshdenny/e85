python3 - <<'PY'
from pathlib import Path
text = Path("/home/research/e85_scratch/occlusion_suppressed.log").read_text(errors="replace")
# find first useful table
for key in ["images (", "skipped ", "TARGET (Mia)", "GENERAL --", "Part-profile", "Target vs general"]:
    i = text.find(key)
    print(f"\n# marker {key!r} at {i}")
# print from first TARGET or 'images'
start = text.find("images (")
if start < 0:
    start = text.find("skipped")
if start < 0:
    start = max(0, len(text)-8000)
print(text[start:])
PY
ls -lh /home/research/e85/abliterated/part_occlusion_suppressed.npy /home/research/e85/interactive_study/suppress_toggle.html
