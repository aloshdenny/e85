python3 - <<'PY'
from pathlib import Path
text = Path("/home/research/e85_scratch/occlusion_suppressed.log").read_text(errors="replace")
# strip carriage returns / tqdm junk by keeping lines with report content
keep_prefixes = (
    "images ", "skipped ", "patched ", "Loading",
    "TARGET", "GENERAL", "=", "-", "part ",
    "Part-profile", "Target vs", "  FACE", "  OFA", "  FFA",
    "    ", "Saved ->", "DONE", "  ...",
)
# find last occurrence of TARGET table (the printed area-adjusted one)
idx = text.rfind("TARGET (Mia)")
if idx < 0:
    idx = text.find("skipped")
chunk = text[idx:] if idx >= 0 else text[-5000:]
lines = []
for line in chunk.splitlines():
    s = line.strip("\r")
    if "\x1b" in s or s.startswith("\r") or "Encoding video" in s or "Loading weights" in s:
        continue
    if s.startswith("%") or "|█" in s or "| " in s[:6]:
        continue
    lines.append(s)
print("\n".join(lines[-120:]))
PY
