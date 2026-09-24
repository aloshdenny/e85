echo "=== studio/wild/lookalike blocks ==="
python3 - <<'PY'
from pathlib import Path
p = Path("/home/research/e85_scratch/followup_all3.log")
text = p.read_text(errors="replace")
# print from first STUDIO through VERDICT
start = text.find("STUDIO MIA")
end = text.find("Fetching fsaverage")
if start < 0:
    start = 0
if end < 0:
    end = len(text)
print(text[start:end])
print("--- import ---")
PY
rg -n "assert_redirectable_path" /home/research/e85/scripts/abliteration.py /home/research/e85/scripts/occlusion_saliency.py || true
echo "=== dlib ==="
/home/research/miniconda3/envs/tribev2/bin/python -c "import dlib; print('dlib', dlib.__version__)"
