echo "=== home disk ==="
df -h /home /home/research /home/research/e85 | uniq
echo "=== dlib predictor ==="
ls -lh /home/research/e85/models/shape_predictor_68_face_landmarks.dat
echo "=== lookalikes occlusion members ==="
python3 - <<'PY'
import zipfile
z=zipfile.ZipFile("/home/research/e85/target/lookalikes.zip")
print([n for n in z.namelist() if n.startswith("occlusion/")][:5], "count", sum(1 for n in z.namelist() if n.startswith("occlusion/")))
print("total", len(z.namelist()))
PY
