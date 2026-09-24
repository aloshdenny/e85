mkdir -p /home/research/e85/data
cp -v /mnt/c/Users/research/w4090/fairface_ffhq.zip /home/research/e85/data/fairface_ffhq.zip
ls -lh /home/research/e85/data/fairface_ffhq.zip
# leave the C: copy for now; can delete after job starts
python3 - <<'PY'
import zipfile
z=zipfile.ZipFile("/home/research/e85/data/fairface_ffhq.zip")
print("members", len(z.namelist()), "test", z.testzip())
print("sample", z.namelist()[:3])
PY
