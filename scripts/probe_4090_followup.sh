echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== e85 ==="
ls /home/research/e85/abliterated/ 2>/dev/null
ls /home/research/e85/target/mia.zip 2>/dev/null
ls /home/research/e85/"fairface + ffhq"/ 2>/dev/null | head
echo "=== predictor ==="
find /home/research/e85 /home/research -name "shape_predictor_68_face_landmarks.dat" 2>/dev/null | head
echo "=== attributes ==="
find /home/research/e85 /home/research/e85_scratch -name "attributes.npz" 2>/dev/null | head
echo "=== occlusion npy ==="
find /home/research -name "part_occlusion*.npy" 2>/dev/null | head
echo "=== idle python ==="
ps aux | grep -E "python scripts" | grep -v grep || echo none
