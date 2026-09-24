cp /mnt/c/Users/research/w4090/apply_suppress_eval.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/part_occlusion_map.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/mia_suppress_readout.py /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/run_followup_all3.sh /home/research/e85/scripts/
cp /mnt/c/Users/research/w4090/viz_suppress.py /home/research/e85/scripts/ 2>/dev/null || true
chmod +x /home/research/e85/scripts/run_followup_all3.sh
ls /home/research/e85/models/shape_predictor_68_face_landmarks.dat
ls /home/research/e85/target_preds/mia.npz 2>/dev/null || echo "no mia.npz"
python3 -c "import sys; sys.path.insert(0,'/home/research/miniconda3/envs/tribev2/lib/python3.11/site-packages'); import nilearn; print('nilearn', nilearn.__version__)" 2>/dev/null || \
  /home/research/miniconda3/envs/tribev2/bin/python -c "import nilearn; print('nilearn', nilearn.__version__)"
