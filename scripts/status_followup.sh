echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== procs ==="
ps aux | grep -E 'apply_suppress|part_occlusion|viz_suppress|run_followup|python scripts' | grep -v grep || echo none
echo "=== log tail ==="
tail -40 /home/research/e85_scratch/followup_all3.log 2>/dev/null || echo "(no log)"
echo "=== artifacts ==="
ls -lh /home/research/e85/target_preds/mia_suppressed.npz \
       /home/research/e85/target_preds/suppress_maps.npz \
       /home/research/e85/target_preds/lookalike_collateral.npz \
       /home/research/e85/abliterated/part_occlusion_suppressed.npy \
       /home/research/e85/interactive_study/suppress_toggle.html \
       /home/research/e85/target/lookalikes.zip 2>&1
echo "=== DONE marker ==="
grep -E "DONE followup|COLLATERAL|VERDICT|Saved ->" /home/research/e85_scratch/followup_all3.log 2>/dev/null | tail -30
