cp /home/research/e85_scratch/followup_all3.log /mnt/c/Users/research/w4090/followup_all3.log
# copy artifacts if present
for f in \
  /home/research/e85/target_preds/mia_suppressed.npz \
  /home/research/e85/target_preds/suppress_maps.npz \
  /home/research/e85/target_preds/lookalike_collateral.npz \
  /home/research/e85/abliterated/part_occlusion_suppressed.npy \
  /home/research/e85/interactive_study/suppress_toggle.html
 do
  if [ -f "$f" ]; then cp "$f" /mnt/c/Users/research/w4090/; fi
done
tail -25 /home/research/e85_scratch/followup_all3.log
grep -E "COLLATERAL|VERDICT|DONE followup|Saved" /home/research/e85_scratch/followup_all3.log | tail -20
