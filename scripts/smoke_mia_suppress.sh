set -x
cd /home/research/e85
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then source ~/miniconda3/etc/profile.d/conda.sh; fi
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then source ~/anaconda3/etc/profile.d/conda.sh; fi
which conda || true
conda activate tribev2
python scripts/mia_suppress_readout.py --help
