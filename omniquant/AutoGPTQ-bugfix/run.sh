module purge
module load gcc/9.2.0

source /software/anaconda3/2022.05/etc/profile.d/conda.sh
# Activate conda environment
conda activate /home/ysj6764/.conda/envs/omniquant

cd /projects/p32301/DNABERT/omniquant/AutoGPTQ-bugfix

pip install -v .
