source /software/anaconda3/2022.05/etc/profile.d/conda.sh
# Activate conda environment
conda activate /home/ysj6764/.conda/envs/smoothquant

cd /projects/p32301/DNABERT/smoothquant/examples

python generate_act_scales.py \
    --model-name "/projects/p32301/DNABERT/models/ckpt_recv/outEff/checkpoint-50000" \
    --output-path "/projects/p32301/DNABERT/smoothquant/act_scales/outEff50K.pt"\
    --num-samples 512 \
    --seq-len 512 \
    --dataset-path "/projects/p32301/DNABERT/data/dnabert_2_pretrain/dev.txt"