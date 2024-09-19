#!/bin/bash

#SBATCH --job-name=DNABERT2-pretraining
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --account=p32301

source /software/anaconda3/2022.05/etc/profile.d/conda.sh
# Activate conda environment
conda activate /projects/p32301/DNABERT/gcc9_env

# Set Wandb API key
export WANDB_API_KEY="763e81099b1af125e29d6b6fc04c587285f69c42"

# Change directory to the project's code directory
cd /projects/p32301/DNABERT/DNABERT-2

# Run the python script with arguments
# python run_mlm.py \
#   --config_name "/projects/p32301/DNABERT/DNABERT-2-117M" \
#   --train_file "/projects/p32301/DNABERT/data/dnabert_2_pretrain/train.txt" \
#   --validation_file "/projects/p32301/DNABERT/data/dnabert_2_pretrain/dev.txt" \
#   --per_device_train_batch_size 512 \
#   --per_device_eval_batch_size 512 \
#   --do_train \
#   --do_eval \
#   --output_dir "/projects/p32301/DNABERT/models/dnabert2_pretrain" \
#   --trust_remote_code=True \
#   --tokenizer_name="/projects/p32301/DNABERT/DNABERT-2-117M" \
#   --gradient_accumulation_steps 4 \
#   --save_total_limit 20 \
#   --weight_decay 1e-05 \
#   --adam_beta2 0.95 \
#   --max_seq_length 128 \
#   --learning_rate 4e-04 \
#   --logging_steps 1 \
#   --max_steps 200000 \
#   --eval_steps 5000 \
#   --save_steps 1000 \
#   --warmup_ratio 0.05

torchrun --nproc_per_node=4 run_mlm.py \
  --config_name "/projects/p32301/DNABERT/DNABERT-2-117M" \
  --train_file "/projects/p32301/DNABERT/data/dnabert_2_pretrain/train.txt" \
  --validation_file "/projects/p32301/DNABERT/data/dnabert_2_pretrain/dev.txt" \
  --per_device_train_batch_size 512 \
  --per_device_eval_batch_size 512 \
  --do_train \
  --do_eval \
  --output_dir "/projects/p32301/DNABERT/models/dnabert2_pretrain" \
  --trust_remote_code=True \
  --tokenizer_name="/projects/p32301/DNABERT/DNABERT-2-117M" \
  --gradient_accumulation_steps 1 \
  --max_seq_length 128 \
  --save_total_limit 20 \
  --weight_decay 1e-05 \
  --adam_beta2 0.95 \
  --learning_rate 4e-04 \
  --logging_steps 1 \
  --max_steps 200000 \
  --eval_steps 5000 \
  --save_steps 1000 \
  --warmup_ratio 0.05 \
  --preprocessing_num_workers 24 \
  --evaluation_strategy "steps"


