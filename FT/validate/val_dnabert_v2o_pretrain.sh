# Set Wandb API key
export WANDB_API_KEY="763e81099b1af125e29d6b6fc04c587285f69c42"
# --resume_from_checkpoint "/projects/p32301/DNABERT/models/dnabert2_pretrain_hopfield_debug_10K/checkpoint-10000" \

source /software/anaconda3/2022.05/etc/profile.d/conda.sh
# Activate conda environment
conda activate /home/ysj6764/.conda/envs/outlier

cd /projects/p32301/DNABERT/FT/validate

# nohup bash val_dnabert_v2o_pretrain.sh > val2.out 2>&1 &

python val2.py \
  --model_name_or_path "/projects/p32301/DNABERT/models/van2out_20K+/checkpoint-200000" \
  --train_file "/projects/p32301/DNABERT/data/dnabert_2_pretrain/train.txt" \
  --validation_file "/projects/p32301/DNABERT/data/dnabert_2_pretrain/dev.txt" \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --do_eval \
  --output_dir "/projects/p32301/DNABERT/FT/validate/output/dnabert2_pretrain_v2o_20K+" \
  --trust_remote_code=True \
  --tokenizer_name "zhihan1996/DNABERT-2-117M" \
  --gradient_accumulation_steps 12 \
  --max_seq_length 128 \
  --save_total_limit 20 \
  --weight_decay 1e-05 \
  --adam_beta2 0.95 \
  --learning_rate 1e-04 \
  --logging_steps 1 \
  --max_steps 200000 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_ratio 0.05 \
  --preprocessing_num_workers 24 \
  --evaluation_strategy "steps" \
  --fp16 \
  --run_name "v2o_dnabert2_val_20K+" 

  