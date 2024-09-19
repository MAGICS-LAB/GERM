export WANDB_API_KEY="763e81099b1af125e29d6b6fc04c587285f69c42"

torchrun --nproc-per-node=2 run_mlm_debug.py \
  --config_name "/home/user/DNA/DNABERT-2-117M" \
  --train_file "../data/dnabert_2_pretrain/train.txt" \
  --validation_file "../data/dnabert_2_pretrain/dev.txt" \
  --per_device_train_batch_size 1024 \
  --per_device_eval_batch_size 1024 \
  --do_train \
  --do_eval \
  --output_dir "../models/dnabert2_pretrain_debug_200K" \
  --trust_remote_code=True \
  --tokenizer_name "zhihan1996/DNABERT-2-117M" \
  --gradient_accumulation_steps 6 \
  --max_seq_length 128 \
  --save_total_limit 20 \
  --weight_decay 1e-05 \
  --adam_beta2 0.95 \
  --learning_rate 4e-04 \
  --logging_steps 1 \
  --max_steps 200000 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_ratio 0.05 \
  --preprocessing_num_workers 48 \
  --evaluation_strategy "steps" \
  --ddp_backend="nccl" \
  --fp16 \
  --report_to "wandb" \
  --run_name "dnabert2_pretrain_debug_200K" \
  --resume_from_checkpoint "../models/dnabert2_pretrain_debug_50K/checkpoint-50000-1" \


