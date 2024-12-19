torchrun --nproc-per-node=1 run_mlm.py \
  --config_name "../DNABERT-2-117M" \
  --train_file "../data/dnabert_2_pretrain/train.txt" \
  --validation_file "../data/dnabert_2_pretrain/dev.txt" \
  --per_device_train_batch_size 1024 \
  --per_device_eval_batch_size 1024 \
  --do_train \
  --do_eval \
  --output_dir "your/path/to/save" \
  --trust_remote_code=True \
  --tokenizer_name "zhihan1996/DNABERT-2-117M" \
  --gradient_accumulation_steps 6 \
  --max_seq_length 128 \
  --save_total_limit 20 \
  --weight_decay 1e-05 \
  --adam_beta2 0.95 \
  --learning_rate 4e-04 \
  --logging_steps 1 \
  --max_steps 10 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_ratio 0.05 \
  --preprocessing_num_workers 1 \
  --evaluation_strategy "steps" \
  --fp16 \
  --run_name "dnabert2" \


