cd /home/user/DNA/FT/validate

# nohup bash run.sh > run.out 2>&1 &

torchrun --nproc_per_node=1 val.py \
  --model_name_or_path "zhihan1996/DNABERT-2-117M" \
  --config_name "zhihan1996/DNABERT-2-117M" \
  --train_file "/home/user/DNA/data/dnabert_2_pretrain/train.txt" \
  --validation_file "/home/user/DNA/data/dnabert_2_pretrain/dev.txt" \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --do_eval \
  --output_dir "/home/user/DNA/FT/validate/output/dnabert2_pretrain_hg_500K" \
  --trust_remote_code=True \
  --tokenizer_name "zhihan1996/DNABERT-2-117M" \
  --gradient_accumulation_steps 1 \
  --max_seq_length 128 \
  --preprocessing_num_workers 128 \
  --fp16 \
  --run_name "vanilla_dnabert2_hg_500K"   