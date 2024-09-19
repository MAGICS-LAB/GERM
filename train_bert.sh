export TASK_NAME=stsb
export HF_HOME="/projects/p32013/.cache/"
export attn=softmax1
python validate_sc2.py \
  --model_name_or_path /projects/p32013/DNA/FT/DNABERT-2-FT/finetune/gleu_output/$attn/$TASK_NAME \
  --task_name $TASK_NAME \
  --attn_softmax $attn \
  --do_eval \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /projects/p32013/DNA/FT/DNABERT-2-FT/finetune/gleu_output/${attn}_quantized/$TASK_NAME/ \
  --quantize True \
  --preprocessing_num_workers 0 \
  --pad_to_max_length True \
  --est_num_batches 16 