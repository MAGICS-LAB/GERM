python generate_act_scales_mod.py \
    --model_name_or_path /projects/p32013/DNA/models/ckpt_recv/outEff/checkpoint-200000 \
    --output-path /projects/p32013/DNA/smoothquant/act_scales/out200.pt \
    --num_samples 512 \
    --seq_len 128 \
    --train_file /scratch/hlv8980/data/0/train.csv \
    --validation_file /scratch/hlv8980/data/0/dev.csv \