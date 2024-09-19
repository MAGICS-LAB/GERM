#!/bin/bash

# ckpt=output_van200k_Full_double
# folder=van200k_dnabert2_Full_double
# type="van200"

# ckpt=output_zhihan_vanilla_Full_double
# folder=zhihan_vanilla_dnabert2_Full_double
# type="zh_van"

# ckpt=output_out200k_Full_double
# folder=out200k_dnabert2_Full_double
# type="out200"

ckpt=output_van180+20_softmax1_Full_double
folder=van180+20_dnabert2_Full_double
type="van180+20"

# ckpt=output_zhihan_softmax1_Full_double
# folder=zhihan_softmax1_dnabert2_Full_double
# type="zh_softmax1"

ckpt_dir=/scratch/hlv8980/FT_result

# checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )
checkpoints=( "covid" )
output_base_dir="/scratch/hlv8980/redo2/validate_int8"

for checkpoint in "${checkpoints[@]}"; do
    accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_sc.py \
    --seed 3407 \
    --preprocessing_num_workers 0 \
    --model_type bert \
    --max_seq_length 128 \
    --mlm_probability 0.15 \
    --per_device_eval_batch_size 128 \
    --attn_softmax softmax1 \
    --data_cache_dir .hf_data \
    --model_cache_dir .hf_cache \
    --model_name_or_path  "${ckpt_dir}/${ckpt}/${folder}_${checkpoint}" \
    --output_dir "${output_base_dir}/${type}/${checkpoint}" \
    --validation_file "/scratch/hlv8980/data/$checkpoint/dev.csv" \
    --train_file "/scratch/hlv8980/data/$checkpoint/train.csv" \
    --trust_remote_code \
    --gradient_accumulation_steps 1 \
    --run_name "dnabert2_val" \
    --n_bits 8 \
    --n_bits_act 8\
    --quantize
done
