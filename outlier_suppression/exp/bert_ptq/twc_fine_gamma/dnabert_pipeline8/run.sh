#!/usr/bin/env bash
export WANDB_DISABLED=true
# Define arrays of checkpoints and datasets
ckpt_dir=/scratch/hlv8980/FT_result

# ckpt=output_out200k_Full_double
# folder=out200k_dnabert2_Full_double
# type="out200"

# ckpt=output_van200k_Full_double
# folder=van200k_dnabert2_Full_double
# type="van200"

ckpt=output_van160+40_Full_double
folder=van160+40_dnabert2_Full_double
type="van160+40"

output_base_dir="/scratch/hlv8980/redo2/validate_outsup8"

# Correctly define the checkpoints
checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )

cd /projects/p32013/DNA/outlier_suppression/exp/bert_ptq/twc_fine_gamma/dnabert_pipeline8

# Loop over checkpoints and datasets
for checkpoint in "${checkpoints[@]}"; do

    # Update the config file with new paths for each checkpoint
    echo "Running with checkpoint: ${ckpt_dir}/${ckpt}/${folder}_${checkpoint}"

    # Update the config file dynamically
    sed -i "s|model_name_or_path: .*|model_name_or_path: ${ckpt_dir}/${ckpt}/${folder}_${checkpoint}|" config.yaml
    sed -i "s|train_file: .*|train_file: /scratch/hlv8980/data/$checkpoint/train.csv|" config.yaml
    sed -i "s|validation_file: .*|validation_file: /scratch/hlv8980/data/$checkpoint/dev.csv|" config.yaml
    sed -i "s|test_file: .*|test_file: /scratch/hlv8980/data/$checkpoint/test.csv|" config.yaml
    sed -i "s|output_dir: .*|output_dir: ${output_base_dir}/${type}/${checkpoint}|" config.yaml

    # Execute the command
    PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
    torchrun --nproc_per_node=1 ../../../../quant_transformer/solver/ptq_dnabert_quant.py --config config.yaml

done
