# Define arrays of checkpoints and datasets
ckpt_dir=/scratch/hlv8980/FT_result

# ckpt=output_van200k_Full_double
# folder=van200k_dnabert2_Full_double
# type="van200"

# ckpt=output_zhihan_vanilla_Full_double
# folder=zhihan_vanilla_dnabert2_Full_double
# type="zh_van"

# ckpt=output_out200k_Full_double
# folder=out200k_dnabert2_Full_double
# type="out200"

# ckpt=output_van180+20_softmax1_Full_double
# folder=van180+20_dnabert2_Full_double
# type="van180+20"

ckpt=output_van160+40_Full_double
folder=van160+40_dnabert2_Full_double
type="van160+40"

# ckpt=output_zhihan_softmax1_Full_double
# folder=zhihan_softmax1_dnabert2_Full_double
# type="zh_softmax1"

# ckpt=output_van100+100_Full_double
# folder=van100+100_dnabert2_Full_double
# type="van100+100"

act_base_dir="/scratch/hlv8980/act_sact_scales"
output_base_dir="/scratch/hlv8980/redo2/validate_smooth8"

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )

cd /projects/p32013/DNA/smoothquant

for checkpoint in "${checkpoints[@]}"; do

    # Update the config file with new paths for each checkpoint
    echo "Running with checkpoint: ${ckpt_dir}/${ckpt}/${folder}_${checkpoint}"

    python smoothquant/ppl_eval_bert.py \
        --alpha 0.5 \
        --model_path ${ckpt_dir}/${ckpt}/${folder}_${checkpoint} \
        --act_scales_path ${act_base_dir}/${type}/${checkpoint}.pt \
        --validation_file /scratch/hlv8980/data/${checkpoint}/dev.csv \
        --per_device_eval_batch_size 32 \
        --output_dir "${output_base_dir}/${type}/${checkpoint}" \
        --seed 3407 \
        --smooth \
        --quantize \


done
