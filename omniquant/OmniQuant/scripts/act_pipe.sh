# Define arrays of checkpoints and datasets
ckpt_dir=path/to/your/model

ckpt=path/to/your/model
folder=path/to/your/model
type="type"

scales_base_dir="path/to/your/scales"
shifts_base_dir="path/to/your/shifts"

attn="softmax1"

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )


cd ../omniquant/OmniQuant

for checkpoint in "${checkpoints[@]}"; do

    # Update the config file with new paths for each checkpoint
    echo "Running with checkpoint: ${ckpt_dir}/${ckpt}/${folder}_${checkpoint}"

    python generate_act_scale_shift_mod.py \
        --model ${ckpt_dir}/${ckpt}/${folder}_${checkpoint} \
        --model_max_length 128 \
        --cache_dir ../omniquant/OmniQuant/cache \
        --data_path /scratch/hlv8980/data/${checkpoint} \
        --preprocessing_num_workers 0 \
        --trust_remote_code True \
        --attn_softmax $attn \
        --scales_output_path ${scales_base_dir}/${type} \
        --shifts_output_path ${shifts_base_dir}/${type} \



done



