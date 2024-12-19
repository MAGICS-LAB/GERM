# Define arrays of checkpoints and datasets
ckpt_dir=path/to/your/model

ckpt=path/to/your/model
folder=path/to/your/model
type="type"

act_base_dir="path/to/act"

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )


for checkpoint in "${checkpoints[@]}"; do

    # Update the config file with new paths for each checkpoint
    echo "Running with checkpoint: ${ckpt_dir}/${ckpt}/${folder}_${checkpoint}"

    python generate_act_scales_mod.py \
        --model_name_or_path ${ckpt_dir}/${ckpt}/${folder}_${checkpoint} \
        --output-path ${act_base_dir}/${type}/${checkpoint}.pt \
        --num_samples 512 \
        --seq_len 128 \
        --train_file path/to/${checkpoint}/train.csv \
        --validation_file path/to/${checkpoint}/dev.csv \

done

