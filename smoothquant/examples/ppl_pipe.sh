# Define arrays of checkpoints and datasets
ckpt_dir=path/to/your/model

ckpt=path/to/your/model
folder=path/to/your/model
type="type"

act_base_dir="path/to/act_sact_scales"
output_base_dir="path/to/your/output"

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )

cd ../smoothquant

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
        --seed 3000 \
        --smooth \
        --quantize \
        
done
