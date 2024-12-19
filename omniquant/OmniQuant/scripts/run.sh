# Define arrays of checkpoints and datasets
ckpt_dir=path/to/your/model

export TOKENIZERS_PARALLELISM=false

ckpt=path/to/your/model
folder=path/to/your/model
type="type"

scales_base_dir="path/to/your/scales"
shifts_base_dir="path/to/your/shifts"
log_base_dir="path/to/log"
save_base_dir="path/to/save"

attn="softmax1"

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )


cd ../omniquant/OmniQuant

for checkpoint in "${checkpoints[@]}"; do

    # Update the config file with new paths for each checkpoint
    echo "Running with checkpoint: ${ckpt_dir}/${ckpt}/${folder}_${checkpoint}"

    python main.py \
        --model ${ckpt_dir}/${ckpt}/${folder}_${checkpoint} \
        --cache_dir ../omniquant/OmniQuant/cache \
        --data_path /scratch/hlv8980/data/${checkpoint} \
        --wbits 4 \
        --abits 4 \
        --lwc \
        --lwc_lr 1e-2 \
        --let_lr 5e-3 \
        --epochs 10 \
        --nsamples 128 \
        --model_max_length 128 \
        --preprocessing_num_workers 0 \
        --per_device_eval_batch_size 128 \
        --tasks dnabert \
        --do_eval True \
        --eval \
        --act-scales ${scales_base_dir}/${type}/${folder}_${checkpoint}.pt \
        --act-shifts ${shifts_base_dir}/${type}/${folder}_${checkpoint}.pt \
        --output_dir ${log_base_dir}/${type}/${checkpoint} \
        --save_dir ${save_base_dir}/${type}/${checkpoint} \
        --real_quant \
        --group_size 128

done
