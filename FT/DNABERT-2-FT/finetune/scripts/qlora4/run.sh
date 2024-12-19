export lr=1e-04
export num_gpu=1 # please change the value based on your setup

export DATA_CACHE_DIR=".hf_data"
export MODEL_CACHE_DIR=".hf_cache"

model_name_or_path="zhihan1996/DNABERT-2-117M"
type="zhihan"
method="QLora_double_4bit"
attn="vanilla"

# Define all your datasets
datasets_one=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4")
datasets_two=("0" "1" "2" "3" "4")
datasets_reconstructed=("reconstructed")
datasets_covid=("covid")

for data in "${datasets_one[@]}"
do
    # Determine the data path and other parameters based on the dataset
    if [[ $data == "H3"* || $data == "H4"* ]]; then
        data_path="../GUE/EMP/$data"
        model_max_length=128
        per_device_train_batch_size=64 
        per_device_eval_batch_size=64 
        gradient_accumulation_steps=1 
        num_train_epochs=3
        save_steps=200 
        eval_steps=200 
        warmup_steps=50 
    elif [[ $data == "prom_core_all" || $data == "prom_core_notata" ]]; then
        data_path="../GUE/prom/$data"
        model_max_length=20
        per_device_train_batch_size=16 
        per_device_eval_batch_size=32 
        gradient_accumulation_steps=1 
        num_train_epochs=4
        save_steps=400
        eval_steps=400
        warmup_steps=50 
    elif [[ $data == "prom_core_tata" ]]; then
        data_path="../GUE/prom/$data"
        model_max_length=20
        per_device_train_batch_size=16 
        per_device_eval_batch_size=32 
        gradient_accumulation_steps=1 
        num_train_epochs=10
        save_steps=200
        eval_steps=200
        warmup_steps=50 
    elif [[ $data == "prom_300_all" || $data == "prom_300_notata" ]]; then
        data_path="../GUE/prom/$data"
        model_max_length=70
        per_device_train_batch_size=16 
        per_device_eval_batch_size=32 
        gradient_accumulation_steps=1 
        num_train_epochs=4
        save_steps=400
        eval_steps=400
        warmup_steps=50 
    elif [[ $data == "prom_300_tata" ]]; then
        data_path="../GUE/prom/$data"
        model_max_length=70
        per_device_train_batch_size=16 
        per_device_eval_batch_size=32 
        gradient_accumulation_steps=1 
        num_train_epochs=10
        save_steps=200
        eval_steps=200
        warmup_steps=50 
    elif [[ $data == "tf0" || $data == "tf1" || $data == "tf2" || $data == "tf3" || $data == "tf4" ]]; then
        data_path="../GUE/tf/$data"
        model_max_length=30
        per_device_train_batch_size=16
        per_device_eval_batch_size=128 
        gradient_accumulation_steps=1 
        num_train_epochs=3
        save_steps=200
        eval_steps=200
        warmup_steps=30
    fi

    # Run the training command
    torchrun --nproc-per-node=${num_gpu} train02.py \
        --model_name_or_path $model_name_or_path \
        --data_path  $data_path \
        --kmer -1 \
        --run_name ${type}_DNABERT2_${lr}_${method}_${data} \
        --model_max_length $model_max_length \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate ${lr} \
        --num_train_epochs $num_train_epochs \
        --fp16 \
        --save_steps $save_steps \
        --output_dir output_${type}_${method}/${type}_dnabert2_${method}_${data} \
        --evaluation_strategy steps \
        --eval_steps $eval_steps \
        --warmup_steps $warmup_steps \
        --attn_type $attn \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --use_qlora True \
        --lora_alpha 16 \
        --lora_r 8 \
        --lora_dropout 0.05 \
        --lora_target_modules Wqkv \
        --save_model True \
        --load_in_4bit
done

    #4 gpu condition
    # per_device_train_batch_size=8
    # per_device_eval_batch_size=64 
    # gradient_accumulation_steps=1 
for data in "${datasets_two[@]}"
do    
    data_path="../GUE/mouse/$data"
    model_max_length=30
    per_device_train_batch_size=256
    per_device_eval_batch_size=256
    gradient_accumulation_steps=1 
    num_train_epochs=5
    save_steps=200
    eval_steps=200
    warmup_steps=30

    # Run the training command
    torchrun --nproc-per-node=${num_gpu} train02.py \
        --model_name_or_path $model_name_or_path \
        --data_path  $data_path \
        --kmer -1 \
        --run_name ${type}_DNABERT2_${lr}_${method}_${data} \
        --model_max_length $model_max_length \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate ${lr} \
        --num_train_epochs $num_train_epochs \
        --max_steps 1000 \
        --fp16 \
        --save_steps $save_steps \
        --output_dir output_${type}_${method}/${type}_dnabert2_${method}_${data} \
        --evaluation_strategy steps \
        --eval_steps $eval_steps \
        --warmup_steps $warmup_steps \
        --attn_type $attn \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --use_qlora True \
        --lora_alpha 16 \
        --lora_r 8 \
        --lora_dropout 0.05 \
        --lora_target_modules Wqkv \
        --save_model True \
        --load_in_4bit
done


for data in "${datasets_reconstructed[@]}"
do    
    data_path="../GUE/splice/$data"

    # Run the training command
    torchrun --nproc-per-node=${num_gpu} train02.py \
        --model_name_or_path $model_name_or_path \
        --data_path  $data_path \
        --kmer -1 \
        --run_name ${type}_DNABERT2_${lr}_${method}_${data} \
        --model_max_length 80 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 5 \
        --fp16 \
        --save_steps 200 \
        --output_dir output_${type}_${method}/${type}_dnabert2_${method}_${data} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --attn_type $attn \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --use_qlora True \
        --lora_alpha 16 \
        --lora_r 8 \
        --lora_dropout 0.05 \
        --lora_target_modules Wqkv \
        --save_model True \
        --load_in_4bit
done

for data in "${datasets_covid[@]}"
do    
    data_path="../GUE/virus/$data"

    # Run the training command
    torchrun --nproc-per-node=${num_gpu} train02.py \
        --model_name_or_path $model_name_or_path \
        --data_path  $data_path \
        --kmer -1 \
        --run_name ${type}_DNABERT2_${lr}_${method}_${data} \
        --model_max_length 256 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 8 \
        --fp16 \
        --save_steps 200 \
        --output_dir output_${type}_${method}/${type}_dnabert2_${method}_${data} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --attn_type $attn \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --use_qlora True \
        --lora_alpha 16 \
        --lora_r 8 \
        --lora_dropout 0.05 \
        --lora_target_modules Wqkv \
        --save_model True \
        --load_in_4bit
done