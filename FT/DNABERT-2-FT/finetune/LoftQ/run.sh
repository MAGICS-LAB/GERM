export attn=softmax1
export HF_ALLOW_CODE_EXECUTION=1

ckpt=path/to/your/model
folder=path/to/your/model
type="type"

ckpt_dir=path/to/your/model

checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" )
covid=( "covid" )
reconstructed=( "reconstructed" )
output_base_dir="path/to/save/dir"

for checkpoint in "${checkpoints[@]}"; do
    python quantize_save.py \
    --model_name_or_path "${ckpt_dir}/${ckpt}/${folder}" \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir "${output_base_dir}/${type}/${checkpoint}" \
    --attn_type $attn\
    --trust_remote_code \
    --num_labels 2
done

for checkpoint in "${covid[@]}"; do
    python quantize_save.py \
    --model_name_or_path "${ckpt_dir}/${ckpt}/${folder}" \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir "${output_base_dir}/${type}/${checkpoint}" \
    --attn_type $attn\
    --trust_remote_code \
    --num_labels 9
done

for checkpoint in "${reconstructed[@]}"; do
    python quantize_save.py \
    --model_name_or_path "${ckpt_dir}/${ckpt}/${folder}" \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir "${output_base_dir}/${type}/${checkpoint}" \
    --attn_type $attn\
    --trust_remote_code \
    --num_labels 3
done