salloc --account=p32013 -p gengpu --gres gpu:a100:1 --cpus-per-task 24 --mem 100G --constraint=sxm -t 48:00:00 
salloc --account=p32013 -p gengpu-long --gres gpu:a100:1 --cpus-per-task 24 --mem 100G -t 240:00:00 

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

# ckpt=output_zhihan_softmax1_Full_double
# folder=zhihan_softmax1_dnabert2_Full_double
# type="zh_softmax1"