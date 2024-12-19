# Making Genomic Foundation Models more Foundational Requires Outlier Removal: A Case Study on DNABERT-2

The repo contains: 
1. **Official Implementation**  
   The official implementation of [Making Genomic Foundation Models more Foundational Requires Outlier Removal: A Case Study on DNABERT-2](to_be_decided).

2. **Quantization Adaptations**  
   Implementations of quantization methods `outlier_suppression`, `omniquant`, and `smoothquant` adapted for DNABERT-2.

3. **Fine-Tuning Code**  
   Fine-tuning code for DNABERT-2, including support for full fine-tuning, LoRA, and the newly added `QLoRA`, `LoftQ` methods.

4. **Outlier-free Pretraining Code**  
   Pretraining code with vanilla and outlier-free method.

5. **Outlier Testing Code**  
   Scripts and tools for testing outliers.

## Contents

- [Making Genomic Foundation Models more Foundational Requires Outlier Removal: A Case Study on DNABERT-2](#making-genomic-foundation-models-more-foundational-requires-outlier-removal-a-case-study-on-dnabert-2)
  - [Contents](#contents)
  - [1. Introduction](#1-introduction)
  - [2. Environment setup](#2-environment-setup)
  - [3. Pre-Training](#3-pre-training)
  - [4. Finetune](#4-finetune)
  - [5. Quantization](#5-quantization)
    - [5.1 outlier\_suppression](#51-outlier_suppression)
    - [5.2 Smoothquant](#52-smoothquant)
    - [5.3 Omniquant](#53-omniquant)
  - [6. Evaluation](#6-evaluation)
  - [7. Citation](#7-citation)

## 1. Introduction

GERM (GEnomic adapt-Robust Model) is a genomic foundation model designed to enhance efficiency and adaptability in genomic analysis. It replaces standard attention mechanisms with an outlier-free layer inspired by associative memory models, improving low-rank adaptation and quantization robustness. Building on DNABERT-2, GERM incorporates QLoRA and LoFTQ for efficient low-rank adaptation, while integrating outlier suppression, OmniQuant, and SmoothQuant for robust quantization, enabling state-of-the-art performance and efficient deployment on resource-constrained devices.

## 2. Environment setup

    # create and activate virtual python environment
    conda create -n germ python=3.8
    conda activate germ
    
    # install required packages
    pip install -r requirements.txt

## 3. Pre-Training
To perform outlier-free pretraining, navigate to `pretrain/outlier_free_pretrain` and run:

```bash
torchrun --nproc_per_node=4 run_mlm.py \
  --config_name "../DNABERT2" \
  --train_file "../data/dnabert_2_pretrain/train.txt" \
  --validation_file "../data/dnabert_2_pretrain/dev.txt" \
  --per_device_train_batch_size 512 \
  --per_device_eval_batch_size 512 \
  --do_train \
  --do_eval \
  --cache_dir .hf_cache \
  --output_dir "your/path/to/save" \
  --trust_remote_code=True \
  --tokenizer_name "zhihan1996/DNABERT-2-117M" \
  --gradient_accumulation_steps 4 \
  --max_seq_length 128 \
  --save_total_limit 5 \
  --weight_decay 1e-05 \
  --adam_beta2 0.95 \
  --learning_rate 1e-04 \
  --logging_steps 1 \
  --max_steps 10 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_ratio 0.05 \
  --preprocessing_num_workers 10 \
  --fp16 \
  --report_to "wandb" \
  --evaluation_strategy "steps" \
  --run_name "dnabert2" \
```

If you want to perform standard pretraining, navigate to the `pretrain/vanilla_pretrain` and run:

```bash
sh run_pretrain.sh
```

## 4. Finetune
To perform fine-tuning, adjust the model path, data path, output directory, and the number of GPUs in the script before running it.
```bash
# Full-finetune
sh finetune/scripts/full/run.sh

# LoRA
sh finetune/scripts/lora/run.sh

# QLoRA
sh finetune/scripts/qlora4/run.sh

# LoftQ
sh finetune/scripts/loftq/run.sh
```
## 5. Quantization

Before execution, make sure to replace the model and data paths accordingly.

### 5.1 outlier_suppression

You can set the quantization bit-width in `config.yaml`. 
```
quant:
    is_remove_padding: True
    ln:
        delay: True
    a_qconfig:
        quantizer: FixedFakeQuantize
        observer: AvgMinMaxObserver
        bit: n
        symmetric: False
        ch_axis: -1
    w_qconfig:
        quantizer: FixedFakeQuantize
        observer: MinMaxObserver
        bit: n
        symmetric: True
        ch_axis: 0
    calibrate: 256
```

```bash
cd outlier_suppression/exp/bert_ptq/twc_fine_gamma/dnabert
sh run.sh
```

### 5.2 Smoothquant

First, you need to generate activation scales.
```bash
cd smoothquant/examples
sh act_pipe.sh
```

After that, proceed with the Smoothquant.
```bash
sh ppl_pipe.sh
```

If you want to change the quantization bit-width, it is recommended to edit `smoothquant/smoothquant/fake_quant.py`, search for *n_bits=8*, and replace it with your desired value *n_bits=n*. After making the changes, reinstall the package.
```bash
cd smoothquant
python setup.py install
```

### 5.3 Omniquant

First, you need to get scales and shifts.
```bash
cd omniquant/OmniQuant/scripts
sh act_pipe.sh
```

After that, proceed with the Omniquant.
```bash
sh run.sh
```
You can modify the quantization bit-width within the script.
```bash
--wbits n \
--abits n \
```
## 6. Evaluation
To perform evaluation, navigate to the `evaluation` directory.
```bash
sh run.sh
```
In the script, you can choose whether to perform traditional WA quantization as shown below:  

```bash
--n_bits 8 \
--n_bits_act 8 \
--quantize
```

If you decide to perform quantization, enable `--quantize` and specify the desired bit-width.

## 7. Citation

If you have any question regarding our paper or codes, please feel free to start an issue.

If you use GERM in your work, please kindly cite our paper:

**GERM**

```
to be filled
```
