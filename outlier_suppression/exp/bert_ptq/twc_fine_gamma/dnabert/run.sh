#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
torchrun --nproc_per_node=1 ../../../../quant_transformer/solver/ptq_dnabert_quant.py --config config.yaml

