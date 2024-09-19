import json
import logging
import math
import os
import random
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, load_dataset, load_from_disk
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from quantization.range_estimators import OptMethod, RangeEstimators
from transformers_language.args import parse_args
from transformers_language.dataset_setups import DatasetSetups
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertSelfAttentionWithExtras,
)
from transformers_language.models.quantized_bert import QuantizedBertForMaskedLM
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)