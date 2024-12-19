import os
import numpy as np
import logging
import sys
import argparse
import transformers
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
)
import datasets
import random
from datasets import load_metric
import torch  # noqa E401
import torch.fx
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import Subset


from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import sklearn


from quant_transformer.solver.utils import nt_utils
from quant_transformer.quantization.state import enable_calibration_woquantization, enable_quantization,\
        disable_all, enable_calibration_quantization, set_observer_name  # noqa: F401
from quant_transformer.quantization.observer import ObserverBase  # noqa: F401
from quant_transformer.quantization.fake_quant import LSQPlusFakeQuantize, QuantizeBase  # noqa: F401
from quant_model import quantize_model
import token_wise_clipping
logger = logging.getLogger("transformer")



class CustomTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if not self.is_world_process_zero():
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        self.save_model(run_dir, safe_serialization=False)

        if self.deepspeed:
            self.deepspeed.save_checkpoint(run_dir)

        self.state.save_to_json(os.path.join(run_dir, "trainer_state.json"))
        self._rotate_checkpoints(use_mtime=True)

def set_logger(config_progress):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config_progress.log_level
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(p: EvalPrediction):
    logits, labels = p.predictions, p.label_ids
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def make_huggingface_training_args(config_train, config_progress):
    training_args = TrainingArguments(
        seed=config_train.seed,
        output_dir=config_train.output_dir,
        overwrite_output_dir=config_train.overwrite_output_dir,
        do_train=config_train.do_train,
        do_eval=config_train.do_eval,
        do_predict=config_train.do_predict,
        evaluation_strategy=config_train.evaluation_strategy,
        eval_steps=config_train.eval_steps,
        per_device_train_batch_size=config_train.per_device_train_batch_size,
        per_device_eval_batch_size=config_train.per_device_eval_batch_size,
        gradient_accumulation_steps=config_train.gradient_accumulation_steps,
        eval_accumulation_steps=config_train.eval_accumulation_steps,
        learning_rate=config_train.learning_rate,
        weight_decay=config_train.weight_decay,
        max_grad_norm=config_train.max_grad_norm,
        num_train_epochs=config_train.num_train_epochs,
        max_steps=config_train.max_steps,
        lr_scheduler_type=config_train.lr_scheduler_type,
        warmup_ratio=config_train.warmup_ratio,
        warmup_steps=config_train.warmup_steps,
        gradient_checkpointing=config_train.gradient_checkpointing,
        log_level=config_progress.log_level,
        log_level_replica=config_progress.log_level_replica,
        logging_dir=config_progress.logging_dir,
        logging_strategy=config_progress.logging_strategy,
        logging_steps=config_progress.logging_steps,
        save_strategy=config_progress.save_strategy,
        save_steps=config_progress.save_steps,
        save_total_limit=config_progress.save_total_limit,
        save_on_each_node=config_progress.save_on_each_node,
        no_cuda=config_progress.no_cuda,
        run_name=config_progress.run_name,
        disable_tqdm=config_progress.disable_tqdm,
        load_best_model_at_end=config_progress.load_best_model_at_end,
        metric_for_best_model=config_progress.metric_for_best_model,
        greater_is_better=config_progress.greater_is_better,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config_progress.log_level = training_args.get_process_log_level()
    return training_args


def prepare_input_output(trainer, cali_data):
    logger.info('**prepare fp input and output**')
    data_loader = trainer.get_eval_dataloader(cali_data)
    fp_input, fp_output = [], []
    with torch.no_grad():
        for p in data_loader:
            tmp = {}
            for k, v in p.items():
                tmp[k] = v.cuda()
            #del tmp['labels']
            
            output = trainer.model(**tmp)[0].detach()
            fp_input.append(tmp)
            fp_output.append(output)
            torch.cuda.empty_cache()
    return fp_input, fp_output


def calibrate(trainer, fp_input):
    logger.info("*** Calibrate ***")
    with torch.no_grad():
        for batch in fp_input:
            trainer.model(**batch)
    torch.cuda.empty_cache()

# def evaluate(trainer, eval_datasets):
#     logger.info("*** Evaluate ***")
#     if not isinstance(eval_datasets, tuple):
#         eval_datasets = [eval_datasets]
#     all_metrics = []
#     dataset_size = len(eval_datasets[0])
#     print(dataset_size)
#     chunk_size =  dataset_size // trainer.args.per_device_train_batch_size    # Choose a smaller subset size
#     print(trainer.args.per_device_train_batch_size)
#     print(chunk_size)
#     trainer.model.eval()
#     for i in range(0, dataset_size, chunk_size):
#         with torch.no_grad():
#             subset = Subset(eval_datasets[0], range(i, min(i + chunk_size, dataset_size)))
#             metric = trainer.evaluate(eval_dataset=subset)
#             all_metrics.append(metric)
#         torch.cuda.empty_cache()
#     for i in range(len(all_metrics)):
#         trainer.log_metrics("eval", all_metrics[i])
#         trainer.save_metrics("eval", all_metrics[i])
#     print(all_metrics)

def evaluate(trainer, eval_datasets):
    logger.info("*** Evaluate ***")
    if not isinstance(eval_datasets, tuple):
        eval_datasets = [eval_datasets]
    metrics = []
    for i in range(len(eval_datasets)):
        metric = trainer.evaluate(eval_dataset=eval_datasets[i])
        metrics.append(metric)
    for i in range(len(metrics)):
        trainer.log_metrics("eval", metrics[i])
        trainer.save_metrics("eval", metrics[i])


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config_path):
    config = nt_utils.parse_config(config_path)
    set_seed(config.train.seed)
    if config.data.task_name == 'nt':
        config.progress.metric_for_best_model = 'perplexity'
    else:
        config.progress.metric_for_best_model = 'accuracy'
    training_args = make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    config.model.max_seq_length =  config.data.max_seq_length
    tokenizer, model = nt_utils.load_model(config.model, config.data)

    # max_seq_length
    config.data.max_seq_length = min(config.data.max_seq_length, tokenizer.model_max_length)

    train_datasets = nt_utils.SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=config.data.train_file, 
                                      kmer=config.data.kmer)
    eval_datasets = nt_utils.SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=config.data.validation_file, 
                                     kmer=config.data.kmer)
    test_datasets = nt_utils.SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=config.data.test_file, 
                                     kmer=config.data.kmer)
    data_collator = nt_utils.DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    # if config.data.pad_to_max_length:
    #     data_collator = default_data_collator
    # elif training_args.fp16:
    #     data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # else:
    #     data_collator = None
    
    model.eval()
    model.cuda()
    if getattr(config, 'quant', None):
        fp_model = model
        fp_model.eval()
        model = quantize_model(model, config)

    # print(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets[0],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if getattr(config, 'quant', None):
        cali_data = train_datasets.shuffle().select(range(config.quant.calibrate))
        fp_input, fp_output = prepare_input_output(trainer, cali_data)
             
        if config.quant.ln.delay:
            from gamma_migration import delay_ln
            trainer.model = delay_ln(trainer.model, config.quant, config.model)

        # calibrate the weight
        enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        calibrate(trainer, [fp_input[0]])

        if 'PruneMinMaxObserver' in config.quant.a_qconfig.observer:
            disable_all(trainer.model)
            set_observer_name(trainer.model)
            token_wise_clipping.token_wise_clipping(trainer, fp_input, fp_output, config)
            if 'LSQ' in config.quant.a_qconfig.quantizer:
                token_wise_clipping.learn_scale(trainer, fp_input, fp_output,
                                                getattr(config.quant, 'learn', {'lr': 1e-5, 'epoch': 3}))
        else:
            # calibrate the activation
            enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
            calibrate(trainer, fp_input)
        torch.cuda.empty_cache()

    if training_args.do_eval:
        
        if getattr(config, 'quant', None):
            enable_quantization(trainer.model)
        print(trainer.model)
        torch.cuda.empty_cache()
        evaluate(trainer, eval_datasets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
