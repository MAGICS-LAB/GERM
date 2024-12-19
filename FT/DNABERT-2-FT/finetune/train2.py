import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
from peft import prepare_model_for_kbit_training
from transformers import TrainerCallback

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from transformers.models.bert.configuration_bert import BertConfig
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    BertForSequenceClassification,
)

from transformers_language.args import parse_args
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertUnpadSelfAttentionWithExtras,
    Q8bitBertUnpadSelfAttentionWithExtras,
    Q4bitBertUnpadSelfAttentionWithExtras,
)
from transformers_language.models.bert_attention2 import (
    AttentionGateType,
    Q8bitBertSelfAttentionWithExtras,
    Q4bitBertSelfAttentionWithExtras,
)
# from transformers_language.models.quantized_dnabert import QuantizedBertForMaskedLM
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            print(f"Step {state.global_step}: loss = {logs['loss']}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_qlora: bool = field(default=False, metadata={"help": "whether to use QLoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    load_in_8bit: bool = field(default=False, metadata={"help": "whether to load in 8-bit"})
    load_in_4bit: bool = field(default=False, metadata={"help": "whether to load in 4-bit"})
    int8: bool = field(default=False, metadata={"help": "whether to use LLM.int8()"})
    attn_type: str = field(default="vanilla", metadata={"help": "which type of attention to use"})
    use_loftq: bool = field(default=False, metadata={"help": "whether to use LoftQ"}) 


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-5)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def save_model(trainer: transformers.Trainer, output_dir: str):
    """Save the model, optimizer state, and other information."""
    if trainer.args.should_save:
        # Save the model
        trainer.save_model(output_dir)
        
        # Save the optimizer state (if available)
        if trainer.optimizer is not None:
            optimizer_state_dict = trainer.optimizer.state_dict()
            torch.save(optimizer_state_dict, os.path.join(output_dir, 'optimizer_state_dict.pth'))
        
        print(f"Model and optimizer state saved to {output_dir}")


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

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
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    bnb_config = None
    # congigure LLM.int8()
    if model_args.int8:
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    # configure QLoRA
    if model_args.use_qlora or model_args.use_loftq:
        if model_args.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif model_args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    # load model
    config = BertConfig.from_pretrained(model_args.model_name_or_path, num_labels=train_dataset.num_labels, trust_remote_code=True,)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        quantization_config = bnb_config,
        # _fast_init=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_args.use_qlora:
        if model_args.load_in_8bit:
            for layer_idx in range(len(model.bert.encoder.layer)):
                    old_self = model.bert.encoder.layer[layer_idx].attention.self
                    print("----------------------------------------------------------")
                    print("Inside BERT custom attention")
                    print("----------------------------------------------------------")
                    new_self = Q8bitBertUnpadSelfAttentionWithExtras(
                        config,
                        position_embedding_type=None,
                        softmax_fn=SOFTMAX_MAPPING[model_args.attn_type],
                        ssm_eps=None,
                        tau=None,
                        max_seq_length=tokenizer.model_max_length,
                        skip_attn=False,
                        fine_tuning=True,
                    ).to(device)
                    
                    
                    
                    # copy loaded weights
                    if model_args.model_name_or_path is not None:
                        new_self.load_state_dict(old_self.state_dict(), strict=False)
                    model.bert.encoder.layer[layer_idx].attention.self = new_self

        elif model_args.load_in_4bit:
            for layer_idx in range(len(model.bert.encoder.layer)):
                    old_self = model.bert.encoder.layer[layer_idx].attention.self
                    print("----------------------------------------------------------")
                    print("Inside BERT custom attention")
                    print("----------------------------------------------------------")
                    new_self = Q4bitBertUnpadSelfAttentionWithExtras(
                        config,
                        position_embedding_type=None,
                        softmax_fn=SOFTMAX_MAPPING[model_args.attn_type],
                        ssm_eps=None,
                        tau=None,
                        max_seq_length=tokenizer.model_max_length,
                        skip_attn=False,
                        fine_tuning=True,
                    ).to(device)

                    # copy loaded weights
                    if model_args.model_name_or_path is not None:
                        new_self.load_state_dict(old_self.state_dict(), strict=False)
                    model.bert.encoder.layer[layer_idx].attention.self = new_self
    # else:
    #     for layer_idx in range(len(model.bert.encoder.layer)):
    #                 old_self = model.bert.encoder.layer[layer_idx].attention.self
    #                 print("----------------------------------------------------------")
    #                 print("Inside BERT custom attention")
    #                 print("----------------------------------------------------------")
    #                 new_self = BertUnpadSelfAttentionWithExtras(
    #                     config,
    #                     position_embedding_type=None,
    #                     softmax_fn=SOFTMAX_MAPPING[model_args.attn_type],
    #                     ssm_eps=None,
    #                     tau=None,
    #                     max_seq_length=tokenizer.model_max_length,
    #                     skip_attn=False,
    #                     fine_tuning=True,
    #                 ).to(device)

    #                 # copy loaded weights
    #                 if model_args.model_name_or_path is not None:
    #                     new_self.load_state_dict(old_self.state_dict(), strict=False)
    #                 model.bert.encoder.layer[layer_idx].attention.self = new_self

        
    if model_args.use_qlora:

        for layer in model.bert.encoder.layer:
            layer.gradient_checkpointing = True
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    
    if model_args.use_lora or model_args.use_qlora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if model_args.use_loftq:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, num_labels=train_dataset.num_labels, trust_remote_code=True,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, 
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            config=config,
            trust_remote_code=True,
        )

        for layer_idx in range(len(base_model.bert.encoder.layer)):
            old_self = base_model.bert.encoder.layer[layer_idx].attention.self
            print("----------------------------------------------------------")
            print("Inside BERT custom attention")
            print("----------------------------------------------------------")
            new_self = Q4bitBertUnpadSelfAttentionWithExtras(
                config,
                position_embedding_type=None,
                softmax_fn=SOFTMAX_MAPPING[model_args.attn_type],
                ssm_eps=None,
                tau=None,
                max_seq_length=tokenizer.model_max_length,
                skip_attn=False,
                fine_tuning=True,
            ).to(device)

            # copy loaded weights
            if model_args.model_name_or_path is not None:
                new_self.load_state_dict(old_self.state_dict(), strict=False)
            base_model.bert.encoder.layer[layer_idx].attention.self = new_self

        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            subfolder="loftq_init",
            is_trainable=True,
        )
        

    # define trainer
    print(model)
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   callbacks=[PrintLossCallback]  # Add the callback here
    )

    trainer.train()
    print('-----Get Trainer-----')
    if training_args.save_model:
        trainer.save_state()
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        save_model(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()
