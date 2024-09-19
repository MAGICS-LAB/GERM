import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

def evaluate_model(model, tokenizer, data):
    model.eval()
    predictions = []
    actuals = data['label'].tolist()
    
    for text in data['sequence']:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)["input_ids"]
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model(inputs)[0]
        predictions.append(torch.argmax(outputs, dim=-1))
    
    predictions = torch.cat(predictions).cpu().numpy()

    # Compute metrics
    return {
        "accuracy": accuracy_score(actuals, predictions),
        "f1": f1_score(actuals, predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(actuals, predictions),
        "precision": precision_score(actuals, predictions, average="macro", zero_division=0),
        "recall": recall_score(actuals, predictions, average="macro", zero_division=0)
    }

# Load tokenizer and model configuration
# model_path = "/projects/p32301/DNABERT/FT/DNABERT-2-FT/finetune/output_vanilla_full/Vanilla_dnabert2_Full_spliceonly/checkpoint-11400"
model_path = "/projects/p32301/DNABERT/FT/DNABERT-2-FT/finetune/output_vanilla_full/Vanilla_dnabert2_Full_prom_core_all/checkpoint-11800"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# config = AutoConfig.from_pretrained(model_path, num_labels=3, local_files_only=True)
config = AutoConfig.from_pretrained(model_path, num_labels=2, local_files_only=True)

# Load and evaluate the normal model
model = BertForSequenceClassification.from_pretrained(model_path, config=config)
data_path = "/projects/p32301/DNABERT/FT/DNABERT-2-FT/GUE/prom/prom_core_all/test.csv"
# data_path = "/projects/p32301/DNABERT/FT/DNABERT-2-FT/GUE/splice/reconstructed/test.csv"
data = pd.read_csv(data_path)
metrics_normal = evaluate_model(model.cuda(), tokenizer, data)

# Load and evaluate the 8-bit quantized model
model_8bit = BertForSequenceClassification.from_pretrained(model_path, config=config, load_in_8bit=True)
metrics_8bit = evaluate_model(model_8bit.cuda(), tokenizer, data)

# Calculate absolute differences and combine results
abs_diffs = {f"abs_diff_{metric}": abs(metrics_normal[metric] - metrics_8bit[metric]) for metric in metrics_normal}
results = pd.DataFrame({
    "model": ["vanilla DNABERT-2", "int8 quantized DNABERT-2"],
    "accuracy": [metrics_normal['accuracy'], metrics_8bit['accuracy']],
    "f1": [metrics_normal['f1'], metrics_8bit['f1']],
    "matthews_correlation": [metrics_normal['matthews_correlation'], metrics_8bit['matthews_correlation']],
    "precision": [metrics_normal['precision'], metrics_8bit['precision']],
    "recall": [metrics_normal['recall'], metrics_8bit['recall']],
    "task": ["core_all", "core_all"],
    **abs_diffs
})

# output_path = "/projects/p32301/DNABERT/FT/DNABERT-2-FT/finetune/output_vanilla_full/Vanilla_dnabert2_Full_spliceonly/eval.csv"
output_path = "/projects/p32301/DNABERT/FT/DNABERT-2-FT/finetune/output_vanilla_full/Vanilla_dnabert2_Full_prom_core_all/eval.csv"
results.to_csv(output_path, index=False)

print(f"Stored evaluation results in {output_path}.")
