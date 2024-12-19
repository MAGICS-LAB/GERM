---
metrics:
- matthews_correlation
- f1
tags:
- biology
- medical
---
This is the official pre-trained model introduced in [DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome
](https://arxiv.org/pdf/2306.15006.pdf).

We sincerely appreciate the MosaicML team for the [MosaicBERT](https://openreview.net/forum?id=5zipcfLC2Z) implementation, which serves as the base of DNABERT-2 development. 

DNABERT-2 is a transformer-based genome foundation model trained on multi-species genome. 

To load the model from huggingface:
```
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
```

To calculate the embedding of a dna sequence
```
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape) # expect to be 768
```