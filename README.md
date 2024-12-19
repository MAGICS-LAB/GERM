# Making Genomic Foundation Models more Foundational Requires Outlier Removal: A Case Study on DNABERT-2

# Repository Contents

1. **Official Implementation**  
   The official implementation of [Making Genomic Foundation Models more Foundational Requires Outlier Removal: A Case Study on DNABERT-2](to_be_decided).

2. **Adaptations**  
   Implementations of `outlier_suppression`, `omniquant`, and `smoothquant` adapted for DNABERT-2.

3. **OutlierEfficiency Pretraining Code**  
   Code for pretraining using the OutlierEfficiency method.

4. **Outlier Testing Code**  
   Scripts and tools for testing outliers.


## 3. Setup environment

    # create and activate virtual python environment
    conda create -n dna python=3.8
    conda activate dna
    
    # install required packages
    python3 -m pip install -r requirements.txt

## 7. Citation

If you have any question regarding our paper or codes, please feel free to start an issue.

If you use GERM in your work, please kindly cite our paper:

**DNABERT-2**

```
@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

**DNABERT**

```
@article{ji2021dnabert,
    author = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
    title = "{DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome}",
    journal = {Bioinformatics},
    volume = {37},
    number = {15},
    pages = {2112-2120},
    year = {2021},
    month = {02},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab083},
    url = {https://doi.org/10.1093/bioinformatics/btab083},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/37/15/2112/50578892/btab083.pdf},
}
```