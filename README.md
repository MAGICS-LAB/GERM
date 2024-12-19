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


## Setup environment

    # create and activate virtual python environment
    conda create -n dna python=3.8
    conda activate dna
    
    # install required packages
    python3 -m pip install -r requirements.txt

## Citation

If you have any question regarding our paper or codes, please feel free to start an issue.

If you use GERM in your work, please kindly cite our paper and code:
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


## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- AutoTimes (https://github.com/thuml/AutoTimes)
- ST-MoE-BERT (https://github.com/he-h/ST-MoE-BERT)





