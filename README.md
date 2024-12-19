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
@misc{luo2024germ,
      title={Making Genomic Foundation Models more Foundational Requires Outlier Removal: A Case Study on DNABERT-2}, 
      author={Haozheng Luo and Chenghao Qiu and Maojiang Su and Zhihan Zhou and Jerry Yao-Chieh Hu and Zoe Mehta and Guo Ye and Han Liu},
      year={2024},
      url={https://github.com/MAGICS-LAB/GERM}
}
```


## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.
- DNABERT-2 (https://github.com/MAGICS-LAB/DNABERT_2)
- HyenaDNA (https://github.com/HazyResearch/hyena-dna)
- SmoothQuant (https://github.com/mit-han-lab/smoothquant)
- OutEffHop (https://github.com/MAGICS-LAB/OutEffHop)
- OmniQuant (https://github.com/OpenGVLab/OmniQuant)
- Outlier Suppression (https://github.com/wimh966/outlier_suppression)
- LoftQ (https://github.com/yxli2123/LoftQ)
- Nucleotide Transformers (https://github.com/instadeepai/nucleotide-transformer)





