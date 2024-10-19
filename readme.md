# scZGCL
## Summary:
A pytorch implementation of the paper "Deep single-cell RNA-seq data clustering with ZINB-based graph contrastive learning".

## Introduction of the files:
### 1. Directories
The `data` directory contains a zip file `processed_data` with four example datasets to reproduce the results of experiments. Before reproducing, you need to extract all the files in the zip `processed data` to `data` folder.

The `layers` directory contains implementation of the layers of our method.

The `logs` directory contains logs for recording experiment results.

The `models` directory contains two main files. `scZGCL.py` is the implementation of scZGCL's model. `scZGCL_MSE.py` is the implementation of scZGCL without ZINB, which is replaced by MSE.

The `scripts` directory contains two files. `script_for_Diaphragm_Smart-seq2.sh` is the example script for reproducing the result of the smaple dataset Diaphragm(Smart-seq2). `scripts_for_all_datasets.sh` is the script for reproducing the results of all datasets.

The `src` directory contains files for defining scripts and reading datasets.

The `results` directory contains the detailed results in Parameter Analysis and Ablation Study.

### 2. Single files:
`requirements.txt` contains the environmental requirement for reproducing our results. 

`main.py`, `embedder.py` are codes for implementing scZGCL.

`t-SNE_drawing.py` is the code for 2-D clustering visualization.

## Quick start:
**First**, configure the Python environment on Linux system according to `requirements.txt`. Just for reference, we run our experiments on an Ubuntu server with an NVIDIA RTX 4090 GPU and 24GB of memory.

`pip install -r requirements.txt`

**Second**, unzip the `processed_data.zip` to `data` folder.

**Third**, reproduce experiments on various datasets. Use the example datasets from `processed_data.zip` directory or download more datasets from [scziDesk](https://github.com/xuebaliang/scziDesk). Take the dataset **Diaphragm_Smart-seq2** as an example:

1. Add executable permissions to `.sh` file:

`chmod +x script_for_Diaphragm_Smart-seq2.sh`

2. Run `.sh` file in the `scZGCL` directory:

`./script_for_Diaphragm_Smart-seq2.sh`

3. After running `.sh`, you will see two txt files which are named `Diaphragm_Smart-seq2_zinb_label.txt` and `Diaphragm_Smart-seq2_zinb_latent.txt`. See results in `Diaphragm_Smart-seq2.log` in the `logs` directory. For 2-D visualization by t-SNE, run `t-SNE_drawing.py` in the `scZGCL` directory.

For other datasets, the procedure is the same. The corresponding script for each dataset is in `scripts_for_all_datasets.sh`.
