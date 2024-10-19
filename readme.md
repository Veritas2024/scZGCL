# scZGCL
## Summary:
A pytorch implementation of the paper "Deep single-cell RNA-seq data clustering with ZINB-based graph contrastive learning".

## Introduction of the files:
### 1. Directories
The `data` directory contains a zip file `processed_data` with twelve datasets to reproduce the results of experiments. Before reproducing, you need to extract all the files in the zip `processed data` to `data` folder.

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
First, configure the Python environment on Linux system according to `requirements.txt`.

`pip install -r requirements.txt`

Second, unzip the `processed_data.zip` to `data` folder.

Third, reproduce experiments on various datasets. Use the example datasets from `processed_data.zip` directory or download more datasets from （网址）. Take the dataset **Diaphragm_Smart-seq2** as an example:

1. Add executable permissions to `.sh` file:

`chmod +x script_for_Diaphragm_Smart-seq2.sh`

2. Run `.sh` file in the `scZGCL` directory:

`./script_for_Diaphragm_Smart-seq2.sh`

3. After running `.sh`, you can see result in `Diaphragm_Smart-seq2.log` in the `logs` directory. For visualization by , run t-SNE_drawing.py

Third, run **script_for_Diaphragm_Smart-seq2.sh** to reproduce result on **Diaphragm(Smart-seq2)** dataset and view the results through the corresponding log file.

or you can download all datasets from （网址）

Forth, after running script `script_for_Diaphragm_Smart-seq2.sh`, you will see two txt files which are named `Diaphragm_Smart-seq2_zinb_label.txt` and `Diaphragm_Smart-seq2_zinb_latent.txt`. Run `t-SNE_drawing.py` to see the 2-D clustering visualization by t-SNE.
