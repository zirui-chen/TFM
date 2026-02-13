# TFM: Beyond Unstructured Data: A Topological Foundation Model for Knowledge Graphs

[![pytorch](https://img.shields.io/badge/PyTorch_2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.4+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

PyG implementation of the **Topological Foundation Model (TFM)** 



## Overview

While large language models (LLMs) excel on unstructured data, building a general-purpose foundation model for knowledge graphs (KGs) that transfers robustly across domains remains challenging. TFM bridges this gap by shifting the learning objective from memorizing unbounded atomic symbols to reasoning over a finite topological basis.

TFM relies on three core components to resolve the epistemological mismatch between conventional KG embeddings and LLM-based serialization:

* **Topologizer**: Maps relation patterns to a finite, complete set of topological primitives.
* **Contextualized Relational Parameterization (CRP)**: Constructs relation representations on the fly from local structure and the query context.
* **Universal Reasoning Layer**: Executes topology-driven instructions without entity/relation-specific embeddings.

By being isomorphism-invariant by construction, TFM preserves structural fidelity while enabling strong zero-shot reasoning across disjoint knowledge domains. Empirical evaluations across 34 benchmarks show that TFM 0-shot achieves a 57.6 Mean Reciprocal Rank (MRR), outperforming specialized SOTA models.



## Installation

You may install the dependencies via either conda or pip. TFM is implemented with Python 3.9, PyTorch 2.1, and PyG 2.4 (CUDA 11.8 or later when running on GPUs). 

### From Conda

```bash
conda install pytorch=2.1.0 pytorch-cuda=11.8 cudatoolkit=11.8 pytorch-scatter=2.1.2 pyg=2.4.0 -c pytorch -c nvidia -c pyg -c conda-forge
conda install ninja easydict pyyaml -c conda-forge
```

### From Pip

```bash
pip install torch==2.1.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 -f [https://data.pyg.org/whl/torch-2.1.0+cu118.html](https://data.pyg.org/whl/torch-2.1.0+cu118.html)
pip install ninja easydict pyyaml
```



## Checkpoints

Pre-trained TFM checkpoints are provided in the `/ckpts` folder for reproducibility. These can be used immediately for zero-shot inference on any graph or as a backbone for fine-tuning.

*(Note for reviewers: Links to external model hubs have been temporarily removed to maintain anonymity. Checkpoints are included in the supplementary code materials.)*



## Run Inference and Fine-tuning

The `/scripts` folder contains executable files for training and evaluation:

* `run.py` - run an experiment on a single dataset
* `run_many.py` - run experiments on several datasets sequentially
* `pretrain.py` - a script for pre-training TFM on several graphs

### Run a single experiment

The `run.py` command requires the following arguments:

* `-c <yaml config>`: a path to the yaml config
* `--dataset`: dataset name (from the list of datasets)
* `--epochs`: number of epochs to train. `--epochs 0` means running zero-shot inference.
* `--bpe`: batches per epoch.
* `--gpus`: number of gpu devices, set to `--gpus null` when running on CPUs.
* `--ckpt`: full path to the TFM checkpoint to use.

An example command for zero-shot inference on an inductive dataset:

```bash
python script/run.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v1 --epochs 0 --bpe null --gpus [0] --ckpt /path/to/tfm/ckpts/tfm_pretrained.pth
```

### Pretraining

Run the pre-training script `pretrain.py` with the provided configs. Pre-training can be computationally heavy; the default experiments were conducted on 2 NVIDIA H100 GPUs. You might need to decrease the batch size for smaller GPU RAM.