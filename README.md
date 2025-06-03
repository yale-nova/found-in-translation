# Found In Translation
This repository contains the code and data used for evaluations in the paper "Found In Translation: A Generative Language Modeling Approach to Memory Access Pattern Attacks".

## Overview
```
.  
 ├── attacks/  
 ├── data/  
 ├── model_weights/  
 ├── plots/  
 └── scripts/
```
The `attacks` directory contains code for evaluating our attack (`language_model`), IHOP, and the Naive Bayes baseline. The `data` directory contains preprocessed training and testing datasets used in our evaluation. `model_weights` stores our trained language models for each evaluated use case. `scripts` contains bash and python scripts to run all attacks and plot the results shown in our paper.

### Datasets
```
data/
├── dlrm/
│   ├── all.csv
│   ├── ihop_dlrm.pkl
│   ├── ihop_dlrm_1_1.pkl
│   ├── sgx.csv
│   ├── times.csv
│   ├── error_traces/
│   │   ├── err1.csv
│   │   ├── err3.csv
│   │   ├── err5.csv
│   │   ├── err7.csv
│   │   └── err10.csv
│   └── eval/
├── hnsw/
└── llm/
```

The `data` directory contains a subdirectory for each use case evaluated in our paper: `dlrm`, `llm`, and `hnsw`.

For DLRM, `all.csv` contains columns `page_i` and `idx_i` for `i=1..26`, where `page_i` is the `i`th page observed to be accessed for an inference request in a Nitro Enclave and `idx_i` is the ground-truth index of the embedding table entry that was accessed. `test.csv` is similarly structured, containing the access sequences used for the evaluation, and `sgx.csv` contains the same but observed page accesses collected from an SGX enclave. The subdirectory `error_traces` contains datasets of the same format but with injected errors in observed page accesses (for our error sensitivity plots -- Figure 9 in the paper). `times.csv` contains the data of request durations needed to reproduce our latency overhead plots (Figure 10). Also included are one or more `.pkl` files containing the results of the [IHOP attack](#ihop) on this data.

LLM and HNSW have similar subdirectory structures to DLRM. `all.csv` and `sgx.csv` contain two columns: `encseq` contains a space-separated list of observed page accesses, and `seq` contains a space-separated list of the ground-truth accesses over objects -- embedding table entries for LLM and nodes for HNSW. Instead of using a separate `test.csv`, the models of LLM and HNSW evaluate on the test split of `all.csv`. 

## Quickstart
We recommend replicating our execution environment using conda. Instructions to install Miniconda are found [here](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions).
To start, create a new conda environment with Python 3.12:
```
conda create -n fit -y python=3.12  
conda activate fit
```

Next, install the required dependencies:
```
 pip install -r requirements.txt
```

Then, you can reproduce our plotted results using the provided scripts.
## Running our attack
The evaluation script loads our BERT-based language model from `model_weights` and runs inference on the test datasets. The following script runs all experiments involving our attack:
```
bash scripts/run.sh
```
We recommend using a GPU for these experiments. If only CPU is available, you can run certain experiments with fewer test samples by editing the following variables in `run.sh`:
```
DLRM_N_TEST_SAMPLES=1000
LLM_N_TEST_SAMPLES=5000
```
This will greatly reduce the execution time and the results should still exhibit similar trends to our work.

## Running compared attacks
### IHOP
With our CPU setup, running the IHOP attack took around 25 hours on DLRM data, 6 hours on HNSW data, and 5 minutes on LLM data.

For convenience, we provide results from our runs of the IHOP attack in the directories corresponding to each use case in `data` as `ihop_dlrm.pkl`, `ihop_llm.pkl`, and `ihop_hnsw.pkl`. Reproducing this attack will save the results to the `eval` directory under each use case, and our plotting script will use the new results if they exist.

To set up the IHOP code for our use cases, run the following from the root directory:
```
git submodule init
git submodule update
cd attacks
cp page_experiment.py fit_use_cases.patch USENIX22-ihop-code
cd USENIX22-ihop-code
git apply fit_use_cases.patch
```

The commands to run each experiment can be found in the following script:
```
bash scripts/run_ihop.sh
```
### Naive Bayes
With our CPU setup, running the Naive Bayes attack took 1 hour in total. The following script runs all experiments sequentially:
```
bash scripts/run_nb.sh
```
## Reproducing plots
After running our attack (and optionally the compared attacks), the following python script will reproduce the plots used in our paper:
```
python3 scripts/plot.py
```
