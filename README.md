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
The `attacks` directory contains code for evaluating our attack (`fit`), IHOP, and the Naive Bayes baseline. The `data` directory contains preprocessed training and testing datasets used in our evaluation. `model_weights` stores our trained language models for each evaluated use case. `scripts` contains bash and Python scripts to run all attacks and plot the results shown in our paper.

For the full contents of `data` and `model_weights` needed for evaluation, please download our artifact from Zenodo with the DOI [10.5281/zenodo.15602651](https://doi.org/10.5281/zenodo.15602651).

### Datasets
```
data/
├── dlrm/
│   ├── all.csv
│   ├── ihop_dlrm.pkl
│   ├── ihop_dlrm_1_1.pkl
│   ├── sgx.csv
│   ├── test.csv
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

For DLRM, `all.csv` contains columns `page_i` and `idx_i` for `i=1..26`, where `page_i` is the `i`th page observed to be accessed for an inference request in a Nitro Enclave and `idx_i` is the ground-truth index of the embedding table entry that was accessed. `test.csv` is similarly structured, containing the access sequences used for evaluation, and `sgx.csv` contains the same information but for observed page accesses collected from an SGX enclave. The subdirectory `error_traces` contains datasets of the same format, but with injected errors in observed page accesses, for our error sensitivity plots (Figure 9 in the paper). `times.csv` contains the data of request durations needed to reproduce our latency overhead plots (Figure 10). Also included are one or more `.pkl` files containing the results of the [IHOP attack](#running-ihop) on this data.

LLM and HNSW have similar subdirectory structures to DLRM. `all.csv` and `sgx.csv` contain two columns: `encseq` contains a space-separated list of observed page accesses, and `seq` contains a space-separated list of the ground-truth accesses over objects -- embedding table entries for LLM and nodes for HNSW. Instead of using a separate `test.csv`, the models of LLM and HNSW evaluate on the test split of `all.csv`. 

## Tested Configurations
We have run our experiments on the following system configurations:

### CPU setup

**Linux**
- Hardware: AMD EPYC 7302P 16-Core Processor
- OS: Ubuntu 22.04.2 LTS

**Mac OS**
- Hardware: Apple M2 Pro 12-Core Processor
- OS: macOS Sonoma 14.6.1

### GPU setup (for our attack)
- Hardware: NVIDIA GeForce RTX 4090
- OS: Ubuntu 24.04.1 LTS

Given these setups, the following table summarizes the main results in our paper and the estimated time needed to run all the experiments for those results sequentially.

The times for Attack Efficacy are itemized by attack: our attack (FIT), IHOP, and Naive Bayes (NB). The IHOP times are based on prior runs with the AMD setup and serve as an upper bound; we have observed a speedup of up to 2x on the Apple M2 Pro for some experiments.

| Experiment Name / Section | Related Figures  | Estimated Time on GPU <br> (FIT + IHOP + NB) | Estimated Time on CPU <br> (FIT + IHOP + NB)|
|---------------------------|------------------|------------------------|------------------------|
| Attack Efficacy           | Fig. 7, Fig. 8    | 3h + 56h + 1.5h = 60.5h     |  68h + 56h + 1.5h = 125.5h   |
| Practical Considerations  | Fig. 9, Fig. 10   |     6.5h     |    115.5h    |


## Quickstart
We recommend replicating our execution environment using conda. Instructions for installing Miniconda can be found [here](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions).
To start, create a new conda environment with Python 3.12:
```
conda create -n fit -y python=3.12  
conda activate fit
```

Next, install the required dependencies:
```
 pip install -r requirements.txt
```
Then, you can run a basic test of all attacks and reproduce our results using the provided scripts.

## Basic Test
The following script runs FIT, IHOP, and the Naive Bayes baseline on a single test sample from the LLM use case:
```
bash scripts/run_basic_test.sh
```
We expect the tests to take around 5 minutes in total. Each attack should populate the `data/llm/eval` directory with a results file -- `llm_nitro.csv`, `nb_llm.csv`, and `ihop_llm.pkl`, respectively. These files will be overwritten when running our experiments, detailed below.

## Reproducing Results

### Attack Efficacy
The Attack Efficacy experiments compare the accuracies of our attack, IHOP, and a Naive Bayes classifier in predicting ground-truth access sequences across application-level objects.

A breakdown of estimated times is shown below. Please note that for DLRM, LLM, and HNSW, the estimated times for our attack (FIT) are doubled as it is evaluated on page traces from Nitro *and* SGX Enclaves. The remaining experiments use only Nitro page traces.

| Use Case          | FIT (GPU)  | FIT (CPU) | IHOP |  Naive Bayes |
|-------------------|----------|------------|-------|-------------|
| DLRM              |   1h 20m |    44h     |  25h     |     12m    |
| LLM               | 1h 10m   |     2h     |    5m    |  1h 15m  |
| HNSW              | 30s   |     10m     |    6h    |  10s  |
| DLRM 1-1 mapping  | 40m   |    22h     |   25h    | 4m   |


#### Running our attack
The evaluation script loads our BERT-based language model from `model_weights` and runs inference on the test datasets. The following script runs our attack, taking as arguments the number of test samples to use for DLRM, LLM, and HNSW, respectively:
```
bash scripts/run_fit.sh 100000 50000 2600
```
While we recommend using a GPU for these experiments, you can also select the CPU-only option and run specific experiments with fewer test samples:
```
bash scripts/run_fit.sh 1000 5000 2600 --use-cpu
```
This will significantly reduce the execution time on the CPU, and the results should still exhibit similar trends to our work.

#### Running IHOP
Due to the long running times of IHOP on some experiments, we provide results from previous runs of the attack in the directories corresponding to each use case in `data` as `ihop_dlrm.pkl`, `ihop_dlrm_1_1.pkl`, `ihop_llm.pkl`, and `ihop_hnsw.pkl`. Reproducing this attack will save the results to the `eval` directory under each use case, and our plotting script will use the new results if they exist.

If cloning from GitHub, please run the following command from the root directory to fetch the IHOP code and apply the necessary patches for our use cases. These steps should be skipped if the artifact is downloaded from Zenodo.
```
git submodule init
git submodule update
cd attacks
mv page_experiment.py fit_use_cases.patch USENIX22-ihop-code
cd USENIX22-ihop-code
git apply fit_use_cases.patch
```

The commands to run each experiment are found in the following script:
```
bash scripts/run_ihop.sh
```
#### Running Naive Bayes
The commands to run each experiment are found in the following script:
```
bash scripts/run_nb.sh
```
#### Plotting
After running our and compared attacks (IHOP optional), use the following Python script to reproduce Figures 7 and 8 in the paper:
```
python3 scripts/plot.py --file-ext .png --data-dir data --plot-dir plots --fig 7
python3 scripts/plot.py --file-ext .png --data-dir data --plot-dir plots --fig 8
```
### Practical Considerations
This section focuses on reproducing the sensitivity analysis of our attack for various error rates, requiring five runs for each evaluated use case. The breakdown of estimated times is shown below.

| Use Case          | FIT (GPU)  | FIT (CPU) | 
|-------------------|----------|------------|
| DLRM              |   3h 20m |    110h     | 
| LLM               |    3h    |     5h     |
| HNSW              |     3m    |     25m     |

The following script runs all experiments sequentially:
```
bash scripts/run_fit_sensitivity.sh
```
For CPU-only platforms, the number of test samples can be set in the same manner as for `run_fit.sh`:
```
bash scripts/run_fit_sensitivity.sh 1000 5000 2600 --use-cpu
```
You can then reproduce Figures 9 and 10:
```
python3 scripts/plot.py --file-ext .png --data-dir data --plot-dir plots --fig 9
python3 scripts/plot.py --file-ext .png --data-dir data --plot-dir plots --fig 10
```
