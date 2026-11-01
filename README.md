# op3_signatures

Perturbation prediction benchmarks on the NeurIPS-2023 dataset, based on [task_perturbation_prediction](https://github.com/openproblems-bio/task_perturbation_prediction/).

## Setup

Create the pipeline environments:

```bash
mamba env create -f venvs/configs/fetching_data/env.yaml -p ./venvs/venvs/fetching_data -y
mamba env create -f venvs/configs/nn_retraining_with_pseudolabels/env.yaml -p ./venvs/venvs/nn_retraining_with_pseudolabels -y
mamba env create -f venvs/configs/metrics/env.yaml -p ./venvs/venvs/metrics -y
```

For notebooks **02–04**:

```bash
mamba env create -f venvs/configs/notebook/env.yaml -p ~/notebook_venv -y
```

## Running the benchmark

Run commands from the **project root**. Pipelines require SLURM.

### Step 1: Download datasets

Downloads `de_train.h5ad`, `de_test.h5ad` and `id_map.csv` to `data/benchmark/resources/datasets/neurips-2023-data/`.

```bash
./run_pipelines/run_fetching_data.sh
```

After downloading, copy `neurips-2023-data/` into `data_<coverage>/benchmark/resources/datasets/` for both `full_coverage` and `coverage_match`.

### Step 2: Run the prediction pipeline

First run [notebooks 01, 02.1 and 02.2](#jupyter-notebooks) to prepare the data and embeddings.

```bash
./run_stability_pipeline_full_coverage.sh
./run_stability_pipeline_coverage_match.sh
```

Defaults: epoch 20, fixed/trainable embeddings, ten seeds (`0–4, 10–14`), including ablations and step-matched runs.

Inputs and results use `data_<coverage>/`, where `<coverage>` is `full_coverage` or `coverage_match` (see [Data structure](#data-structure)).

#### Individual runs

Set `RUN_TAG` to select coverage:

```bash
RUN_TAG=full_coverage ./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh \
    -D <subsample|original> \
    -e <embedding_type> \
    [-l <concat|fixed|trainable>] \
    [-d] \
    [-f <embedding_file>] \
    [-S <seeds>]
```

| Flag | Values | Default | Purpose |
|---|---|---|---|
| `-D` | `subsample`, `original` | `subsample` | Dataset |
| `-e` | `lpm`, `fp`, `chemb1`, `chemb2`, `molformer`, `pca_logfc`, `pca_t`, `none` | `none` | Embedding type |
| `-l` | `concat`, `fixed`, `trainable` | `concat` | Embedding mode |
| `-d` | flag | off | Dense projection before concatenation |
| `-f` | filename or path | `op3_emb.pkl` | Embedding pickle; tag inferred from `op3_emb_<tag>.pkl` |
| `-S` | comma-separated seeds | `0,1,2,3,4,10,11,12,13,14` | One job per seed |

**Embedding modes:**

- `concat`: append external embeddings; optionally project them with `-d`.
- `fixed`: initialize compound embeddings from external features and freeze them.
- `trainable`: initialize the same way, then fine-tune them.

### Step 3: Compute stability metrics

After predictions finish, compute RMSE, MAE, Pearson, Spearman and cosine metrics:

```bash
./run_pipelines/run_metrics_stability.sh -r full_coverage
./run_pipelines/run_metrics_stability.sh -r coverage_match
```

## Jupyter notebooks

Use `notebooks/` as the working directory. Run **01–02 before Step 2**, and **03–04 after Step 3**.

For **01**, follow the [LPM repository setup](https://github.com/theislab/chem-perturbridge_lpm) to install `perturb_lib`, and select the LPM kernel.

For **02–04**, select `~/notebook_venv`:

```bash
mamba activate ~/notebook_venv
jupyter lab
```

| Notebook | Purpose |
|---|---|
| [01](notebooks/01_get_precomputed_LPM_based_embeddings.ipynb) | Export LPM embeddings to `files/`. |
| [02.1](notebooks/02.1_match_LPM_based_embeddings_with_OP3_full_coverage.ipynb) | Prepare full-coverage data and embeddings. |
| [02.2](notebooks/02.2_match_LPM_based_embeddings_with_OP3_coverage_match.ipynb) | Prepare coverage-matched data and embeddings. |
| [03.1](notebooks/03.1_plotting_model_results_full_coverage.ipynb) | Summarize full-coverage metrics. |
| [03.2](notebooks/03.2_plotting_model_results_full_coverage.ipynb) | Summarize coverage-matched metrics. |
| [04](notebooks/04_bootstrap.ipynb) | Bootstrap main and ablation results. |

External inputs: `files/df_pubchem_op3.csv` (**02.1/02.2**) and `~/Chem-PerturBridge_analysis/scripts/cluster_bootstrap_ci.py` (**04**).

## Project structure

```text
├── data_full_coverage/           # Full-coverage inputs and results
├── data_coverage_match/          # Coverage-matched inputs and results
├── files/                       # LPM embeddings and PubChem mapping
├── notebooks/
├── results/                     # Analysis CSVs
├── pipelines/                   # SLURM jobs
├── run_pipelines/               # Launchers
├── run_stability_pipeline_full_coverage.sh
├── run_stability_pipeline_coverage_match.sh
├── src/                         # Data, methods, metrics and utilities
└── venvs/
    ├── configs/                 # Environment definitions
    └── venvs/                   # Installed environments
```

## Logs

Downloads: `logs/fetching_data/`. Predictions and metrics:

```text
logs_<coverage>/
├── metrics_stability/
└── methods/<pipeline>/
```

- Launcher: `run_<name>.PID<pid>.out` / `.err`
- SLURM job: `<name>.<jobid>.out` / `.err`

## Data structure

```text
data_<coverage>/benchmark/
├── resources/datasets/
│   ├── neurips-2023-data/             # Full dataset (-D original)
│   │   ├── de_train.h5ad
│   │   ├── de_test.h5ad
│   │   └── id_map.csv
│   └── neurips-2023-data-subsample/   # -D subsample
│       ├── de_train.h5ad
│       ├── de_test.h5ad
│       ├── id_map.csv
│       └── op3_emb_*.pkl
└── results/
    ├── methods/<pipeline>/stability/
    │   └── predictions_seed_<n>.h5ad
    └── metrics/methods/<pipeline>/stability/seed_<n>/
        ├── mean_rowwise_error.h5ad
        └── mean_rowwise_correlation.h5ad
```
