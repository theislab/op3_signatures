# op3_signatures

Benchmarking of perturbation prediction methods on the NeurIPS-2023 single-cell perturbation dataset. The code (methods and metrics) is based on the following repo: [**task_perturbation_prediction**](https://github.com/openproblems-bio/task_perturbation_prediction/).

## Setup

Create all conda environments before running any pipeline:

```bash
mamba env create -f venvs/configs/fetching_data/env.yaml -p ./venvs/venvs/fetching_data
mamba env create -f venvs/configs/pyboost/env.yaml       -p ./venvs/venvs/pyboost
mamba env create -f venvs/configs/metrics/env.yaml       -p ./venvs/venvs/metrics
```

## Running the benchmark

All commands must be run from the **project root**.

### Step 1 — Download datasets

Downloads `de_train.h5ad`, `de_test.h5ad`, and `id_map.csv` from S3 to
`data/benchmark/resources/datasets/neurips-2023-data/`.

```bash
./run_pipelines/run_fetching_data.sh
```

### Step 2 — Run a prediction method

```bash
./run_pipelines/run_methods.sh -m pyboost
```

Available methods: `pyboost`

Predictions are saved to `data/benchmark/results/<method>/predictions.h5ad`.

### Step 3 — Compute metrics

Computes mean rowwise error (RMSE, MAE) and mean rowwise correlation
(Pearson, Spearman, Cosine) between predictions and `de_test.h5ad`.

```bash
./run_pipelines/run_metrics.sh
```

Results are saved to `data/benchmark/results/metrics/`.

## Project structure

```
├── data/                  # Downloaded datasets and results (not tracked by git)
├── pipelines/             # Pipeline scripts (sbatch jobs)
│   ├── common/            # Shared pipelines (data fetching, metrics)
│   ├── pyboost/           # Pyboost pipeline
│   └── .../               # Method-specific pipelines
├── run_pipelines/         # Thin launchers that call pipelines/ scripts
├── src/
│   ├── data/              # Dataset downloading utilities
│   ├── methods/           # Prediction method implementations
│   ├── metrics/           # Evaluation metric scripts (R)
│   └── utils/             # Shared utilities
└── venvs/
    ├── configs/           # Conda environment definitions (env.yaml per method)
    └── venvs/             # Installed environments (not tracked by git)
```

## Logs

All logs are written to `logs/<pipeline>/`:

```
logs/
├── fetching_data/    # Data download logs
├── pyboost/          # Method-specific logs
└── metrics/          # Metrics computation logs
```

Each pipeline produces two files per run:
- `run_<name>.PID<pid>.out` / `.err` — launcher output
- `<name>.<jobid>.out` / `.err` — sbatch job output

## Data structure

```
data/
└── benchmark/
    ├── resources/
    │   └── datasets/
    │       └── neurips-2023-data/
    │           ├── de_train.h5ad   # Training perturbation data
    │           ├── de_test.h5ad    # Test perturbation data (ground truth)
    │           └── id_map.csv      # Cell type / compound mapping for test set
    └── results/
        ├── pyboost/
        │   └── predictions.h5ad   # Pyboost predictions
        └── metrics/
            ├── mean_rowwise_error.h5ad        # RMSE and MAE
            └── mean_rowwise_correlation.h5ad  # Pearson, Spearman, Cosine
```
