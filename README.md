# op3_signatures

Benchmarking of perturbation prediction methods on the NeurIPS-2023 single-cell perturbation dataset. The code (methods and metrics) is based on the following repo: [**task_perturbation_prediction**](https://github.com/openproblems-bio/task_perturbation_prediction/).

## Setup

Create all conda environments before running any pipeline:

```bash
mamba env create -f venvs/configs/fetching_data/env.yaml   -p ./venvs/venvs/fetching_data -y
mamba env create -f venvs/configs/pyboost/env.yaml         -p ./venvs/venvs/pyboost -y
mamba env create -f venvs/configs/jn_ap_op2/env.yaml      -p ./venvs/venvs/jn_ap_op2 -y
mamba env create -f venvs/configs/nn_retraining_with_pseudolabels/env.yaml -p ./venvs/venvs/nn_retraining_with_pseudolabels -y
mamba env create -f venvs/configs/scape/env.yaml          -p ./venvs/venvs/scape -y
mamba env create -f venvs/configs/transformer_ensemble/env.yaml -p ./venvs/venvs/transformer_ensemble -y
mamba env create -f venvs/configs/lgc_ensemble/env.yaml  -p ./venvs/venvs/lgc_ensemble -y
mamba env create -f venvs/configs/control_methods/env.yaml -p ./venvs/venvs/control_methods -y
mamba env create -f venvs/configs/metrics/env.yaml       -p ./venvs/venvs/metrics -y
```

## Running the benchmark

All commands must be run from the **project root**.

### Step 1 — Download datasets

Downloads `de_train.h5ad`, `de_test.h5ad`, and `id_map.csv` from S3 to
`data/benchmark/resources/datasets/neurips-2023-data/`.

```bash
./run_pipelines/run_fetching_data.sh
```

### Step 2a — Run a prediction method

```bash
./run_pipelines/run_methods.sh -m <method>
```

Available methods:

| Method | Description |
|--------|-------------|
| `pyboost` | Py-boost gradient boosting |
| `jn_ap_op2` | Deep learning ensemble (OP2-based) |
| `nn_retraining_with_pseudolabels` | Neural network with pseudolabel retraining |
| `scape` | SCAPE TF-based model |
| `transformer_ensemble` | Transformer ensemble |
| `lgc_ensemble_direct` | LGC LSTM/GRU/Conv ensemble (single-job) |

Predictions are saved to `data/benchmark/results/methods/<method>/predictions.h5ad`.

### Step 2b — Run control / baseline methods

Run **all** control methods in one go:

```bash
./run_pipelines/run_control_methods.sh
# or explicitly:
./run_pipelines/run_control_methods.sh -a
```

Run a **single** control method:

```bash
./run_pipelines/run_control_methods.sh -m <method>
```

Available control methods:

| Method | Description |
|--------|-------------|
| `zeros` | All-zeros prediction |
| `mean_across_celltypes` | Per-gene mean across all compounds per cell type |
| `mean_across_compounds` | Per-gene mean across all cell types per compound |
| `mean_outcome` | Global per-gene mean across all samples |
| `ground_truth` | Returns test labels (upper bound) |
| `sample` | Randomly samples training values |

### Step 3 — Compute metrics

Computes mean rowwise error (RMSE, MAE) and mean rowwise correlation
(Pearson, Spearman, Cosine) between predictions and `de_test.h5ad`.

```bash
./run_pipelines/run_metrics.sh
```

Results are saved to `data/benchmark/results/metrics/methods/<method>/` and `data/benchmark/results/metrics/control_methods/<method>/`.

## Project structure

```
├── data/                              # Downloaded datasets and results (not tracked by git)
├── pipelines/                         # Pipeline scripts (sbatch jobs)
│   ├── common/                        # Shared pipelines (data fetching, metrics)
│   ├── control_methods/               # Single dispatcher script for all control methods
│   ├── pyboost/
│   ├── jn_ap_op2/
│   ├── nn_retraining_with_pseudolabels/
│   ├── scape/
│   ├── transformer_ensemble/
│   └── lgc_ensemble_direct/
├── run_pipelines/                     # Thin launchers that call pipelines/ scripts
├── src/
│   ├── data/                          # Dataset downloading utilities
│   ├── control_methods/               # Baseline / control method implementations
│   │   ├── zeros/
│   │   ├── mean_across_celltypes/
│   │   ├── mean_across_compounds/
│   │   ├── mean_outcome/
│   │   ├── ground_truth/
│   │   └── sample/
│   ├── methods/                       # Prediction method implementations
│   │   ├── pyboost/
│   │   ├── jn_ap_op2/
│   │   ├── nn_retraining_with_pseudolabels/
│   │   ├── scape/
│   │   ├── transformer_ensemble/
│   │   ├── lgc_ensemble_direct/       # Single-job LGC ensemble entry point
│   │   ├── lgc_ensemble_prepare/      # LGC data preparation step
│   │   ├── lgc_ensemble_train/        # LGC per-fold training step
│   │   ├── lgc_ensemble_predict/      # LGC prediction step
│   │   └── lgc_ensemble_helpers/      # Shared LGC helper modules
│   ├── metrics/                       # Evaluation metric scripts (R)
│   └── utils/                         # Shared utilities
└── venvs/
    ├── configs/                       # Conda environment definitions (env.yaml per method)
    └── venvs/                         # Installed environments (not tracked by git)
```

## Logs

All logs are written under `logs/`:

```
logs/
├── fetching_data/                      # Data download logs
├── metrics/                            # Metrics pipeline logs
├── methods/                            # Prediction method logs
│   ├── pyboost/
│   ├── jn_ap_op2/
│   ├── nn_retraining_with_pseudolabels/
│   ├── scape/
│   ├── transformer_ensemble/
│   └── lgc_ensemble_direct/
│   ├── mean_outcome/
│   ├── ground_truth/
│   └── sample/
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
        ├── methods/                    # Prediction method outputs
        │   ├── pyboost/
        │   ├── jn_ap_op2/
        │   ├── nn_retraining_with_pseudolabels/
        │   ├── scape/
        │   ├── transformer_ensemble/
        │   └── lgc_ensemble_direct/
        │       └── predictions.h5ad   # (each method dir)
        ├── control_methods/           # Control method outputs
        │   ├── zeros/
        │   ├── mean_across_celltypes/
        │   ├── mean_across_compounds/
        │   ├── mean_outcome/
        │   ├── ground_truth/
        │   └── sample/
        │       └── predictions.h5ad   # (each control method dir)
        └── metrics/                   # Metric outputs (per method / control)
            ├── methods/
            │   └── <method>/
            │       ├── mean_rowwise_error.h5ad
            │       └── mean_rowwise_correlation.h5ad
            └── control_methods/
                └── <method>/
                    ├── mean_rowwise_error.h5ad
                    └── mean_rowwise_correlation.h5ad
```
