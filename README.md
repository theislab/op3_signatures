# op3_signatures

Benchmarking of perturbation prediction methods on the NeurIPS-2023 single-cell perturbation dataset. The code (methods and metrics) is based on the following repo: [**task_perturbation_prediction**](https://github.com/openproblems-bio/task_perturbation_prediction/).

## Setup

Create all conda environments before running any pipeline:

```bash
mamba env create -f venvs/configs/fetching_data/env.yaml   -p ./venvs/venvs/fetching_data -y
mamba env create -f venvs/configs/nn_retraining_with_pseudolabels/env.yaml -p ./venvs/venvs/nn_retraining_with_pseudolabels -y
mamba env create -f venvs/configs/metrics/env.yaml       -p ./venvs/venvs/metrics -y
```

For the Jupyter notebooks under `notebooks/` (see [Jupyter notebooks](#jupyter-notebooks) below), create the analysis env at `~/notebook_venv`:

```bash
mamba env create -f venvs/configs/notebook/env.yaml -p ~/notebook_venv -y
```

## Running the benchmark

All commands must be run from the **project root**.

### Step 1 — Download datasets

Downloads `de_train.h5ad`, `de_test.h5ad`, and `id_map.csv` from S3 to
`data/benchmark/resources/datasets/neurips-2023-data/`.

```bash
./run_pipelines/run_fetching_data.sh
```

### Step 2 — Run the prediction pipeline

> **Prerequisite:** the stability pipeline reads `op3_emb_*.pkl` from
> `data/benchmark/resources/datasets/neurips-2023-data-subsample/`. Generate
> them first by running [notebooks 01 and 02](#jupyter-notebooks) (or copy
> existing pickles from `data_archive/`).

Runs `nn_retraining_with_pseudolabels_mol_emb_learning_missed` as a multi-seed
stability sweep across LPM/FP embedding versions. The top-level script
orchestrates all per-epoch and per-embedding-mode configurations:

```bash
./run_stability_pipeline.sh                # sweep default EPOCHS
./run_stability_pipeline.sh 5 10 15        # sweep just these epochs
```

Predictions are saved to
`data/benchmark/results/methods/<pipeline>_<epoch>/stability/predictions_seed_<n>.h5ad`.

#### `nn_retraining_with_pseudolabels_mol_emb_learning_missed`

This method is the **general case** that subsumes earlier specialised variants
(rows below show the equivalent flag combination for each retired variant):

| Earlier method | Equivalent configuration |
|---|---|
| `nn_retraining_with_pseudolabels` | `--embedding_type none` |
| `nn_retraining_with_pseudolabels_fingerprints` | `--embedding_type fp --embedding_layer concat` |
| `nn_retraining_with_pseudolabels_embeddings` | `--embedding_type lpm --embedding_layer concat` |

For ad-hoc single-configuration runs, the dedicated stability runner can also be
called directly:

```bash
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh \
    -D <subsample|original> \
    -e <lpm|fp|none> \
    [-l <concat|fixed|trainable>] \
    [-d] \
    [-f <embedding_file>] \
    [-S <seeds>]
```

Options:

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `-D` | `subsample`, `original` | `subsample` | Dataset to use |
| `-e` | `lpm`, `fp`, `none` | `none` | Molecular embedding type |
| `-l` | `concat`, `fixed`, `trainable` | `concat` | How the embedding is integrated into the model |
| `-d` | flag | off | Project fingerprints through an extra Dense layer before concatenation (`concat` mode only) |
| `-f` | filename | `op3_emb.pkl` | Embedding pickle (auto-derives a `<X>` tag from `op3_emb_<X>.pkl`) |
| `-S` | csv of seeds | `0,1,2,3,4,10,11,12,13,14` | Seeds to run (one sbatch job per seed) |

**Embedding layer modes:**

- `concat` — standard learnable embeddings for cell type and compound; external fp/lpm features are appended as a separate input (optionally projected through a Dense layer with `-d`). No external embeddings are used when `-e none`.
- `fixed` — learnable embeddings for cell types; compound embeddings are **initialised** from fp/lpm features and **frozen** (not updated during training), then projected to `emb_out`.
- `trainable` — same as `fixed` but compound embedding weights are **fine-tuned** during training.

Output predictions are named:
- `nn_retraining_with_pseudolabels_mol_emb_learning_missed_<dataset>_none[_<tag>]` for `--embedding_type none`
- `nn_retraining_with_pseudolabels_mol_emb_learning_missed_<dataset>_<type>_concat_<dense|not_dense>[_<tag>]` for `concat` mode
- `nn_retraining_with_pseudolabels_mol_emb_learning_missed_<dataset>_<type>_<fixed|trainable>[_<tag>]` for `fixed`/`trainable` modes

### Step 3 — Compute stability metrics

Computes mean rowwise error (RMSE, MAE) and mean rowwise correlation
(Pearson, Spearman, Cosine) between every seed's predictions and `de_test.h5ad`.

```bash
./run_stability_metrics.sh
```

Results are saved to
`data/benchmark/results/metrics/methods/<pipeline>/stability/seed_<n>/`.

## Jupyter notebooks

Exploratory and post-hoc analysis lives under `notebooks/`. All notebooks are
intended to be run from the `~/notebook_venv` mamba environment created in
[Setup](#setup):

```bash
mamba activate ~/notebook_venv
jupyter lab   # or: jupyter notebook
```

| Notebook | Purpose |
|---|---|
| `01_get_precomputed_LPM_based_embeddings.ipynb` | Extract embedding tables from a trained LPM checkpoint. Run **before** Step 2. |
| `02_match_LPM_based_embeddings_with_OP3.ipynb` | Match LPM embeddings to OP3 compounds + Morgan FPs → writes the `op3_emb_*.pkl` files consumed by Step 2. Run **before** Step 2. |
| `03_plotting_model_results.ipynb` | Aggregate stability metrics and plot results. Run **after** Step 3. |

### Prerequisites for notebook 01 (LPM checkpoint loader)

`01_get_precomputed_LPM_based_embeddings.ipynb` is the only notebook that
depends on the upstream LPM codebase, [perturb-lib](https://github.com/perturblib/perturblib).
It expects a **source checkout** sitting at `~/lpm_style/` (a fork or clone of
`perturb-lib`) and at least one trained LPM checkpoint under
`~/lpm_style/.plib_cache/results/...`. The notebook injects the local source
into `sys.path` at import time:

```python
LPM_STYLE_ROOT = "../../lpm_style"
sys.path = [LPM_STYLE_ROOT] + [p for p in sys.path if "perturblib" not in p and p != LPM_STYLE_ROOT]
```

Two important caveats:

1. **Do not `pip install perturblib`** in the `~/notebook_venv` env. The notebook's
   `sys.path` filter explicitly drops any entry containing `"perturblib"` so the
   local fork wins; a pip-installed copy would shadow your local source if the
   filter ever fails, and the assertion at the bottom of the import cell
   (`"lpm_style" in inspect.getfile(plib)`) is there to catch exactly that.
2. **Training new LPM checkpoints** lives in the upstream repo, not here. Follow
   the [perturb-lib README](https://github.com/perturblib/perturblib) — e.g.
   `poetry run python -m perturb_gym.training train_from_config_file --config_file_id_or_path=lincs_paper_lpm`
   — and the resulting `.ckpt` files will appear under
   `~/lpm_style/.plib_cache/results/<run_name>/LPM_<hash>/seed_<n>/checkpoints/`.

If you only want to **run the stability pipeline + metrics** (Steps 1–3 above)
on existing pickles, you can skip notebook 01 entirely — the
`op3_emb_*.pkl` files in your `data/benchmark/resources/datasets/neurips-2023-data-subsample/`
tree are all that's needed.

## Project structure

```
├── data/                              # Downloaded datasets and results (not tracked by git)
├── notebooks/                         # Jupyter notebooks (run via the ~/notebook_venv mamba env)
├── pipelines/                         # Pipeline scripts (sbatch jobs)
│   ├── common/                        # Shared pipelines (data fetching, metrics, stability metrics)
│   └── nn_retraining_with_pseudolabels_mol_emb_learning_missed/
├── run_pipelines/                     # Thin launchers that call pipelines/ scripts
├── run_stability_pipeline.sh          # Top-level: multi-seed stability sweep
├── run_stability_metrics.sh           # Top-level: stability metrics aggregator
├── src/
│   ├── data/                          # Dataset downloading utilities
│   ├── methods/                       # Prediction method implementations
│   │   └── nn_retraining_with_pseudolabels_mol_emb_learning_missed/
│   ├── metrics/                       # Evaluation metric scripts (R)
│   └── utils/                         # Shared utilities
└── venvs/
    ├── configs/                       # Conda environment definitions (env.yaml per env)
    │   ├── fetching_data/
    │   ├── nn_retraining_with_pseudolabels/
    │   ├── metrics/
    │   └── notebook/                  # Jupyter / analysis env (installed at ~/notebook_venv)
    └── venvs/                         # Installed environments (not tracked by git)
```

## Logs

All logs are written under `logs/`:

```
logs/
├── fetching_data/                     # Data download logs
├── metrics_stability/                 # Stability metrics (wrapper output + sbatch logs)
└── methods/                           # Prediction method logs
    └── nn_retraining_with_pseudolabels_mol_emb_learning_missed_<config>/
        # One dir per config (dataset+embedding+layer+epoch)
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
    │       └── neurips-2023-data-subsample/
    │           ├── de_train.h5ad   # Training perturbation data
    │           ├── de_test.h5ad    # Test perturbation data (ground truth)
    │           ├── id_map.csv      # Cell type / compound mapping for test set
    │           └── op3_emb_*.pkl   # LPM / fingerprint embeddings (per epoch / variant)
    └── results/
        ├── methods/                # Prediction method outputs (one dir per config)
        │   └── nn_retraining_with_pseudolabels_mol_emb_learning_missed_<config>/
        │       └── stability/
        │           └── predictions_seed_<n>.h5ad
        └── metrics/                # Metric outputs
            └── methods/
                └── <method>/
                    └── stability/
                        └── seed_<n>/
                            ├── mean_rowwise_error.h5ad
                            └── mean_rowwise_correlation.h5ad
```
