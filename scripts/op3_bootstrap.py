"""Paired OP3 bootstrap comparisons against the descriptor-free reference."""

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Reuse the bootstrap implementation maintained with Chem-PerturBridge.
CHEM_SCRIPTS = Path.home() / "Chem-PerturBridge_analysis/scripts"
sys.path.append(str(CHEM_SCRIPTS))
from cluster_bootstrap_ci import (
    OBSERVED_COMPOUND_PANEL_SCOPE,
    cluster_bca_nested_mean_ci_table,
)

SEEDS = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
METHOD_PREFIX = "nn_retraining_with_pseudolabels_mol_emb_learning_missed_subsample_"
DATASET = "resources/datasets/neurips-2023-data-subsample"
METRICS = {
    "mrrmse": ("mean_rowwise_rmse", "delta_mean_rowwise_mrrmse", False),
    "mae": ("mean_rowwise_mae", "delta_mean_rowwise_mae", False),
    "spearman": ("mean_rowwise_spearman", "delta_mean_rowwise_spearman", True),
    "cosine": ("mean_rowwise_cosine", "delta_mean_rowwise_cosine", True),
}
MAIN_METHODS = [
    "lpm_trainable_all_20", "lpm_fixed_all_20", "fp_trainable_fp",
    "lpm_trainable_l1000_20", "lpm_fixed_l1000_20", "fp_fixed_fp",
    "chemb1_fixed_chemb1", "chemb1_trainable_chemb1",
    "chemb2_fixed_chemb2", "chemb2_trainable_chemb2",
    "molformer_fixed_molformer", "molformer_trainable_molformer",
    "lpm_trainable_all_20_narrowed", "lpm_fixed_all_20_narrowed",
]


def method_roots(project_root, coverage, analysis):
    """Select 18 main or ablation variants and their prediction folders."""
    if coverage not in ("full_coverage", "coverage_match"):
        raise ValueError(f"Unknown coverage: {coverage}")
    if analysis not in ("main", "ablation"):
        raise ValueError(f"Unknown analysis: {analysis}")
    if analysis == "main":
        all_step = ("all_last_full_cov_stepmatched" if coverage == "full_coverage"
                    else "all_last_stepmatched_coverage_match")
        extra = [f"lpm_{mode}_{arm}" for arm in (all_step, "l1000_last_stepmatched")
                 for mode in ("trainable", "fixed")]
        methods = extra + MAIN_METHODS
    else:
        extra = []
        for family in ("all", "l1000"):
            for removed in ("dataset", "dose_time", "dataset_dose_time"):
                if family == "all" and coverage == "full_coverage":
                    arm = f"all_20_full_cov_abl_{removed}"
                elif family == "all":
                    arm = f"all_20_abl_{removed}_coverage_match"
                else:
                    arm = f"l1000_20_abl_{removed}"
                extra.extend(f"lpm_{mode}_{arm}" for mode in ("trainable", "fixed"))
        methods = extra + [m for m in MAIN_METHODS if m.startswith("lpm_")]

    full = Path(project_root) / "data_full_coverage/benchmark"
    matched = Path(project_root) / "data_coverage_match/benchmark"
    roots = {"none": full}
    for method in methods:
        reuse_full = "l1000" in method or "narrowed" in method
        roots[method] = full if coverage == "full_coverage" or reuse_full else matched
    return roots


def _categorical(handle, name):
    group = handle[f"obs/{name}"]
    return group["categories"].asstr()[:][group["codes"][:]]


def _load_test(root):
    with h5py.File(root / DATASET / "de_test.h5ad", "r") as handle:
        return (
            handle["layers/clipped_sign_log10_pval"][:].astype(np.float64),
            handle["var/_index"].asstr()[:],
            _categorical(handle, "SMILES").astype(str),
            _categorical(handle, "cell_type").astype(str),
        )


def _load_prediction(path, genes, shape):
    with h5py.File(path, "r") as handle:
        prediction = handle["layers/prediction"][:].astype(np.float64)
        pred_genes = handle["var/_index"].asstr()[:]
    if not np.array_equal(pred_genes, genes):
        positions = {gene: i for i, gene in enumerate(pred_genes)}
        prediction = prediction[:, [positions[gene] for gene in genes]]
    if prediction.shape != shape:
        raise ValueError(f"Prediction shape differs from ground truth: {path}")
    if np.isnan(prediction).any():
        prediction = np.nan_to_num(prediction, nan=0.0)
    return prediction


def row_mrrmse(y, prediction):
    return np.sqrt(np.mean((y - prediction) ** 2, axis=1))


def row_mae(y, prediction):
    return np.mean(np.abs(y - prediction), axis=1)


def row_spearman(y, prediction):
    yr = np.apply_along_axis(rankdata, 1, y).astype(np.float64)
    pr = np.apply_along_axis(rankdata, 1, prediction).astype(np.float64)
    yr -= yr.mean(axis=1, keepdims=True)
    pr -= pr.mean(axis=1, keepdims=True)
    numerator = (yr * pr).sum(axis=1)
    denominator = np.sqrt((yr * yr).sum(axis=1) * (pr * pr).sum(axis=1))
    out = np.zeros(y.shape[0], dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    out[~np.isfinite(out)] = 0.0
    return out


def row_cosine(y, prediction):
    numerator = (y * prediction).sum(axis=1)
    denominator = np.linalg.norm(y, axis=1) * np.linalg.norm(prediction, axis=1)
    out = np.zeros(y.shape[0], dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    out[~np.isfinite(out)] = 0.0
    return out


ROW_SCORES = {"mrrmse": row_mrrmse, "mae": row_mae,
              "spearman": row_spearman, "cosine": row_cosine}


def load_scores(roots, seeds=SEEDS):
    """Recompute per-row scores and check them against saved pipeline metrics."""
    y, genes, compounds, cell_types = _load_test(roots["none"])
    for root in set(roots.values()) - {roots["none"]}:
        other = _load_test(root)
        if not all(np.array_equal(a, b) for a, b in zip((y, genes, compounds, cell_types), other)):
            raise ValueError(f"Test values, genes, or row order differ in {root}")
    all_scores, checks = {}, []
    for method, root in roots.items():
        print(f"Loading {method}", flush=True)
        run = root.parent.name
        name = METHOD_PREFIX + method
        scores = {score: np.empty((len(seeds), len(y)), dtype=np.float64) for score in METRICS}
        for i, seed in enumerate(seeds):
            path = root / "results/methods" / name / "stability" / f"predictions_seed_{seed}.h5ad"
            prediction = _load_prediction(path, genes, y.shape)
            stored = {}
            for kind in ("mean_rowwise_error", "mean_rowwise_correlation"):
                path = root / "results/metrics/methods" / name / "stability" / f"seed_{seed}" / f"{kind}.h5ad"
                with h5py.File(path, "r") as handle:
                    stored.update(zip(handle["uns/metric_ids"].asstr()[:], handle["uns/metric_values"][:]))
            for score, function in ROW_SCORES.items():
                scores[score][i] = function(y, prediction)
                op3_name = METRICS[score][0]
                recomputed = float(scores[score][i].mean())
                checks.append(dict(run=run, method=method, seed=seed, score=score,
                                   op3_name=op3_name, recomputed=recomputed,
                                   stored=float(stored[op3_name]),
                                   abs_diff=abs(recomputed - stored[op3_name])))
        all_scores[method] = scores
    check_df = pd.DataFrame(checks)
    if not (check_df["abs_diff"] < 1e-8).all():
        raise ValueError("Recomputed metrics disagree with stored metrics:\n" +
                         check_df.loc[~(check_df["abs_diff"] < 1e-8)].to_string(index=False))
    print(f"All {len(check_df)} metric checks passed; maximum difference {check_df.abs_diff.max():.3g}")
    return all_scores, compounds, cell_types, check_df


def paired_results(all_scores, compounds, cell_types, roots):
    """Seed-average scores, then construct paired row and compound comparisons."""
    averages = {method: {score: values.mean(axis=0) for score, values in scores.items()}
                for method, scores in all_scores.items()}
    units, compound_rows = [], []
    for method in roots:
        if method == "none":
            continue
        for score, (op3_name, metric, higher_is_better) in METRICS.items():
            ref, variant = averages["none"][score], averages[method][score]
            delta = variant - ref if higher_is_better else ref - variant
            wins = variant > ref if higher_is_better else variant < ref
            common = dict(run=roots[method].parent.name, var_method=method,
                          score=score, op3_name=op3_name, metric=metric)
            for i, compound in enumerate(compounds):
                units.append(dict(**common, compound=compound, cell_type=cell_types[i],
                                  delta_row=float(delta[i]), win_row=float(wins[i])))
            grouped = pd.DataFrame({"compound": compounds, "ref": ref, "var": variant})
            grouped = grouped.groupby("compound", as_index=False)[["ref", "var"]].mean()
            delta = grouped["var"] - grouped["ref"] if higher_is_better else grouped["ref"] - grouped["var"]
            for compound, value in zip(grouped["compound"], delta):
                compound_rows.append(dict(**common, compound=compound,
                                          delta_compound=float(value), win_compound=float(value > 0)))
    return pd.DataFrame(units), pd.DataFrame(compound_rows)


def bootstrap_summary(units, compound_wins, seeds=SEEDS, n_boot=2000,
                      bootstrap_seed=20260505, ci_level=0.95):
    """Resample compounds; average within cell types, then across cell types."""
    # These keys/labels also determine the helper's reproducible random streams.
    keys = ["var_method", "score", "op3_name"]
    options = dict(group_cols=keys, cluster_col="compound", inner_cols=["cell_type"],
                   outer_cols=None, n_boot=n_boot, ci_level=ci_level, seed=bootstrap_seed,
                   uncertainty_scope=OBSERVED_COMPOUND_PANEL_SCOPE)
    delta_ci = cluster_bca_nested_mean_ci_table(
        units, metric_cols={"delta_row": "delta_row"},
        summary_level="op3_variant_metric", **options,
    ).rename(columns={
        "mean": "delta_hat", "ci_low": "delta_ci_low", "ci_high": "delta_ci_high",
        "ci_half_width": "delta_ci_half_width", "ci_method": "delta_ci_method",
        "ci_status": "delta_ci_status", "n_bootstrap_valid": "delta_n_bootstrap_valid",
    })
    win_ci = cluster_bca_nested_mean_ci_table(
        units, metric_cols={"win_rate": "win_row"},
        summary_level="op3_variant_metric_winrate_nested_celltype", **options,
    ).rename(columns={
        "mean": "win_rate", "ci_low": "win_rate_ci_low", "ci_high": "win_rate_ci_high",
        "ci_half_width": "win_rate_ci_half_width", "ci_method": "win_rate_ci_method",
        "ci_status": "win_rate_ci_status", "n_bootstrap_valid": "win_rate_n_bootstrap_valid",
    })
    delta_ci["metric"] = delta_ci["score"].map(lambda score: METRICS[score][1])
    summary = delta_ci[keys + [
        "metric", "delta_hat", "delta_ci_low", "delta_ci_high", "delta_ci_half_width",
        "delta_ci_method", "delta_ci_status", "delta_n_bootstrap_valid",
        "n_compounds", "n_rows", "n_finite_rows", "inner_strata", "cluster_col",
        "ci_level", "n_bootstrap_iterations", "uncertainty_scope",
    ]].merge(win_ci[keys + [
        "win_rate", "win_rate_ci_low", "win_rate_ci_high", "win_rate_ci_half_width",
        "win_rate_ci_method", "win_rate_ci_status", "win_rate_n_bootstrap_valid",
    ]], on=keys, validate="one_to_one")
    compound_rate = compound_wins.groupby(keys, as_index=False).agg(win_rate_compound=("win_compound", "mean"))
    summary = summary.merge(compound_rate, on=keys, validate="one_to_one")
    summary["ref_method"] = "none"
    summary["excludes_zero"] = (summary.delta_ci_low > 0) | (summary.delta_ci_high < 0)
    summary["win_rate_excludes_0.5"] = (summary.win_rate_ci_low > 0.5) | (summary.win_rate_ci_high < 0.5)
    summary["n_seeds"] = len(seeds)
    summary["bootstrap_seed"] = bootstrap_seed
    summary["design"] = "chem_style_unit_delta_and_win_nested_celltype_seedavg_first"
    return summary.sort_values(["score", "delta_hat"], ascending=[True, False]).reset_index(drop=True)


def readable_summary(summary):
    """Format estimates and intervals as one row per method."""
    def bound(value):
        return f"{value:.5f}" if 0 < abs(value) < 1e-4 else f"{value:.4f}"

    rows = []
    methods = summary.loc[summary.score == "mrrmse"].sort_values("delta_hat").var_method
    indexed = summary.set_index(["var_method", "score"])
    for method in methods:
        row = {"method": method}
        for score in ("spearman", "cosine", "mrrmse", "mae"):
            value = indexed.loc[(method, score)]
            row[f"{score} delta [95% CI]"] = f"{value.delta_hat:+.4f} [{bound(value.delta_ci_low)}, {bound(value.delta_ci_high)}]"
            row[f"{score} win rate [95% CI]"] = f"{value.win_rate:.3f} [{value.win_rate_ci_low:.3f}, {value.win_rate_ci_high:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def save_summary(project_root, coverage, analysis, summary):
    results = Path(project_root) / "results"
    results.mkdir(parents=True, exist_ok=True)
    prefix = "df_bootstrap_summary" if analysis == "main" else "df_bootstrap_ablation_summary"
    path = results / f"{prefix}_{coverage}.csv"
    readable_summary(summary).to_csv(path, index=False)
    return path
