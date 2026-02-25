import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

if torch.cuda.is_available():
    print("using device: cuda", flush=True)
else:
    print("using device: cpu", flush=True)

from src.methods.lgc_ensemble_predict.cli import parse_args
from src.methods.lgc_ensemble_helpers.helper_functions import (
    combine_features,
    lazy_load_trained_models,
    average_prediction,
    weighted_average_prediction,
)


def main():
    par = parse_args()
    print(f"par: {par}", flush=True)

    aug = par["train_data_aug_dir"]

    print("\nReading data...", flush=True)
    train_config = json.load(open(f"{aug}/config.json"))
    test_config = {
        "MODEL_COEFS": [0.29, 0.33, 0.38],
        "FOLD_COEFS":  [0.25, 0.15, 0.2, 0.15, 0.25],
        "KF_N_SPLITS": train_config["KF_N_SPLITS"],
    }

    id_map = pd.read_csv(par["id_map"])

    with open(f"{aug}/gene_names.json") as fh:
        gene_names = json.load(fh)

    mean_cell_type = pd.read_csv(f"{aug}/mean_cell_type.csv")
    std_cell_type  = pd.read_csv(f"{aug}/std_cell_type.csv")
    mean_sm_name   = pd.read_csv(f"{aug}/mean_sm_name.csv")
    std_sm_name    = pd.read_csv(f"{aug}/std_sm_name.csv")
    quantiles_df   = pd.read_csv(f"{aug}/quantiles_cell_type.csv")

    test_chem_feat      = np.load(f"{aug}/chemberta_test.npy")
    test_chem_feat_mean = np.load(f"{aug}/chemberta_test_mean.npy")
    one_hot_test        = pd.DataFrame(np.load(f"{aug}/one_hot_test.npy"))

    test_vec       = combine_features([mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],
                                      [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_light = combine_features([mean_cell_type, mean_sm_name],
                                      [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test)
    test_vec_heavy = combine_features([quantiles_df, mean_cell_type, mean_sm_name],
                                      [test_chem_feat, test_chem_feat_mean], id_map, one_hot_test, quantiles_df)

    print("\nLoading trained models...", flush=True)
    trained_models = lazy_load_trained_models(
        aug, par["model_files"], kf_n_splits=test_config["KF_N_SPLITS"]
    )
    fold_weights = (
        test_config["FOLD_COEFS"]
        if test_config["KF_N_SPLITS"] == len(test_config["FOLD_COEFS"])
        else [1.0 / test_config["KF_N_SPLITS"]] * test_config["KF_N_SPLITS"]
    )

    print("\nStarting predictions...", flush=True)
    t0 = time.time()
    if "light" in train_config["SCHEMES"]:
        print("\nPredicting light models...", flush=True)
        pred1 = average_prediction(test_vec_light, trained_models["light"])
        pred2 = weighted_average_prediction(test_vec_light, trained_models["light"],
                                            model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    if "initial" in train_config["SCHEMES"]:
        print("\nPredicting initial models...", flush=True)
        pred3 = average_prediction(test_vec, trained_models["initial"])
        pred4 = weighted_average_prediction(test_vec, trained_models["initial"],
                                            model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    if "heavy" in train_config["SCHEMES"]:
        print("\nPredicting heavy models...", flush=True)
        pred5 = average_prediction(test_vec_heavy, trained_models["heavy"])
        pred6 = weighted_average_prediction(test_vec_heavy, trained_models["heavy"],
                                            model_wise=test_config["MODEL_COEFS"], fold_wise=fold_weights)
    print(f"Prediction time: {time.time() - t0:.1f}s", flush=True)

    print("\nEnsembling predictions and writing to file...", flush=True)

    df_sub_ix  = id_map.set_index(["cell_type", "sm_name"])
    submission = pd.DataFrame(index=df_sub_ix.index, columns=gene_names)

    def weighted_blend(coefs):
        submission[gene_names] = 0
        weight = 0.0
        if "light" in train_config["SCHEMES"]:
            submission[gene_names] += coefs[0] * pred1 + coefs[1] * pred2
            weight += coefs[0] + coefs[1]
        if "initial" in train_config["SCHEMES"]:
            submission[gene_names] += coefs[2] * pred3 + coefs[3] * pred4
            weight += coefs[2] + coefs[3]
        if "heavy" in train_config["SCHEMES"]:
            submission[gene_names] += coefs[4] * pred5 + coefs[5] * pred6
            weight += coefs[4] + coefs[5]
        submission[gene_names] /= weight
        return submission.copy()

    df1 = weighted_blend([0.23, 0.15, 0.18, 0.15, 0.15, 0.14])
    df2 = weighted_blend([0.13, 0.15, 0.23, 0.15, 0.20, 0.16])
    df3 = weighted_blend([0.17, 0.16, 0.17, 0.16, 0.18, 0.16])

    df_sub = (0.34 * df1 + 0.33 * df2 + 0.33 * df3)
    df_sub.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(par["output"]) or ".", exist_ok=True)

    output = ad.AnnData(
        layers={"prediction": df_sub.to_numpy()},
        obs=pd.DataFrame(index=id_map["id"]),
        var=pd.DataFrame(index=gene_names),
        uns={
            "dataset_id": train_config["DATASET_ID"],
            "method_id":  "lgc_ensemble",
        },
    )
    print(output)
    output.write_h5ad(par["output"], compression="gzip")
    print("\nDone.")


if __name__ == "__main__":
    main()
