import os
import json

import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.model_selection import KFold as KF

if torch.cuda.is_available():
    print("using device: cuda", flush=True)
else:
    print("using device: cpu", flush=True)

from src.methods.lgc_ensemble_prepare.cli import parse_args
from src.methods.lgc_ensemble_helpers.helper_functions import (
    seed_everything,
    one_hot_encode,
    save_ChemBERTa_features,
    combine_features,
)
from src.utils.anndata_to_dataframe import anndata_to_dataframe


def main():
    par = parse_args()

    seed_everything()

    os.makedirs(par["train_data_aug_dir"], exist_ok=True)

    print("\nPreparing data...", flush=True)
    de_train = ad.read_h5ad(par["de_train"])
    de_train_df = anndata_to_dataframe(de_train, par["layer"])
    de_train_df = de_train_df.drop(columns=["split"])
    id_map = pd.read_csv(par["id_map"])

    gene_names = list(de_train.var_names)

    print("Create data augmentation", flush=True)
    de_cell_type = de_train_df.iloc[:, [0] + list(range(5, de_train_df.shape[1]))]
    de_sm_name   = de_train_df.iloc[:, [1] + list(range(5, de_train_df.shape[1]))]

    mean_cell_type = de_cell_type.groupby("cell_type").mean().reset_index()
    mean_sm_name   = de_sm_name.groupby("sm_name").mean().reset_index()
    std_cell_type  = de_cell_type.groupby("cell_type").std().reset_index()
    std_sm_name    = de_sm_name.groupby("sm_name").std().reset_index()
    std_sm_name    = std_sm_name.fillna(0)

    cell_types = de_cell_type.groupby("cell_type").quantile(0.1).reset_index()["cell_type"]
    quantiles_cell_type = pd.concat(
        [pd.DataFrame(cell_types)] + [
            de_cell_type.groupby("cell_type")[col]
            .quantile([0.25, 0.50, 0.75], interpolation="linear")
            .unstack()
            .reset_index(drop=True)
            for col in list(de_train_df.columns)[5:]
        ],
        axis=1,
    )

    print("Save data augmentation features", flush=True)
    aug = par["train_data_aug_dir"]
    mean_cell_type.to_csv(f"{aug}/mean_cell_type.csv", index=False)
    std_cell_type.to_csv(f"{aug}/std_cell_type.csv",   index=False)
    mean_sm_name.to_csv(f"{aug}/mean_sm_name.csv",     index=False)
    std_sm_name.to_csv(f"{aug}/std_sm_name.csv",       index=False)
    quantiles_cell_type.to_csv(f"{aug}/quantiles_cell_type.csv", index=False)
    with open(f"{aug}/gene_names.json", "w") as f:
        json.dump(gene_names, f)

    print("Create one hot encoding features", flush=True)
    one_hot_train, _ = one_hot_encode(
        de_train_df[["cell_type", "sm_name"]],
        id_map[["cell_type", "sm_name"]],
        out_dir=aug,
    )
    one_hot_train = pd.DataFrame(one_hot_train)

    print("Prepare ChemBERTa features", flush=True)
    train_chem_feat, train_chem_feat_mean = save_ChemBERTa_features(
        de_train_df["SMILES"].tolist(), out_dir=aug, on_train_data=True
    )
    sm_name2smiles = dict(zip(de_train_df["sm_name"], de_train_df["SMILES"]))
    test_smiles = list(map(sm_name2smiles.get, id_map["sm_name"].values))
    _, _ = save_ChemBERTa_features(test_smiles, out_dir=aug, on_train_data=False)

    cell_types_sm_names = de_train_df[["cell_type", "sm_name"]]
    cell_types_sm_names.to_csv(f"{aug}/cell_types_sm_names.csv", index=False)

    print("Store Xs and y", flush=True)
    ylist = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    y = de_train_df.drop(columns=ylist)

    X_vec = combine_features(
        [mean_cell_type, std_cell_type, mean_sm_name, std_sm_name],
        [train_chem_feat, train_chem_feat_mean],
        de_train_df, one_hot_train,
    )
    np.save(f"{aug}/X_vec_initial.npy", X_vec)

    X_vec_light = combine_features(
        [mean_cell_type, mean_sm_name],
        [train_chem_feat, train_chem_feat_mean],
        de_train_df, one_hot_train,
    )
    np.save(f"{aug}/X_vec_light.npy", X_vec_light)

    X_vec_heavy = combine_features(
        [quantiles_cell_type, mean_cell_type, mean_sm_name],
        [train_chem_feat, train_chem_feat_mean],
        de_train_df, one_hot_train, quantiles_cell_type,
    )
    np.save(f"{aug}/X_vec_heavy.npy", X_vec_heavy)

    np.save(f"{aug}/y.npy", y.values)

    print("Store config and shapes", flush=True)
    config = {
        "LEARNING_RATES": [0.001, 0.001, 0.0003],
        "CLIP_VALUES":    [5.0, 1.0, 1.0],
        "EPOCHS":         par["epochs"],
        "KF_N_SPLITS":    par["kf_n_splits"],
        "SCHEMES":        par["schemes"],
        "MODELS":         par["models"],
        "DATASET_ID":     de_train.uns["dataset_id"],
    }
    with open(f"{aug}/config.json", "w") as fh:
        json.dump(config, fh)

    shapes = {
        "xshapes": {
            "initial": list(X_vec.shape),
            "light":   list(X_vec_light.shape),
            "heavy":   list(X_vec_heavy.shape),
        },
        "yshape": list(y.shape),
    }
    with open(f"{aug}/shapes.json", "w") as fh:
        json.dump(shapes, fh)

    print("Store cross-validation indices", flush=True)
    kf_cv = KF(n_splits=config["KF_N_SPLITS"], shuffle=True, random_state=42)

    def get_kf_index(X, kf):
        return [(tr.tolist(), va.tolist()) for tr, va in kf.split(X)]

    json.dump(get_kf_index(X_vec,       kf_cv), open(f"{aug}/kf_cv_initial.json", "w"))
    json.dump(get_kf_index(X_vec_light, kf_cv), open(f"{aug}/kf_cv_light.json",   "w"))
    json.dump(get_kf_index(X_vec_heavy, kf_cv), open(f"{aug}/kf_cv_heavy.json",   "w"))

    print("### Done.")


if __name__ == "__main__":
    main()
