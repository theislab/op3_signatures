import os

import anndata as ad
import numpy as np
import pandas as pd

from src.control_methods.mean_outcome.cli import parse_args
from src.utils.anndata_to_dataframe import anndata_to_dataframe


def main():
    par = parse_args()

    de_train   = ad.read_h5ad(par["de_train"])
    id_map     = pd.read_csv(par["id_map"])
    gene_names = list(de_train.var_names)
    de_train_df = anndata_to_dataframe(de_train, par["layer"])

    mean_pred  = de_train_df[gene_names].mean(axis=0)
    prediction = np.vstack([mean_pred.values] * id_map.shape[0])

    os.makedirs(os.path.dirname(par["output"]) or ".", exist_ok=True)
    output = ad.AnnData(
        layers={"prediction": prediction},
        obs=pd.DataFrame(index=id_map["id"]),
        var=pd.DataFrame(index=gene_names),
        uns={
            "dataset_id": de_train.uns["dataset_id"],
            "method_id":  "mean_outcome",
        },
    )
    output.write_h5ad(par["output"], compression="gzip")


if __name__ == "__main__":
    main()
