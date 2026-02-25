import os

import anndata as ad
import pandas as pd

from src.control_methods.mean_across_compounds.cli import parse_args
from src.utils.anndata_to_dataframe import anndata_to_dataframe


def main():
    par = parse_args()

    de_train   = ad.read_h5ad(par["de_train"])
    id_map     = pd.read_csv(par["id_map"])
    gene_names = list(de_train.var_names)
    de_train_df = anndata_to_dataframe(de_train, par["layer"])

    mean_compound = de_train_df.groupby("sm_name")[gene_names].mean()
    mean_compound = mean_compound.loc[id_map.sm_name]

    os.makedirs(os.path.dirname(par["output"]) or ".", exist_ok=True)
    output = ad.AnnData(
        layers={"prediction": mean_compound.values},
        obs=pd.DataFrame(index=id_map["id"]),
        var=pd.DataFrame(index=gene_names),
        uns={
            "dataset_id": de_train.uns["dataset_id"],
            "method_id":  "mean_across_compounds",
        },
    )
    output.write_h5ad(par["output"], compression="gzip")


if __name__ == "__main__":
    main()
