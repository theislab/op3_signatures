# According to the author, the output is created by running two notebooks:
#
#  * notebook 264: https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153045206
#  * notebook 266: https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153141755
#
# This component was created by:
#  * Taking the code of both notebooks
#  * Moving the code corresponding to the weights and models from each notebook to a separate helper file
#  * Write this script:
#      - Load the data in this script
#      - Run notebook 264 on it
#      - Run notebook 266 on the combined training data and output of notebook 264

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import anndata as ad

from src.methods.nn_retraining_with_pseudolabels_embeddings.cli import parse_args
from src.methods.nn_retraining_with_pseudolabels_embeddings.notebook_264 import run_notebook_264
from src.methods.nn_retraining_with_pseudolabels_embeddings.notebook_266 import run_notebook_266
from src.utils.anndata_to_dataframe import anndata_to_dataframe


def merge_smiles_sm_name_for_id_map(id_map, smiles_df):
    """Merge SMILES into id_map by sm_name.

    Args:
        id_map: DataFrame with at least column sm_name (and typically id, cell_type).
        smiles_df: DataFrame with columns sm_name and SMILES.

    Returns:
        id_map with SMILES column added (left join on sm_name). Rows whose sm_name
        is missing in smiles_df get NaN in SMILES.
    """
    smiles_df = smiles_df.copy()
    sm = smiles_df[["sm_name", "SMILES"]].drop_duplicates(subset="sm_name")
    return id_map.merge(sm, on="sm_name", how="left")


par = parse_args()

de_train = ad.read_h5ad(par["de_train"])
id_map = pd.read_csv(par["id_map"])

de_train_df = anndata_to_dataframe(de_train, par["layer"])
id_map = merge_smiles_sm_name_for_id_map(id_map, de_train_df)

de_train_df = de_train_df.sample(frac=1.0, random_state=42)
de_train_df = de_train_df.reset_index(drop=True)


gene_names = list(de_train.var_names)

de_train_df = de_train_df.loc[:, ["SMILES", "cell_type", "sm_name"] + gene_names]

pseudolabel = run_notebook_264(de_train_df, id_map, gene_names, par["reps"], par["use_fp_dense"])

pseudolabel = pd.concat(
    [id_map[["SMILES", "cell_type", "sm_name"]], pseudolabel.loc[:, gene_names]], axis=1
)

df = run_notebook_266(de_train_df, id_map, pseudolabel, gene_names, par["reps"], par["use_fp_dense"])

print('Write output to file', flush=True)
import os
os.makedirs(os.path.dirname(par["output"]) or ".", exist_ok=True)
output = ad.AnnData(
    layers={"prediction": df[gene_names].to_numpy()},
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
        "dataset_id": de_train.uns["dataset_id"],
        "method_id": "nn_retraining_with_pseudolabels",
    }
)

output.write_h5ad(par["output"], compression="gzip")
