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

from src.methods.nn_retraining_with_pseudolabels.cli import parse_args
from src.methods.nn_retraining_with_pseudolabels.notebook_264 import run_notebook_264
from src.methods.nn_retraining_with_pseudolabels.notebook_266 import run_notebook_266
from src.utils.anndata_to_dataframe import anndata_to_dataframe

par = parse_args()

de_train = ad.read_h5ad(par["de_train"])
de_train_df = anndata_to_dataframe(de_train, par["layer"])

de_train_df = de_train_df.sample(frac=1.0, random_state=42)
de_train_df = de_train_df.reset_index(drop=True)

id_map = pd.read_csv(par["id_map"])

gene_names = list(de_train.var_names)

de_train_df = de_train_df.loc[:, ["cell_type", "sm_name"] + gene_names]

pseudolabel = run_notebook_264(de_train_df, id_map, gene_names, par["reps"])

pseudolabel = pd.concat(
    [id_map[["cell_type", "sm_name"]], pseudolabel.loc[:, gene_names]], axis=1
)

df = run_notebook_266(de_train_df, id_map, pseudolabel, gene_names, par["reps"])

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
