# This script is based on a Kaggle competition solution using transformer ensembles.

import os
import pandas as pd
import anndata as ad
import torch

from src.methods.transformer_ensemble.cli import parse_args
from src.methods.transformer_ensemble.utils import prepare_augmented_data, prepare_augmented_data_mean_only
from src.methods.transformer_ensemble.train import train_k_means_strategy, train_non_k_means_strategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

par = parse_args()

if par["output_model"]:
    os.makedirs(par["output_model"], exist_ok=True)

de_train = ad.read_h5ad(par["de_train"])
id_map = pd.read_csv(par["id_map"])

for col in de_train.obs.select_dtypes(include=["category"]).columns:
    de_train.obs[col] = de_train.obs[col].astype(str)
de_train.obs.reset_index(drop=True, inplace=True)

gene_names = list(de_train.var_names)
n_components = len(gene_names)

argsets = [
    {"mean_std": "mean_std", "uncommon": False, "sampling_strategy": "random",  "weight": 0.5},
    {"mean_std": "mean_std", "uncommon": True,  "sampling_strategy": "random",  "weight": 0.25},
    {"mean_std": "mean_std", "uncommon": False, "sampling_strategy": "k-means", "weight": 0.25},
    {"mean_std": "mean",     "uncommon": False, "sampling_strategy": "random",  "weight": 0.3},
]

predictions = []

print(f"Train and predict models", flush=True)
for i, argset in enumerate(argsets):
    print(f"Train and predict model {i+1}/{len(argsets)}", flush=True)

    print(f"> Prepare augmented data", flush=True)
    if argset["mean_std"] == "mean_std":
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data(
            de_train=de_train,
            id_map=id_map,
            layer=par["layer"],
            uncommon=argset["uncommon"],
        )
    elif argset["mean_std"] == "mean":
        one_hot_encode_features, targets, one_hot_test = prepare_augmented_data_mean_only(
            de_train=de_train,
            id_map=id_map,
            layer=par["layer"],
        )
    else:
        raise ValueError("Invalid mean_std argument")

    print(f"> Train model", flush=True)
    if argset["sampling_strategy"] == "k-means":
        label_reducer, scaler, transformer_model = train_k_means_strategy(
            n_components=n_components,
            d_model=par["d_model"],
            one_hot_encode_features=one_hot_encode_features,
            targets=targets,
            num_epochs=par["num_train_epochs"],
            early_stopping=par["early_stopping"],
            batch_size=par["batch_size"],
            device=device,
            mean_std=argset["mean_std"],
        )
    elif argset["sampling_strategy"] == "random":
        label_reducer, scaler, transformer_model = train_non_k_means_strategy(
            n_components=n_components,
            d_model=par["d_model"],
            one_hot_encode_features=one_hot_encode_features,
            targets=targets,
            num_epochs=par["num_train_epochs"],
            early_stopping=par["early_stopping"],
            batch_size=par["batch_size"],
            device=device,
            mean_std=argset["mean_std"],
        )
    else:
        raise ValueError("Invalid sampling_strategy argument")

    print(f"> Predict model", flush=True)
    unseen_data = torch.tensor(one_hot_test, dtype=torch.float32).to(device)
    num_features = one_hot_encode_features.shape[1]

    if n_components == num_features:
        label_reducer = None
        scaler = None

    num_samples = len(unseen_data)
    transformed_data = []
    for j in range(0, num_samples, par["batch_size"]):
        batch_result = transformer_model(unseen_data[j: j + par["batch_size"]])
        transformed_data.append(batch_result)
    transformed_data = torch.vstack(transformed_data)

    if scaler:
        transformed_data = torch.tensor(
            scaler.inverse_transform(
                label_reducer.inverse_transform(transformed_data.cpu().detach().numpy())
            )
        ).to(device)

    pred = transformed_data.cpu().detach().numpy()

    if par["output_model"]:
        torch.save(transformer_model.state_dict(), f"{par['output_model']}/model_{i}.pt")
        pd.DataFrame(pred).to_csv(f"{par['output_model']}/pred_{i}.csv", index=False)

    predictions.append(pred)

print(f"Combine predictions", flush=True)
weighted_pred = sum([pred * argset["weight"] for argset, pred in zip(argsets, predictions)])

print('Write output to file', flush=True)
os.makedirs(os.path.dirname(par["output"]) or ".", exist_ok=True)
output = ad.AnnData(
    layers={"prediction": weighted_pred},
    obs=pd.DataFrame(index=id_map["id"]),
    var=pd.DataFrame(index=gene_names),
    uns={
        "dataset_id": de_train.uns["dataset_id"],
        "method_id": "transformer_ensemble",
    }
)

output.write_h5ad(par["output"], compression="gzip")
