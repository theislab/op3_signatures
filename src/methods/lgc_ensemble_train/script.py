import json

import numpy as np
import pandas as pd
import torch

if torch.cuda.is_available():
    print("using device: cuda", flush=True)
else:
    print("using device: cpu", flush=True)

from src.methods.lgc_ensemble_train.cli import parse_args
from src.methods.lgc_ensemble_helpers.models import Conv, LSTM, GRU
from src.methods.lgc_ensemble_helpers.helper_functions import train_function


def main():
    par = parse_args()
    print(f"par: {par}", flush=True)

    aug = par["train_data_aug_dir"]

    print("Load data...", flush=True)
    with open(f"{aug}/kf_cv_{par['scheme']}.json") as fh:
        kf_cv = json.load(fh)

    train_idx, val_idx = kf_cv[par["fold"]]

    X = np.load(f"{aug}/X_vec_{par['scheme']}.npy")
    y = np.load(f"{aug}/y.npy")
    cell_types_sm_names = pd.read_csv(f"{aug}/cell_types_sm_names.csv")

    with open(f"{aug}/config.json") as fh:
        config = json.load(fh)

    print("Prepare data...", flush=True)
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    info_data = {
        "train_cell_type": cell_types_sm_names.iloc[train_idx]["cell_type"].tolist(),
        "val_cell_type":   cell_types_sm_names.iloc[val_idx]["cell_type"].tolist(),
        "train_sm_name":   cell_types_sm_names.iloc[train_idx]["sm_name"].tolist(),
        "val_sm_name":     cell_types_sm_names.iloc[val_idx]["sm_name"].tolist(),
    }

    schemes = ["initial", "light", "heavy"]
    clip_norm = config["CLIP_VALUES"][schemes.index(par["scheme"])]

    models = {"LSTM": LSTM, "GRU": GRU, "Conv": Conv}
    ModelClass = models[par["model"]]
    model = ModelClass(par["scheme"], X.shape, y.shape)

    print("Start training...", flush=True)
    model, results = train_function(
        model, model.name,
        x_train, y_train, x_val, y_val,
        info_data, config=config, clip_norm=clip_norm,
    )
    model.to("cpu")

    print("Save model...", flush=True)
    torch.save(model.state_dict(), par["model_file"])
    with open(par["log_file"], "w") as fh:
        json.dump(results, fh)

    print("### Done.")


if __name__ == "__main__":
    main()
