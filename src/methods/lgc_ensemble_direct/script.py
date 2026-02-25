import shutil
import tempfile

import torch
if torch.cuda.is_available():
    print(f"detected {torch.cuda.device_count()} cuda devices", flush=True)
else:
    print("using device: cpu", flush=True)

from src.methods.lgc_ensemble_direct.cli import parse_args
from src.methods.lgc_ensemble_helpers.prepare_data import prepare_data
from src.methods.lgc_ensemble_helpers.train import train
from src.methods.lgc_ensemble_helpers.predict import predict


def main():
    par = parse_args()

    output_model = par["output_model"] or tempfile.mkdtemp(dir="/tmp")
    paths = {
        "output":            par["output"],
        "output_model":      output_model,
        "train_data_aug_dir": f"{output_model}/train_data_aug_dir",
        "model_dir":         f"{output_model}/model_dir",
        "logs_dir":          f"{output_model}/logs",
    }

    if not par["output_model"]:
        import atexit
        atexit.register(lambda: shutil.rmtree(output_model, ignore_errors=True))

    print("\n\n## Preparing data\n")
    prepare_data(par, paths)

    print("\n\n## Training models\n")
    train(par, paths)

    print("\n\n## Generating predictions\n")
    predict(par, paths)


if __name__ == "__main__":
    main()
