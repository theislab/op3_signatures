import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LGC ensemble — prediction step")
    parser.add_argument("--id_map",             required=True,   help="Path to id_map.csv")
    parser.add_argument("--train_data_aug_dir",  required=True,   help="Directory with prepared features")
    parser.add_argument("--model_files",         nargs="+", required=True, help="Paths to trained model files (.pt)")
    parser.add_argument("--output",              required=True,   help="Path to output .h5ad file")
    args = parser.parse_args()
    return vars(args)
