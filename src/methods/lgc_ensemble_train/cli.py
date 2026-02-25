import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LGC ensemble — model training step")
    parser.add_argument("--train_data_aug_dir", required=True,                                   help="Directory with prepared features")
    parser.add_argument("--scheme",             required=True, choices=["initial", "light", "heavy"], help="Ensemble scheme")
    parser.add_argument("--model",              required=True, choices=["LSTM", "GRU", "Conv"],   help="Model architecture")
    parser.add_argument("--fold",               type=int, required=True,                          help="Cross-validation fold index")
    parser.add_argument("--model_file",         required=True,                                    help="Path to save trained model (.pt)")
    parser.add_argument("--log_file",           required=True,                                    help="Path to save training log (.json)")
    args = parser.parse_args()
    return vars(args)
