import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LGC ensemble prediction (direct single-step)")
    parser.add_argument("--de_train",     required=True,                              help="Path to de_train.h5ad")
    parser.add_argument("--id_map",       required=True,                              help="Path to id_map.csv")
    parser.add_argument("--layer",        default="clipped_sign_log10_pval",          help="AnnData layer to use")
    parser.add_argument("--schemes",      nargs="+", default=["initial", "light", "heavy"],    help="Ensemble schemes to use")
    parser.add_argument("--models",       nargs="+", default=["LSTM", "GRU", "Conv"],    help="Model types to train")
    parser.add_argument("--epochs",       type=int,  default=250,                       help="Number of training epochs")
    parser.add_argument("--kf_n_splits",  type=int,  default=5,                       help="Number of cross-validation folds")
    parser.add_argument("--output",       required=True,                              help="Path to output .h5ad file")
    parser.add_argument("--output_model", default=None,                               help="Directory to save models (optional)")
    args = parser.parse_args()
    return vars(args)
