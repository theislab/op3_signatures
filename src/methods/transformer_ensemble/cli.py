import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer ensemble prediction")
    parser.add_argument("--de_train",          required=True,                   help="Path to de_train.h5ad")
    parser.add_argument("--id_map",            required=True,                   help="Path to id_map.csv")
    parser.add_argument("--layer",             default="clipped_sign_log10_pval",       help="AnnData layer to use")
    parser.add_argument("--output",            required=True,                   help="Path to output .h5ad file")
    parser.add_argument("--output_model",      default=None,                    help="Directory to save models (optional)")
    parser.add_argument("--num_train_epochs",  type=int, default=20000,            help="Number of training epochs")
    parser.add_argument("--early_stopping",    type=int, default=5000,          help="Early stopping patience")
    parser.add_argument("--batch_size",        type=int, default=32,            help="Batch size")
    parser.add_argument("--d_model",           type=int, default=128,           help="Transformer model dimension")
    args = parser.parse_args()
    return vars(args)
