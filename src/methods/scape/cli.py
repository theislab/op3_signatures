import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SCAPE prediction")
    parser.add_argument("--de_train",         required=True,                     help="Path to de_train.h5ad")
    parser.add_argument("--id_map",           required=True,                     help="Path to id_map.csv")
    parser.add_argument("--layer",            default="clipped_sign_log10_pval", help="AnnData layer to use")
    parser.add_argument("--output",           required=True,                     help="Path to output .h5ad file")
    parser.add_argument("--output_model",     default=None,                      help="Directory to save model (optional)")
    parser.add_argument("--cell",             default="NK cells",                help="Cell type to use")
    parser.add_argument("--epochs",           type=int, default=300,               help="Training epochs (base model)")
    parser.add_argument("--epochs_enhanced",  type=int, default=800,               help="Training epochs (enhanced model)")
    parser.add_argument("--n_genes",          type=int, default=64,              help="Number of top genes (base model)")
    parser.add_argument("--n_genes_enhanced", type=int, default=256,              help="Number of top genes (enhanced model)")
    parser.add_argument("--n_drugs",          type=int, default=None,            help="Max number of drugs (None = all)")
    parser.add_argument("--min_n_top_drugs",  type=int, default=50,              help="Minimum number of top drugs")
    args = parser.parse_args()
    return vars(args)
