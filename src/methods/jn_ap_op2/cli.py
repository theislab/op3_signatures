import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="JN-AP-OP2 deep learning prediction")
    parser.add_argument("--de_train",         required=True,               help="Path to de_train.h5ad")
    parser.add_argument("--id_map",           required=True,               help="Path to id_map.csv")
    parser.add_argument("--layer",            default="clipped_sign_log10_pval",   help="AnnData layer to use")
    parser.add_argument("--output",           required=True,               help="Path to output .h5ad file")
    parser.add_argument("--n_replica",        type=int, default=10,         help="Number of model replicas")
    parser.add_argument("--submission_names", nargs="+", default=["dl40", "dl200"], help="Submission strategy names")
    args = parser.parse_args()
    return vars(args)
