import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="NN retraining with pseudolabels prediction")
    parser.add_argument("--de_train", required=True,                        help="Path to de_train.h5ad")
    parser.add_argument("--id_map",   required=True,                        help="Path to id_map.csv")
    parser.add_argument("--embeddings", default=None,                         help="Path to op3_emb.pkl (required when --embedding_type is lpm or fp)")
    parser.add_argument("--embedding_type", choices=["lpm", "fp", "none"], default="none", help="Type of external embedding to concatenate: lpm, fp, or none (default: none)")
    parser.add_argument("--embedding_layer", choices=["concat", "fixed", "trainable"], default="concat", help="How to integrate the external embedding: concat, fixed, or trainable (default: concat)")
    parser.add_argument("--layer",    default="clipped_sign_log10_pval",    help="AnnData layer to use")
    parser.add_argument("--output",   required=True,                        help="Path to output .h5ad file")
    parser.add_argument("--reps",         type=int, default=10,   help="Number of repetitions")
    parser.add_argument("--use_fp_dense", action="store_true",    help="Project fingerprints through a Dense layer before concatenation")
    args = parser.parse_args()
    return vars(args)
