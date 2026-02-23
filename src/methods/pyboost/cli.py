import argparse

PREDICTOR_CHOICES = ["py_boost", "ridge_recommender", "knn_recommender", "predict_extratrees"]


def parse_args():
    parser = argparse.ArgumentParser(description="Py-boost predicting t-scores")
    parser.add_argument("--de_train", required=True, help="Path to de_train.h5ad")
    parser.add_argument("--layer", default="clipped_sign_log10_pval", help="AnnData layer to use")
    parser.add_argument("--id_map", required=True, help="Path to id_map.csv")
    parser.add_argument(
        "--predictor_names",
        nargs="+",
        default=["py_boost"],
        choices=PREDICTOR_CHOICES,
        help="Which predictor(s) to use",
    )
    parser.add_argument("--output", required=True, help="Path to output .h5ad file")
    args = parser.parse_args()
    return vars(args)
