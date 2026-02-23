import argparse

from src.data.downloading import download_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NeurIPS-2023 benchmark datasets from S3."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Directory where files will be saved.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_datasets(data_root=args.data_root, skip_existing=not args.force)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
