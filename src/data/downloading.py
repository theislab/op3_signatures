from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import s3fs

logger = logging.getLogger(__name__)

_S3_BASE = "s3://openproblems-data/resources/task_perturbation_prediction/datasets/neurips-2023-data"

DATASET_URLS: dict[str, str] = {
    "de_train": f"{_S3_BASE}/de_train.h5ad",
    "de_test": f"{_S3_BASE}/de_test.h5ad",
    "id_map": f"{_S3_BASE}/id_map.csv",
}


def _download_from_s3(url: str, file_path: Union[str, Path]) -> None:
    """Download a single file from S3 to a local path, creating parent dirs."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    fs.get(url, str(file_path))


def download_de_train(file_path: Union[str, Path]) -> None:
    """
    Download the NeurIPS-2023 de_train dataset from S3.

    Parameters
    ----------
    file_path : str or Path
        Full path where the file will be saved (e.g.
        ``resources/datasets/neurips-2023-data/de_train.h5ad``).
    """
    url = DATASET_URLS["de_train"]
    logger.info("Downloading de_train from S3 to %s", file_path)
    _download_from_s3(url, file_path)
    logger.info("Download complete: de_train")


def download_de_test(file_path: Union[str, Path]) -> None:
    """
    Download the NeurIPS-2023 de_test dataset from S3.

    Parameters
    ----------
    file_path : str or Path
        Full path where the file will be saved (e.g.
        ``resources/datasets/neurips-2023-data/de_test.h5ad``).
    """
    url = DATASET_URLS["de_test"]
    logger.info("Downloading de_test from S3 to %s", file_path)
    _download_from_s3(url, file_path)
    logger.info("Download complete: de_test")


def download_id_map(file_path: Union[str, Path]) -> None:
    """
    Download the NeurIPS-2023 id_map from S3.

    Parameters
    ----------
    file_path : str or Path
        Full path where the file will be saved (e.g.
        ``resources/datasets/neurips-2023-data/id_map.csv``).
    """
    url = DATASET_URLS["id_map"]
    logger.info("Downloading id_map from S3 to %s", file_path)
    _download_from_s3(url, file_path)
    logger.info("Download complete: id_map")


def download_datasets(
    data_root: Union[str, Path] = "./data/benchmark/resources/datasets/neurips-2023-data",
    skip_existing: bool = True,
) -> dict[str, Path]:
    """
    Download all NeurIPS-2023 datasets to *data_root*.

    Downloads:
    - ``de_train.h5ad``
    - ``de_test.h5ad``
    - ``id_map.csv``

    Parameters
    ----------
    data_root : str or Path
        Directory where files will be saved.
    skip_existing : bool, default True
        If ``True``, skip files that already exist locally.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset name to the local file path.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    targets: dict[str, tuple[Path, callable]] = {
        "de_train": (data_root / "de_train.h5ad", download_de_train),
        "de_test": (data_root / "de_test.h5ad", download_de_test),
        "id_map": (data_root / "id_map.csv", download_id_map),
    }

    paths: dict[str, Path] = {}
    for name, (path, download_fn) in targets.items():
        if skip_existing and path.exists():
            logger.info("Skipping %s – already exists at %s", name, path)
        else:
            download_fn(path)
        paths[name] = path

    return paths
