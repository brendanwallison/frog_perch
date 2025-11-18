# utils/download_perch.py

import kagglehub
from pathlib import Path

def get_perch_savedmodel_path() -> str:
    """
    Ensures Perch v2 is downloaded via KaggleHub and returns the SavedModel directory.
    KaggleHub manages caching automatically, so repeated calls are cheap.
    """
    path = kagglehub.model_download(
        "google/bird-vocalization-classifier/tensorFlow2/perch_v2"
    )
    return str(Path(path))
