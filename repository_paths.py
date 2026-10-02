"""Resolve repository data and model paths independently of the working directory."""

from pathlib import Path


REPOSITORY_MARKERS = ("repository_paths.py", "app/fraud_app.py")


def find_repository_root(start):
    """Find this repository from a file or directory beneath its root."""
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if all((candidate / marker).is_file() for marker in REPOSITORY_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not locate the repository root from {start}. "
        "Run from within the cloned repository."
    )


def resolve_dataset_path(start):
    """Return the documented local dataset path or explain how to provide it."""
    dataset_path = find_repository_root(start) / "data" / "AIML Dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Fraud dataset not found at {dataset_path}. Download the dataset "
            "separately and place it at <repository-root>/data/AIML Dataset.csv; "
            "the dataset is not included in this repository."
        )
    return dataset_path


def resolve_model_path(start):
    """Return the repository's committed model artifact path."""
    return find_repository_root(start) / "models" / "fraud_detection_model.pkl"
