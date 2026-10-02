from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}
TARGET_COLUMN = "isFraud"


def validate_dataset(path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, [
            f"Dataset not found: {path}",
            "Obtain the CSV from https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset, "
            "review the current Kaggle page terms, then place it at data/AIML Dataset.csv.",
        ]

    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return False, [f"Could not read CSV header: {error}"]

    missing = sorted(EXPECTED_COLUMNS - columns)
    unexpected = sorted(columns - EXPECTED_COLUMNS)
    problems = []
    if missing:
        problems.append(f"Missing required columns: {', '.join(missing)}")
    if unexpected:
        problems.append(f"Unexpected columns: {', '.join(unexpected)}")
    if problems:
        return False, problems

    try:
        target = pd.read_csv(path, usecols=[TARGET_COLUMN])[TARGET_COLUMN]
    except (OSError, pd.errors.ParserError, ValueError, UnicodeDecodeError) as error:
        return False, [f"Could not read target column {TARGET_COLUMN}: {error}"]
    if target.empty:
        problems.append("Dataset contains no rows.")
    if target.isna().any():
        problems.append(f"Target column {TARGET_COLUMN} contains missing values.")
    labels = set(target.dropna().unique())
    if not labels.issubset({0, 1}):
        problems.append(f"Target column {TARGET_COLUMN} must contain only 0 and 1; found {sorted(labels)}")
    if labels != {0, 1}:
        problems.append(f"Target column {TARGET_COLUMN} must contain both classes 0 and 1.")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Kaggle CSV schema before notebook use.")
    parser.add_argument("path", nargs="?", type=Path, default=Path("data/AIML Dataset.csv"))
    args = parser.parse_args()
    valid, problems = validate_dataset(args.path)
    if valid:
        print(f"Dataset schema and binary target validated: {args.path}")
        return 0
    print("Dataset validation failed:")
    for problem in problems:
        print(f"- {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
