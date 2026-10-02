"""Record integrity and selector evidence for a notebook candidate export.

Candidates are not promoted to the application automatically. A matching hash
is file-integrity evidence, not proof of performance or training-data identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def active_predictors(pipeline):
    """Read consumed input columns from this project's fitted ColumnTransformer."""
    columns = list(pipeline.feature_names_in_)
    selected = []
    for _, transformer, selector in pipeline.named_steps["prep"].transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        for column in selector:
            name = column if isinstance(column, str) else columns[int(column)]
            if name not in selected:
                selected.append(name)
    return selected


def write_export_metadata(artifact, pipeline, repository_root):
    """Write a paired candidate sidecar; never modify shipped legacy metadata."""
    root = Path(repository_root).resolve()
    artifact = Path(artifact).resolve()
    expected = root / "models" / "fraud_detection_model.candidate.pkl"
    if artifact != expected:
        raise ValueError("Candidate metadata must target the documented candidate path")
    contents = artifact.read_bytes()
    from importlib.metadata import version

    metadata = {
        "artifact": artifact.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "git_blob": hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest(),
        "status": "candidate-unvalidated",
        "feature_schema": list(pipeline.feature_names_in_),
        "active_predictors": active_predictors(pipeline),
        "notebook_sha256": hashlib.sha256(
            (root / "notebooks" / "Fraud_Detection.ipynb").read_bytes()
        ).hexdigest(),
        "serialization_versions": {name: version(name) for name in ("joblib", "scikit-learn", "numpy", "pandas")},
        "training_provenance": "Exported by notebook source; training data identity, training run, and evaluation are not independently verified. No performance metrics are asserted. Review and record these before promotion.",
    }
    sidecar = artifact.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sidecar
