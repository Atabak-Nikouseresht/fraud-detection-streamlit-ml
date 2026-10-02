from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "fraud_detection_model.pkl"
METADATA = ROOT / "models" / "artifact-metadata.json"
EXPECTED_VERSIONS = {
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "scikit-learn": "1.6.1",
    "joblib": "1.4.2",
    "streamlit": "1.54.0",
    "Pillow": "12.3.0",
}


def verify_artifact() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    contents = ARTIFACT.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != metadata["sha256"]:
        raise SystemExit(
            f"Artifact SHA-256 mismatch: expected {metadata['sha256']}, got {digest}"
        )
    if metadata["artifact"] != ARTIFACT.relative_to(ROOT).as_posix():
        raise SystemExit("Artifact path does not match artifact metadata")
    blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
    if metadata["git_blob"] != blob:
        raise SystemExit("Artifact Git blob does not match artifact metadata")
    # Deserialize only after confirming the trusted committed artifact checksum.
    import joblib
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.export_metadata import active_predictors

    pipeline = joblib.load(ARTIFACT)
    if list(pipeline.feature_names_in_) != metadata["feature_schema"]:
        raise SystemExit("Artifact accepted feature schema does not match metadata")
    if active_predictors(pipeline) != metadata["active_predictors"]:
        raise SystemExit("Artifact active predictor selectors do not match metadata")


def main() -> int:
    errors = []
    actual_python = ".".join(map(str, sys.version_info[:3]))
    if actual_python != "3.11.16":
        errors.append(f"Python: expected 3.11.16, got {actual_python}")

    from importlib.metadata import PackageNotFoundError, version

    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            errors.append(f"{package}: expected {expected}, got {actual}")

    try:
        verify_artifact()
    except (OSError, KeyError, json.JSONDecodeError, SystemExit) as error:
        errors.append(f"Artifact integrity: {error}")

    if errors:
        print("Environment check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("Environment versions and model artifact SHA-256 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
