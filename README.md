# Fraud Detection Demo — scikit-learn and Streamlit

[![CI](https://github.com/Atabak-Nikouseresht/fraud-detection-streamlit-ml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Atabak-Nikouseresht/fraud-detection-streamlit-ml/actions/workflows/ci.yml)

A local fraud-classification demonstration that connects a saved scikit-learn pipeline to a Streamlit form. It shows a model-to-interface workflow for transaction predictions.

## What is included

- `app/fraud_app.py` — Streamlit interface for one-transaction predictions.
- `models/fraud_detection_model.pkl` — serialized model used by the app.
- `notebooks/Fraud_Detection.ipynb` — exploratory analysis and model-training workflow.
- `requirements.txt` — pinned application runtime dependencies.
- `requirements-notebook.txt` — runtime dependencies plus notebook and visualization tools.
- `pyproject.toml` and `uv.lock` — the Python 3.11.16 environment and resolved dependency lock; notebook packages are an optional extra.
- `requirements-lock.txt` and `requirements-notebook-lock.txt` — hash-checked pip exports of the same lock, for users who do not use uv.

Earlier notebook outputs recorded 6,362,620 observations and 8,213 fraud cases (about 0.13%). These are historical dataset counts, not recomputed in this checkout. All saved notebook outputs and execution counts have now been cleared to avoid presenting the old workflow as evidence for the corrected source; the source CSV is not included.

## Run the app

Python 3.11.16 is pinned in `.python-version`. The recommended install uses the committed lock:

```bash
uv sync --locked --no-dev
uv run streamlit run app/fraud_app.py
```

`requirements.txt` remains the human-readable list of serialization-sensitive direct runtime pins. For pip-only installs, use `python -m pip install --require-hashes -r requirements-lock.txt`; install notebook tooling with `python -m pip install --require-hashes -r requirements-notebook-lock.txt` instead. The unexpanded `requirements-notebook.txt` is the direct-pin list, not the transitive lock.

The app loads the model relative to its own file. Enter the transaction fields and select **Predict** to see the model's class label.

## Run the notebook

The CSV is not included. Obtain it from the [dataset source](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset), and follow the terms shown on the current Kaggle page (this repository does not assert a separate license). Save it as `data/AIML Dataset.csv` in the repository root. Validate its expected 11 columns and binary `isFraud` target before notebook use:

```bash
uv run python scripts/validate_dataset.py
```

From the repository root, install the locked notebook extra and open the notebook. Its path helper locates the repository root whether Jupyter starts here or in `notebooks/`:

```bash
uv sync --locked --extra notebook
uv run jupyter notebook notebooks/Fraud_Detection.ipynb
```

The notebook reports a clear error if the separately obtained CSV is missing. The dataset is intentionally not committed. No download is performed by setup or CI.

## Checks

Run the lightweight tests without the Kaggle dataset; GitHub Actions installs the locked runtime, checks dependencies and artifact integrity, exercises a synthetic eight-feature training smoke test, then runs the suite on pushes and pull requests:

```bash
python -m unittest discover -s tests -v
```

For a local locked CI-equivalent run: `uv sync --locked --no-dev`, `uv pip check`, `uv run python scripts/check_environment.py`, `uv run python scripts/synthetic_training_smoke.py`, and `uv run python -m unittest discover -s tests -v`.

## Method and limits

The notebook uses scaling, one-hot encoding, and a class-weighted logistic-regression classifier with deterministic split/model seeds. The corrected notebook selects eight predictors: transaction type, amount, four raw balances, and two derived balance-difference features. The app constructs and the legacy artifact accepts those eight columns, but the legacy fitted ColumnTransformer consumes only six predictors (`type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`); it drops `balanceDiffOrig` and `balanceDiffDest` via `remainder="drop"`. Accepted input schema is not the same as active predictors. Negative differences are arithmetic observations, not automatic evidence of invalid data or fraud. Evaluation reports ROC-AUC, average precision (a precision-recall ranking summary), fraud-class precision/recall/F1, and a confusion matrix at the default 0.5 threshold; accuracy alone is not meaningful for this extreme class imbalance. The committed `models/fraud_detection_model.pkl` predates the corrected notebook methodology and was intentionally left unchanged; `models/artifact-metadata.json` records its known SHA-256 and feature schema but explicitly does not claim training provenance, package versions, or metrics. The notebook has not been retrained because the full source dataset is unavailable. The synthetic smoke test only checks that the pipeline executes; its metrics are not empirical model results. The repository does not establish a validated operating threshold, calibrated probability, or real-world fraud-detection performance; treat the interface as a demonstration, not a decision recommendation.

## Export and artifact evidence

The shipped model remains byte-for-byte unchanged (SHA-256 `2686468a5bf063d461f21ee617d52b1707e7f06ecac1e62290f6d7a8447b7a4e`). Its exact training date, source-data identity, original environment, split seed, and empirical performance are unknown. Its fitted structure establishes preprocessing and classifier configuration, not a verified training run. No saved notebook evaluation validates this legacy artifact or the corrected source.

The final notebook cell imports `joblib` through the model-import cell and exports to **`models/fraud_detection_model.candidate.pkl`**, never the shipped path. `scripts/export_metadata.py` writes **`models/fraud_detection_model.candidate.json`** with the candidate SHA-256, Git blob identity, accepted schema, actual fitted selectors, notebook source hash, installed serialization versions, and an explicit unvalidated status. This links an export to its file and source; it does not prove training-data identity or performance. Both candidate files are ignored by Git. If either export/metadata write fails, do not use or promote the candidate.

Promotion is a separate, deliberate maintenance action: review and record the dataset identity, training run and held-out evaluation; replace the shipped artifact and update `models/artifact-metadata.json` together with the matching hash, Git blob, accepted schema, actual active predictors and truthful provenance/status. Then run `scripts/check_environment.py` and the complete tests before committing. The validation rejects mismatched bytes, blob identity, accepted schema or fitted selectors. No promotion or retraining was performed in this remediation.

## License and author

MIT License.

Atabak Nikouseresht — MSc Applied Economics and Markets, University of Bologna · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
