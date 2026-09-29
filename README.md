# Fraud Detection Demo — scikit-learn and Streamlit

A local fraud-classification demonstration that connects a saved scikit-learn pipeline to a Streamlit form. It shows a model-to-interface workflow for transaction predictions.

## What is included

- `app/fraud_app.py` — Streamlit interface for one-transaction predictions.
- `models/fraud_detection_model.pkl` — serialized model used by the app.
- `notebooks/Fraud_Detection.ipynb` — exploratory analysis and model-training workflow.
- `requirements.txt` — pinned application runtime dependencies.
- `requirements-notebook.txt` — runtime dependencies plus notebook and visualization tools.

The notebook's saved output describes 6,362,620 observations and 8,213 fraud cases (about 0.13%). These are the notebook's recorded dataset counts; the source CSV is not included.

## Run the app

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/fraud_app.py
```

The app loads the model relative to its own file. Enter the transaction fields and select **Predict** to see the model's class label.

## Run the notebook

The CSV is not included. Obtain it from the [dataset source](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset), follow its terms, and save it as `data/AIML Dataset.csv` in the repository root.

From the repository root, install the notebook environment and open the notebook. Its path helper locates the repository root whether Jupyter starts here or in `notebooks/`:

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/Fraud_Detection.ipynb
```

The notebook reports a clear error if the separately obtained CSV is missing. The dataset is intentionally not committed.

## Checks

Run the lightweight tests without the Kaggle dataset:

```bash
python -m unittest discover -s tests -v
```

## Method and limits

The notebook uses scaling, one-hot encoding, and a class-weighted logistic-regression classifier. The repository does not establish a validated operating threshold, calibrated probability, or real-world fraud-detection performance; treat the interface as a demonstration, not a decision recommendation.

## License and author

MIT License.

Atabak Nikouseresht — MSc Applied Economics and Markets, University of Bologna · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
