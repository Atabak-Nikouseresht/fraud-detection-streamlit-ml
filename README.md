# Fraud Detection Demo — scikit-learn and Streamlit

A small, local fraud-classification demonstration that connects a saved scikit-learn model pipeline to a Streamlit form. It illustrates the path from transaction features to a model prediction; it is not a deployed fraud service or a validated operational risk control.

## What is included

- `app/fraud_app.py` — loads the committed model artifact and collects one transaction's features for a prediction.
- `models/fraud_detection_model.pkl` — serialized model used by the app.
- `notebooks/Fraud_Detection.ipynb` — exploratory analysis and model-training workflow.
- `requirements.txt` — pinned package versions.

The notebook's saved output describes 6,362,620 observations and 8,213 fraud cases (about 0.13%). These are the notebook's recorded dataset counts; the source CSV is not included.

## Run the app

From the repository root:

```bash
git clone https://github.com/Atabak-Nikouseresht/fraud-detection-streamlit-ml.git
cd fraud-detection-streamlit-ml
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/fraud_app.py
```

The app loads `models/fraud_detection_model.pkl` relative to its own file, so launch it from the repository root or another working directory. Enter the transaction fields and select **Predict** to see the model's class label.

## Reproduce the notebook

The notebook expects the Kaggle CSV at `data/AIML Dataset.csv` relative to the notebook's working directory (`notebooks/`). The dataset is not included; obtain it from the [dataset source](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) and follow its terms. The notebook also imports Jupyter and Seaborn, which are not currently listed as direct requirements; install those in the environment before running the notebook.

## Method and limits

The notebook uses a preprocessing pipeline with scaling and one-hot encoding and a class-weighted logistic-regression classifier. The repository contains no test suite or CI, and the committed documentation does not establish a validated operating threshold, calibrated probability, or real-world fraud-detection performance. Treat the interface output as a demonstration, not a decision recommendation.

## License and author

MIT License.

Atabak Nikouseresht — MSc Applied Economics and Markets, University of Bologna · [GitHub](https://github.com/Atabak-Nikouseresht) · [LinkedIn](https://linkedin.com/in/atabak-nikouseresht)
