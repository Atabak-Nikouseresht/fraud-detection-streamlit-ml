# Fraud Detection System with Machine Learning and Streamlit

This repository presents an end-to-end fraud detection system built using Python, Scikit-Learn, and Streamlit. The project demonstrates the complete workflow typically used in FinTech and risk analytics: data preprocessing, modeling, handling extreme class imbalance, exporting a trained model, and deploying an interactive application for real-time predictions.

The objective is to provide a clean, production-oriented example of how machine learning pipelines are implemented for fraud detection tasks.

---

## Project Overview

This project includes:

- A reproducible machine learning pipeline for fraud classification  
- A trained model exported in `.pkl` format  
- A Streamlit application for real-time fraud prediction  
- A Jupyter Notebook with exploratory data analysis and model training  
- A structured, professional repository layout suitable for portfolio and deployment  

The dataset used in this project is highly imbalanced, reflecting real-world financial fraud patterns.

---

## Repository Structure

fraud-detection-streamlit-ml/
│
├─ app/
│ └─ fraud_app.py # Streamlit application
│
├─ notebooks/
│ └─ Fraud_Detection.ipynb # EDA and model training notebook
│
├─ models/
│ └─ fraud_detection_model.pkl # Trained machine learning model
│
├─ requirements.txt # Python dependencies
└─ README.md # Documentation


---

## Dataset

The dataset is not included in this repository due to its large size.  
Download it from Kaggle:

https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

To run the notebook, download the dataset and place it in a local directory.  
Update the file path inside the notebook accordingly.

---

## Machine Learning Pipeline

The model was developed using:

- Logistic Regression classifier  
- ColumnTransformer for preprocessing  
  - StandardScaler for numerical features  
  - OneHotEncoder for categorical features  
- Handling extreme class imbalance with `class_weight="balanced"`  
- A complete Scikit-Learn Pipeline  
- Model export using `joblib`  

The notebook includes exploratory data analysis, class distribution inspection, feature preparation, model training, and evaluation.

---

## Running the Streamlit Application

### 1. Clone the repository

git clone https://github.com/Atabak-Nikouseresht/fraud-detection-streamlit-ml.git

cd fraud-detection-streamlit-ml


### 2. Install dependencies

pip install -r requirements.txt


### 3. Launch the Streamlit application

cd app
streamlit run fraud_app.py


---

## Running the Jupyter Notebook

1. Download the dataset from Kaggle.  
2. Place the CSV file in a local directory.  
3. Open the notebook:

jupyter notebook notebooks/Fraud_Detection.ipynb


4. Update the dataset path inside the notebook if necessary.

---

## Technologies Used

- Python 3  
- Scikit-Learn  
- Pandas  
- NumPy  
- Streamlit  
- Joblib  
- Jupyter Notebook  

---

## Future Improvements

Potential enhancements:

- Advanced feature engineering  
- Testing models such as Random Forest, XGBoost, and LightGBM  
- Hyperparameter tuning  
- Threshold optimization using precision–recall curves  
- SHAP-based model explainability  
- Containerization and cloud deployment  

---

## License

This project is released under the MIT License.

---

## Contact

GitHub: https://github.com/Atabak-Nikouseresht  
Email: atabak.nikouseresht@gmail.com
