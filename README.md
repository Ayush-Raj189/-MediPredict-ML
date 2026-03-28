# MediPredict-ML

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Ready-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-10b981)](#)

Production-style multi-disease prediction web app built with Streamlit and ensemble machine learning.

## Overview

MediPredict-ML provides risk prediction for three clinical domains:

| Module | Problem | Input Type | Primary Output |
|---|---|---|---|
| Diabetes | Diabetes likelihood | 8 biomarker fields | Class + probability |
| Heart Disease | Cardiac risk likelihood | 13 clinical fields | Class + probability |
| Parkinson's | Parkinson's likelihood | 22 voice features | Class + probability |

## Key Features

| Feature | Details |
|---|---|
| Unified app | One interface for all 3 disease modules |
| ML pipelines | Outlier handling, scaling, feature engineering, tuning |
| Ensemble inference | Voting-based models for robust predictions |
| Pretrained artifacts | Model files included in `models/` |
| Reproducibility | Training notebooks included in `notebooks/` |
| Deployment ready | Configured for Streamlit Community Cloud |

## Project Structure

```text
MediPredict-ML/
├── app.py
├── requirements.txt
├── runtime.txt
├── .streamlit/
│   └── config.toml
├── datasets/
│   ├── diabetes.csv
│   ├── heart_disease_data.csv
│   └── parkinsons.csv
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_imputer.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── parkinsons_model.pkl
│   └── parkinsons_scaler.pkl
└── notebooks/
    ├── Diabetes_Prediction_Streamlit.ipynb
    ├── Heart_Disease_Prediction_Streamlit.ipynb
    └── Parkinsons_Prediction_Streamlit.ipynb
```

## Datasets

| Dataset | Rows | Features | Target |
|---|---:|---:|---|
| PIMA Indians Diabetes | 768 | 8 | Outcome (1/0) |
| Cleveland Heart Disease | 303 | 13 | target (1/0) |
| Oxford Parkinson's | 195 | 22 | status (1/0) |

## ML Workflow

| Stage | Methods Used |
|---|---|
| Preprocessing | Missing handling, domain cleaning, IQR capping |
| Feature Engineering | Interactions and derived features (module-specific) |
| Balancing | SMOTE (where applicable) |
| Scaling | RobustScaler |
| Training | Multiple classifiers + cross-validation |
| Optimization | Randomized search / tuned ensembles |
| Export | `joblib` model artifacts |

## Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.11 |
| Web App | Streamlit |
| ML | scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Serialization | Joblib |
| Notebook Workflow | Jupyter |

## Local Setup

1. Clone the repository.
```bash
git clone https://github.com/Ayush-Raj189/-MediPredict-ML.git
cd ./-MediPredict-ML
```

2. Create and activate a virtual environment.
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies.
```bash
pip install -r requirements.txt
```

4. Run the app.
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

| Setting | Value |
|---|---|
| Repository | `Ayush-Raj189/-MediPredict-ML` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.11` (from `runtime.txt`) |

Deployment steps:
1. Open Streamlit Community Cloud.
2. Create New app.
3. Select the repository and values above.
4. Deploy.

## Notes

- App loads `.pkl` artifacts from `models/`.
- Training datasets are stored in `datasets/`.
- Keep secrets in `.streamlit/secrets.toml` (do not commit).

## Disclaimer

This project is for education and research only. It is not a certified medical device and must not replace professional medical advice, diagnosis, or treatment.
