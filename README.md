# MediPredict-ML

AI-powered Streamlit app for prediction of:
- Diabetes
- Heart Disease
- Parkinson's Disease

Built with ensemble machine learning pipelines, feature engineering, and a responsive clinical-style UI.

## Highlights
- 3 disease prediction modules in one app
- Pre-trained model artifacts included in `models/`
- Reproducible training notebooks in `notebooks/`
- Deployment-ready setup for Streamlit Community Cloud

## Repository
- GitHub: https://github.com/Ayush-Raj189/-MediPredict-ML

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

## Disease Modules

### Diabetes
- Dataset: PIMA Indians Diabetes
- Inputs: 8 clinical biomarkers
- Pipeline includes zero-to-NaN handling, KNN imputation, outlier capping, SMOTE, and ensemble modeling

### Heart Disease
- Dataset: Cleveland Heart Disease
- Inputs: 13 clinical indicators
- Pipeline includes outlier handling, robust scaling, and voting ensemble classifier

### Parkinson's
- Dataset: Oxford Parkinson's
- Inputs: 22 voice biomarkers
- Pipeline includes feature interactions, robust scaling, and ensemble modeling

## ML Pipeline (Summary)
1. Data cleaning and preprocessing
2. Outlier handling (IQR capping)
3. Feature engineering (module-specific)
4. Class balancing (SMOTE where used)
5. Robust scaling
6. Multi-model training and tuning
7. Soft voting ensemble selection
8. Artifact export with `joblib`

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush-Raj189/-MediPredict-ML.git
   cd ./-MediPredict-ML
   ```
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud Deployment
1. Open https://share.streamlit.io and create a new app.
2. Select repository: `Ayush-Raj189/-MediPredict-ML`
3. Branch: `main`
4. Main file path: `app.py`
5. Python version: `3.11` (from `runtime.txt`)
6. Deploy.

## Dependencies
Defined in `requirements.txt`:
- streamlit
- numpy
- pandas
- scikit-learn
- joblib
- xgboost

## Notes
- Model files are loaded from `models/`.
- Datasets are in `datasets/`.
- Keep `.streamlit/secrets.toml` local and never commit secrets.

## Disclaimer
This project is for educational and research use only. It is not a medical device and should not replace professional clinical judgment.
