# MediPredict ML

A Streamlit web app for multi-disease prediction:
- Diabetes
- Heart Disease
- Parkinson's

## Project structure
- `app.py` - Streamlit application entry point
- `models/` - trained model artifacts (`.pkl`)
- `datasets/` - CSV datasets used during model development
- `notebooks/` - training and evaluation notebooks

## Local run
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Streamlit:
   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud deployment
1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Use these settings:
   - Main file path: `app.py`
   - Python version: `3.11` (from `runtime.txt`)
4. Deploy.

## Notes
- Model files are loaded from `models/`.
- Keep `.streamlit/secrets.toml` local and do not commit secrets.
