# NoCode ML Studio - Streamlit App

This is a scaffolded Streamlit application for a NoCode/Low-Code ML studio. It supports data upload, basic cleaning, EDA, visualizations, and running several baseline models for regression and classification.

Setup (Windows PowerShell):

1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements_streamlit.txt
```

3. Run the app

```powershell
streamlit run streamlit_app/app.py
```

Notes
- This is an initial scaffold. Time series models, extensive hyperparameter tuning, VIF-based filtering, and hypothesis testing are planned in subsequent iterations.
