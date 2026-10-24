# Multiple Disease Prediction

A Streamlit educational demo that provides sample-based and interactive predictions for diabetes, heart disease, and Parkinson's disease using three trained machine-learning models. Their test accuracies are 77% for diabetes, 80% for heart disease, and 87% for Parkinson's disease.

## Features

- Sample buttons for trying example inputs
- Charts for exploring data and prediction results
- An explainability tab for understanding predictions
- Light, dark, and system theme options

## Project structure

```text
.
├── .gitignore                                  # Ignored local and generated files
├── app.py                                      # Streamlit application
├── train_models.py                             # Model training script
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project overview and quick start
├── .streamlit/
│   └── config.toml                             # Light and dark theme settings
├── theme_sync/
│   └── index.html                              # Browser theme synchronization
├── scripts/
│   ├── verify_help.py                          # App and prediction checks
│   ├── verify_samples.py                       # Dataset and sample checks
│   └── verify_theme.py                         # Theme helper checks
├── docs/
│   └── RUN_AND_DEPLOY.md                       # Run and deployment guide
├── notebooks/
│   ├── Multiple disease prediction system - diabetes.ipynb  # Diabetes training
│   ├── Multiple disease prediction system - heart.ipynb     # Heart training
│   └── Multiple disease prediction system - Parkinsons.ipynb # Parkinson's training
├── dataset/
│   ├── diabetes.csv                            # Diabetes dataset
│   ├── heart.csv                               # Heart disease dataset
│   └── parkinsons.csv                          # Parkinson's dataset
└── saved_models/
    ├── diabetes_model.sav                      # Diabetes model
    ├── heart_disease_model.sav                 # Heart disease model
    └── parkinsons_model.sav                    # Parkinson's model
```

## Quick start (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

See [docs/RUN_AND_DEPLOY.md](docs/RUN_AND_DEPLOY.md) for setup and deployment details.

Educational demo, not a medical diagnosis.
