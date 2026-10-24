# Multiple Disease Prediction – Run & Deploy Guide

Streamlit app with 3 models: **Diabetes** (linear SVM), **Heart disease** (logistic regression), **Parkinson's** (linear SVM).
Educational/hackathon project – not a medical device.

## Project layout
```
app.py              Streamlit UI (loads the 3 models, 3 pages)
train_models.py     Retrains the models from dataset/ -> saved_models/
saved_models/       *.sav pickled models (already regenerated for modern scikit-learn)
dataset/            diabetes.csv, heart.csv, parkinsons.csv
notebooks/          original training notebooks
requirements.txt
```

## 1. Run locally

Needs Python 3.9 – 3.13 (check with `python --version`).

```bash
# in the project folder
python -m venv venv

# activate:  Windows  ->  venv\Scripts\activate
#            Mac/Linux -> source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Browser opens at http://localhost:8501.

If you see a scikit-learn "version mismatch" warning or model-loading error, regenerate the models for your machine:
```bash
python train_models.py
```

### VS Code
1. Open the project folder (File > Open Folder). Install the *Python* extension.
2. `Ctrl+Shift+P` > **Python: Select Interpreter** > pick the `venv` one.
3. Open the integrated terminal (`Ctrl+``) and run `streamlit run app.py`.
   (Don't use the ▶ Run button — Streamlit apps must be started with `streamlit run`.)

### Spyder
Spyder's F5 doesn't start Streamlit. Instead, in the IPython console (with the folder as working directory):
```
!streamlit run app.py
```
or use an Anaconda Prompt / terminal: `cd` to the folder and run `streamlit run app.py`.
Tip: install the packages into the same environment Spyder uses (`pip install -r requirements.txt`).

### Anaconda
```bash
conda create -n health python=3.11 -y
conda activate health
pip install -r requirements.txt
streamlit run app.py
```

## 2. Deploy (free options)

### A. Streamlit Community Cloud (easiest)
1. Create a GitHub repo and push this folder (`app.py`, `requirements.txt`, `saved_models/`, `dataset/` etc. at the repo root).
   ```bash
   git init && git add . && git commit -m "Multiple disease prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/multiple-disease-prediction.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `app.py` > **Deploy**. You get a public `*.streamlit.app` URL.

### B. Hugging Face Spaces
New Space > SDK **Streamlit** > upload the files (app file must be `app.py`; `requirements.txt` is picked up automatically).

### C. Render / Railway (Docker-free)
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Notes
- Keep `saved_models/` in the repo; the app loads them at startup.
- The original `requirements.txt` pinned `streamlit==1.29.0` which breaks with current `pyarrow`; this version uses ranges instead.
- Fields are free-text numbers; empty/invalid input now shows an error instead of crashing.
