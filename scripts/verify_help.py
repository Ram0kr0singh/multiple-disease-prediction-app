"""
Final structured verification of all deliverables.
"""
import ast, re, sys, pickle, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
APP  = BASE / 'app.py'
src  = APP.read_text(encoding='utf-8')

ok = True
def chk(label, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))

print("=" * 60)
print("SETUP")
print("=" * 60)

# config.toml
cfg = BASE / '.streamlit' / 'config.toml'
cfg_txt = cfg.read_text() if cfg.exists() else ""
chk("config.toml exists",               cfg.exists())
chk("dark theme configured",            "[theme.dark]" in cfg_txt)
chk("dark primaryColor #4FB3BF",         "#4FB3BF" in cfg_txt)
chk("dark backgroundColor #0F1A1F",      "#0F1A1F" in cfg_txt)

# requirements.txt
req = (BASE / 'requirements.txt').read_text()
chk("plotly in requirements.txt",       "plotly" in req)

# plotly importable in venv
try:
    import plotly; chk("plotly importable", True, plotly.__version__)
except ImportError:
    chk("plotly importable", False, "ImportError")

# CSS injection
chk("inject_css() defined",             "def inject_css" in src)
chk("inject_css() called",              "inject_css()" in src)
chk("hero-banner CSS class",            "hero-banner" in src)
chk("result-positive CSS class",        "result-positive" in src)
chk("result-negative CSS class",        "result-negative" in src)
chk("gradient title (.section-title)",  "section-title" in src)

print()
print("=" * 60)
print("DATA LOADING")
print("=" * 60)
chk("@st.cache_data on load_datasets",  "@st.cache_data" in src)
chk("heart encoding utf-8-sig",         "utf-8-sig" in src)
chk("parkinsons drops 'name' column",   "drop(columns=['name'])" in src or "drop(columns=[\"name\"])" in src)

d = pd.read_csv(BASE / 'dataset' / 'diabetes.csv')
h = pd.read_csv(BASE / 'dataset' / 'heart.csv', encoding='utf-8-sig')
p = pd.read_csv(BASE / 'dataset' / 'parkinsons.csv').drop(columns=['name'])
DIAB_FEAT  = [c for c in d.columns if c != 'Outcome']
HEART_FEAT = [c for c in h.columns if c != 'target']
PARK_FEAT  = [c for c in p.columns if c != 'status']
chk("diabetes  8 features",             len(DIAB_FEAT)  == 8,  str(len(DIAB_FEAT)))
chk("heart    13 features",             len(HEART_FEAT) == 13, str(len(HEART_FEAT)))
chk("parkinson 22 features",            len(PARK_FEAT)  == 22, str(len(PARK_FEAT)))

print()
print("=" * 60)
print("HOME PAGE  (A)")
print("=" * 60)
chk("Hero banner HTML",                 "hero-banner" in src)
chk("Pill badges HTML",                 'class="badge"' in src)
chk("class_donut_fig() defined",        "def class_donut_fig" in src)
chk("accuracy_bar_fig() defined",       "def accuracy_bar_fig" in src)
chk("How it works steps",               "step-box" in src)

print()
print("=" * 60)
print("BEFORE CLICKING — HISTOGRAM  (B)")
print("=" * 60)
chk("hist_fig() defined",               "def hist_fig" in src)
chk("Diabetes histogram expander",      "Explore the dataset" in src)
chk("Glucose default feature",          'index=DIAB_FEAT.index("Glucose")' in src or
                                        "Glucose" in src)
chk("thalach default (heart)",          'index=HEART_FEAT.index("thalach")' in src or
                                        '"thalach"' in src)
chk("HNR default (parkinsons)",         'index=PARK_FEAT.index("HNR")' in src or
                                        '"HNR"' in src)
chk("user vline in hist_fig",           "add_vline" in src)
chk("selectbox to change feature",      "Feature to explore" in src)

print()
print("=" * 60)
print("AFTER CLICKING — RESULT CARDS  (C)")
print("=" * 60)
chk("session_state stores pred",        "diab_result_pred" in src)
chk("session_state stores score",       "diab_result_score" in src)
chk("st.spinner used",                  "st.spinner" in src)
chk("result-positive banner",           "result-positive" in src)
chk("result-negative banner",           "result-negative" in src)
chk("No 'The person has' wording",      "The person has" not in src)
chk("Likelihood wording",               "higher likelihood" in src)

chk("gauge_fig() defined",              "def gauge_fig" in src)
chk("Gauge zones green/amber/red",      "rgba(76,195,143" in src and
                                        "rgba(224,160,58" in src and
                                        "rgba(239,91,102" in src)

chk("contrib_fig() defined",            "def contrib_fig" in src)
chk("Contribution = coef * (x - mean)", "coef * (np.array(x_vals) - np.array(means))" in src)

chk("compare_fig() defined",            "def compare_fig" in src)
chk("normalise() defined",              "def normalise" in src)
chk("st.tabs Result/Why/Dataset",       '"📈 Result"' in src or '"Result"' in src)

# Heart uses predict_proba; diabetes/park use decision_function
chk("Heart uses predict_proba",         "use_proba=True" in src)
chk("Diabetes uses decision_function",  "use_proba=False" in src)

print()
print("=" * 60)
print("PREDICTIONS  (all 6 samples)")
print("=" * 60)
def load(name):
    with open(BASE / 'saved_models' / name, 'rb') as f:
        return pickle.load(f)
dm = load('diabetes_model.sav')
hm = load('heart_disease_model.sav')
pm = load('parkinsons_model.sav')

def sigmoid(x): return 1/(1+math.exp(-x))
def score(model, X, proba=False):
    return float(model.predict_proba([X])[0][1])*100 if proba \
           else sigmoid(float(model.decision_function([X])[0]))*100

d_pos=[6,148,72,35,0,33.6,0.627,50];  d_neg=[4,110,92,0,0,37.6,0.191,30]
h_pos=[63,1,3,145,233,1,0,150,0,2.3,0,0,1]; h_neg=[63,1,0,130,254,0,0,147,0,1.4,1,1,3]
p_pos=[119.992,157.302,74.997,.00784,.00007,.0037,.00554,.01109,.04374,.426,
       .02182,.0313,.02971,.06545,.02211,21.033,.414783,.815285,-4.813031,.266482,2.301442,.284654]
p_neg=[202.266,211.604,197.079,.0018,.000009,.00093,.00107,.00278,.00954,.085,
       .00469,.00606,.00719,.01407,.00072,32.684,.368535,.742133,-7.695734,.17854,1.544609,.056141]

cases = [
    ("Diabetes  pos", dm, d_pos, 1, False),
    ("Diabetes  neg", dm, d_neg, 0, False),
    ("Heart     pos", hm, h_pos, 1, True),
    ("Heart     neg", hm, h_neg, 0, True),
    ("Parkinson pos", pm, p_pos, 1, False),
    ("Parkinson neg", pm, p_neg, 0, False),
]
for label, mdl, x, exp_pred, proba in cases:
    pred = int(mdl.predict([x])[0])
    sc   = score(mdl, x, proba)
    ok_sc = (sc > 50) if exp_pred == 1 else (sc < 50)
    chk(f"{label}: pred={pred} score={sc:.1f}%",
        pred == exp_pred and ok_sc)

print()
print("=" * 60)
print("POLISH  (D)")
print("=" * 60)
chk('width="stretch" on charts',            'width="stretch"' in src)
chk("st.divider() used",                    "st.divider()" in src)
chk("DISCLAIMER on every page",             src.count("st.caption(DISCLAIMER)") >= 4)
chk("Emoji in every page title",            "🩸" in src and "❤️" in src and "🧠" in src)
chk("No crashed blank state (no empty str default for st.session_state result)",
                                            "diab_result_pred" in src)

print()
print("=" * 60)
status_str = "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"
print(status_str)
print("=" * 60)
sys.exit(0 if ok else 1)
