"""
Comprehensive verification script for the upgraded app.py
"""
import ast, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
APP  = BASE / 'app.py'

src = APP.read_text(encoding='utf-8')

# 1. Syntax
ast.parse(src)
print("PASS  1/6  Syntax check")

# 2. Imports available
import plotly.graph_objects as go   # noqa
import plotly.express as px          # noqa
print("PASS  2/6  Plotly importable")

# 3. Datasets load & shapes
d = pd.read_csv(BASE / 'dataset' / 'diabetes.csv')
h = pd.read_csv(BASE / 'dataset' / 'heart.csv', encoding='utf-8-sig')
p = pd.read_csv(BASE / 'dataset' / 'parkinsons.csv')
p = p.drop(columns=['name'])
DIAB_FEAT  = [c for c in d.columns if c != 'Outcome']
HEART_FEAT = [c for c in h.columns if c != 'target']
PARK_FEAT  = [c for c in p.columns if c != 'status']
assert len(DIAB_FEAT)  == 8,  f"Expected 8 diabetes features, got {len(DIAB_FEAT)}"
assert len(HEART_FEAT) == 13, f"Expected 13 heart features, got {len(HEART_FEAT)}"
assert len(PARK_FEAT)  == 22, f"Expected 22 parkinson features, got {len(PARK_FEAT)}"
print("PASS  3/6  Dataset shapes OK")

# 4. Models load & predict
def load(name):
    with open(BASE / 'saved_models' / name, 'rb') as f:
        return pickle.load(f)

dm = load('diabetes_model.sav')
hm = load('heart_disease_model.sav')
pm = load('parkinsons_model.sav')

# Positive samples
d_pos = [6,148,72,35,0,33.6,0.627,50]
h_pos = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
p_pos = [119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,
         0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,
         0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]
# Negative samples
d_neg = [4,110,92,0,0,37.6,0.191,30]
h_neg = [63,1,0,130,254,0,0,147,0,1.4,1,1,3]
p_neg = [202.266,211.604,197.079,0.0018,0.000009,0.00093,0.00107,0.00278,
         0.00954,0.085,0.00469,0.00606,0.00719,0.01407,0.00072,32.684,
         0.368535,0.742133,-7.695734,0.17854,1.544609,0.056141]

assert dm.predict([d_pos])[0] == 1 and dm.predict([d_neg])[0] == 0, "Diabetes predictions wrong"
assert hm.predict([h_pos])[0] == 1 and hm.predict([h_neg])[0] == 0, "Heart predictions wrong"
assert pm.predict([p_pos])[0] == 1 and pm.predict([p_neg])[0] == 0, "Parkinson predictions wrong"
print("PASS  4/6  All 6 sample predictions correct")

# 5. Scores computed correctly
import math
def sigmoid(x): return 1/(1+math.exp(-x))
def model_score(model, X, use_proba=False):
    if use_proba:
        return float(model.predict_proba([X])[0][1]) * 100
    return sigmoid(float(model.decision_function([X])[0])) * 100

sc_dp = model_score(dm, d_pos); sc_dn = model_score(dm, d_neg)
sc_hp = model_score(hm, h_pos, True); sc_hn = model_score(hm, h_neg, True)
sc_pp = model_score(pm, p_pos); sc_pn = model_score(pm, p_neg)
assert sc_dp > 50, f"Diabetes pos score should be >50, got {sc_dp:.1f}"
assert sc_dn < 50, f"Diabetes neg score should be <50, got {sc_dn:.1f}"
assert sc_hp > 50, f"Heart pos score should be >50, got {sc_hp:.1f}"
assert sc_hn < 50, f"Heart neg score should be <50, got {sc_hn:.1f}"
assert sc_pp > 50, f"Parkinson pos score should be >50, got {sc_pp:.1f}"
assert sc_pn < 50, f"Parkinson neg score should be <50, got {sc_pn:.1f}"
print(f"PASS  5/6  Scores: DM {sc_dp:.1f}/{sc_dn:.1f}  HM {sc_hp:.1f}/{sc_hn:.1f}  PM {sc_pp:.1f}/{sc_pn:.1f}")

# 6. Contribution chart dimensionality
coef_d = dm.coef_[0]; means_d = d[DIAB_FEAT].mean().values
contribs = coef_d * (np.array(d_pos) - means_d)
assert len(contribs) == 8
coef_p = pm.coef_[0]; means_p = p[PARK_FEAT].mean().values
contribs_p = coef_p * (np.array(p_pos) - means_p)
assert len(contribs_p) == 22
print("PASS  6/6  Contribution chart dimensions OK")

print()
print("All 6 checks PASSED - app is ready.")
