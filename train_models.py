"""Retrain the three models and save them to saved_models/.

Mirrors the original Colab notebooks (same algorithms, split, random_state),
but reads the CSVs from ./dataset and writes pickles with YOUR installed
scikit-learn version, which avoids version-mismatch problems.

Run:  python train_models.py
"""
import os
import pickle

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "dataset")
OUT = os.path.join(BASE, "saved_models")
os.makedirs(OUT, exist_ok=True)


def fit_and_save(name, model, X, y, stratify):
    # .values -> plain arrays, so the app can call predict() with a plain list
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2,
        stratify=y if stratify else None, random_state=2)
    model.fit(X_train, y_train)
    tr = accuracy_score(y_train, model.predict(X_train))
    te = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<14} train acc = {tr:.4f}   test acc = {te:.4f}")
    with open(os.path.join(OUT, f"{name}.sav"), "wb") as f:
        pickle.dump(model, f)


# Diabetes: linear SVM
d = pd.read_csv(os.path.join(DATA, "diabetes.csv"))
fit_and_save("diabetes_model", svm.SVC(kernel="linear"),
             d.drop(columns="Outcome"), d["Outcome"], stratify=True)

# Heart disease: logistic regression (utf-8-sig strips the BOM in heart.csv)
h = pd.read_csv(os.path.join(DATA, "heart.csv"), encoding="utf-8-sig")
fit_and_save("heart_disease_model", LogisticRegression(max_iter=1000),
             h.drop(columns="target"), h["target"], stratify=True)

# Parkinson's: linear SVM
p = pd.read_csv(os.path.join(DATA, "parkinsons.csv"))
fit_and_save("parkinsons_model", svm.SVC(kernel="linear"),
             p.drop(columns=["name", "status"]), p["status"], stratify=False)

print("Saved models to", OUT)
