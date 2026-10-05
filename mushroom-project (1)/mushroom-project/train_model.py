"""Trains a small, explainable Decision Tree and saves everything to model.pkl.

Run:  python train_model.py
"""
import os
import pickle
import urllib.request

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

CSV_PATH = "mushroom.csv"
UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "mushroom/agaricus-lepiota.data")

ALL_COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]

# The ONLY features used for training AND prediction (so the form matches the model).
FEATURES = ["odor", "spore-print-color", "gill-size", "stalk-root",
            "stalk-surface-below-ring", "habitat"]
MAX_DEPTH = 3


def load_data():
    if not os.path.exists(CSV_PATH):
        print("mushroom.csv not found - downloading from UCI...")
        urllib.request.urlretrieve(UCI_URL, CSV_PATH)
    with open(CSV_PATH) as f:
        has_header = f.readline().lower().startswith(("class", "poisonous"))
    # keep_default_na=False: letters like 'n' / '?' must never become NaN
    if has_header:
        df = pd.read_csv(CSV_PATH, keep_default_na=False)
        df.columns = ALL_COLUMNS
    else:
        df = pd.read_csv(CSV_PATH, header=None, names=ALL_COLUMNS, keep_default_na=False)
    return df


def main():
    df = load_data()
    X = df[FEATURES]
    y = df["class"].map({"e": 0, "p": 1})  # 0 = edible, 1 = poisonous

    # One-hot encoding -> every tree question reads like "Odor is None?"
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_enc = encoder.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Features : {FEATURES}")
    print(f"Train accuracy: {train_acc * 100:.2f}%  ({len(X_train)} rows)")
    print(f"Test accuracy : {test_acc * 100:.2f}%  ({len(X_test)} rows)")

    bundle = {
        "model": model,
        "encoder": encoder,
        "features": FEATURES,
        "options": {c: sorted(df[c].unique().tolist()) for c in FEATURES},
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "max_depth": MAX_DEPTH,
        "distribution": {"edible": int((y == 0).sum()), "poisonous": int((y == 1).sum())},
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
