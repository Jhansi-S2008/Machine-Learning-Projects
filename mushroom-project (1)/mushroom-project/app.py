"""Flask backend. model.pkl is loaded once at startup; /predict never retrains."""
import os
import pickle
import subprocess
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.pkl")

if not os.path.exists(MODEL_PATH):  # first run convenience
    print("model.pkl not found - running train_model.py once...")
    subprocess.run([sys.executable, "train_model.py"], cwd=BASE, check=True)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
encoder = bundle["encoder"]
FEATURES = bundle["features"]
OPTIONS = bundle["options"]

app = Flask(__name__)

# Friendly dropdown names (codes come from the dataset itself).
LABELS = {
    "odor": {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty",
             "n": "None", "p": "Pungent", "s": "Spicy"},
    "spore-print-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green",
                          "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "stalk-root": {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs",
                   "r": "Rooted", "?": "Unknown"},
    "stalk-surface-below-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "habitat": {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban",
                "w": "Waste", "d": "Woods"},
}


def label(col, code):
    return LABELS.get(col, {}).get(code, code)


def title(col):
    return col.replace("-", " ").capitalize()


# One-hot column i  ->  (feature, category code). Same order the encoder produces.
ONEHOT = [(col, code) for col, cats in zip(FEATURES, encoder.categories_) for code in cats]


def build_tree():
    """Turn the trained sklearn tree into JSON the browser can draw."""
    t = model.tree_

    def walk(i):
        counts = t.value[i][0]
        node = {
            "id": int(i),
            "samples": int(t.n_node_samples[i]),
            "prediction": "Poisonous" if counts[1] > counts[0] else "Edible",
        }
        if t.children_left[i] == -1:
            node["leaf"] = True
        else:
            col, code = ONEHOT[t.feature[i]]
            node["leaf"] = False
            node["feature"] = title(col)
            node["value"] = label(col, code)
            node["no"] = walk(t.children_left[i])     # one-hot value <= 0.5  -> "No"
            node["yes"] = walk(t.children_right[i])   # one-hot value >  0.5  -> "Yes"
        return node

    return walk(0)


TREE = build_tree()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    fields = []
    for col in FEATURES:
        opts = [{"value": v, "label": label(col, v)} for v in OPTIONS[col]]
        opts.sort(key=lambda o: o["label"])
        fields.append({"name": col, "title": title(col), "options": opts})
    return jsonify({
        "fields": fields,
        "train_accuracy": round(bundle["train_accuracy"] * 100, 2),
        "test_accuracy": round(bundle["test_accuracy"] * 100, 2),
        "train_size": bundle["train_size"],
        "test_size": bundle["test_size"],
        "max_depth": bundle["max_depth"],
        "distribution": bundle["distribution"],
        "tree": TREE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the mushroom features."}), 400

    for col in FEATURES:
        if col not in data:
            return jsonify({"error": f"Missing field: {col}"}), 400
        if data[col] not in OPTIONS[col]:
            return jsonify({"error": f"Invalid value for {col}: {data[col]!r}"}), 400

    # Same feature order + same fitted encoder as training.
    row = pd.DataFrame([[data[c] for c in FEATURES]], columns=FEATURES)
    X = encoder.transform(row)

    pred = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][pred])
    path = [int(n) for n in model.decision_path(X).indices]  # nodes visited, root -> leaf

    return jsonify({
        "prediction": "Poisonous" if pred == 1 else "Edible",
        "confidence": round(confidence * 100, 2),
        "test_accuracy": round(bundle["test_accuracy"] * 100, 2),
        "path": path,
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
