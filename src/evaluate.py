import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

with open("params.yaml") as f:
    thr = yaml.safe_load(f)["evaluate"]["threshold"]

model = joblib.load("models/model.pkl")


def score(path):
    df = pd.read_csv(path)
    X, y = df.drop(columns=["Class"]), df["Class"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)
    return {
        "pr_auc": round(average_precision_score(y, proba), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "n": int(len(df)),
        "n_fraud": int(y.sum()),
    }


metrics = {
    "normal": score("data/processed/test.csv"),
    "drift": score("data/processed/drift_test.csv"),
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
