import os
import joblib
import pandas as pd
import yaml
from xgboost import XGBClassifier

with open("params.yaml") as f:
    p = yaml.safe_load(f)["train"]

df = pd.read_csv("data/processed/train.csv")
X, y = df.drop(columns=["Class"]), df["Class"]

# Gestion du desequilibre extreme des classes
spw = (y == 0).sum() / max((y == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=p["n_estimators"],
    max_depth=p["max_depth"],
    learning_rate=p["learning_rate"],
    scale_pos_weight=spw,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
).fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print(f"scale_pos_weight={spw:.1f}")
