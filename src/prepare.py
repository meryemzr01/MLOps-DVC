import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    p = yaml.safe_load(f)["prepare"]

df = pd.read_csv("data/raw/creditcard.csv")

# Decoupage stratifie : train / flux "production" labellise / test
train, rest = train_test_split(
    df, test_size=1 - p["train_ratio"], stratify=df["Class"], random_state=42
)
prod_share = p["prod_ratio"] / (1 - p["train_ratio"])
prod, test = train_test_split(
    rest, test_size=1 - prod_share, stratify=rest["Class"], random_state=42
)


def apply_drift(d, seed):
    """Simule un drift sur les variables qui portent le signal de fraude."""
    d = d.copy()
    rng = np.random.default_rng(seed)
    d["Amount"] *= p["amount_factor"]
    for c in p["drift_features"]:
        d[c] += rng.normal(p["shift_mean"], 0.5, len(d))
    return d


drift_prod = apply_drift(prod, seed=1)   # donnees derivees avec labels recus
drift_test = apply_drift(test, seed=2)   # meme jeu de test, mais derive

if p["retrain_with_drift"]:
    train = pd.concat([train, drift_prod], ignore_index=True)

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
drift_test.to_csv("data/processed/drift_test.csv", index=False)
print(f"train={len(train)} test={len(test)} (fraudes={int(test['Class'].sum())}) "
      f"drift_test={len(drift_test)}")
