import os
import pandas as pd
import yaml

with open("params.yaml") as f:
    p = yaml.safe_load(f)["prepare"]

df = pd.read_csv("data/raw/creditcard.csv").sort_values("Time").reset_index(drop=True)
n = len(df)
i_train = int(n * p["train_ratio"])
i_test = int(n * (p["train_ratio"] + p["test_ratio"]))

train = df.iloc[:i_train]
test = df.iloc[i_train:i_test]

# Flux de production derive : 1re moitie = labels recus, 2e moitie = test de drift
drift = pd.read_csv("data/production/stream_drifted.csv")
half = len(drift) // 2
drift_labeled, drift_test = drift.iloc[:half], drift.iloc[half:]

if p["retrain_with_drift"]:
    train = pd.concat([train, drift_labeled], ignore_index=True)

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
drift_test.to_csv("data/processed/drift_test.csv", index=False)
print(f"train={len(train)} test={len(test)} drift_test={len(drift_test)}")
