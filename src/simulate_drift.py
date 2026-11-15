import os
import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv").sort_values("Time").reset_index(drop=True)
tail = df.iloc[int(len(df) * 0.70):].copy()

rng = np.random.default_rng(42)
tail["Amount"] *= 3  # montants en hausse (type Black Friday)
for c in ["V1", "V2", "V3", "V4"]:
    tail[c] += rng.normal(2.0, 0.5, len(tail))  # decalage des distributions

os.makedirs("data/production", exist_ok=True)
tail.to_csv("data/production/stream_drifted.csv", index=False)
print(f"stream_drifted: {len(tail)} lignes, {int(tail['Class'].sum())} fraudes")

