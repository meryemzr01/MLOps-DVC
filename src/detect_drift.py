import json
import os
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

with open("params.yaml") as f:
    cfg = yaml.safe_load(f)["drift"]

ref = pd.read_csv("data/processed/test.csv").drop(columns=["Class"])
cur = pd.read_csv("data/processed/drift_test.csv").drop(columns=["Class"])


def psi(a, b, bins=10):
    """Population Stability Index, bins definis sur la reference."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


features = {}
for c in ref.columns:
    s = psi(ref[c].values, cur[c].values)
    ks = ks_2samp(ref[c].values, cur[c].values)
    features[c] = {
        "psi": round(s, 4),
        "ks_stat": round(float(ks.statistic), 4),
        "drift": bool(s > cfg["psi_threshold"]),
    }

drifted = sorted(
    [c for c, v in features.items() if v["drift"]],
    key=lambda c: -features[c]["psi"],
)
report = {
    "n_features": len(features),
    "n_drifted": len(drifted),
    "share_drifted": round(len(drifted) / len(features), 3),
    "max_psi": max(v["psi"] for v in features.values()),
    "dataset_drift": bool(len(drifted) / len(features) > cfg["share_threshold"]),
    "drifted_features": drifted,
    "features": features,
}

os.makedirs("metrics", exist_ok=True)
with open("metrics/drift_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Drift detecte sur {len(drifted)}/{len(features)} variables: {drifted}")
print(f"Drift global: {report['dataset_drift']} (max PSI = {report['max_psi']})")