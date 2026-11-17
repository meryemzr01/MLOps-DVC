import sys
import time

import pandas as pd
import requests

mode = sys.argv[1] if len(sys.argv) > 1 else "both"   # normal | drift | both
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
DELAY = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
URL = "http://localhost:8000/predict"
session = requests.Session()


def send(path, label):
    df = pd.read_csv(path).drop(columns=["Class"]).sample(N, random_state=0)
    ok = 0
    for row in df.to_dict(orient="records"):
        ok += session.post(URL, json={"features": row}).status_code == 200
        time.sleep(DELAY)
    print(f"{label}: {ok}/{N} requetes OK")


if mode in ("normal", "both"):
    send("data/processed/test.csv", "trafic normal")
if mode in ("drift", "both"):
    send("data/processed/drift_test.csv", "trafic derive")