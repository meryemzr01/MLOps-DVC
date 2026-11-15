import sys
import pandas as pd
import requests

URL = "http://localhost:8000/predict"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500

session = requests.Session()


def send(path, label):
    df = pd.read_csv(path).drop(columns=["Class"]).sample(N, random_state=0)
    ok = 0
    for row in df.to_dict(orient="records"):
        r = session.post(URL, json={"features": row})
        ok += r.status_code == 200
    print(f"{label}: {ok}/{N} requetes OK")


send("data/processed/test.csv", "trafic normal")
send("data/processed/drift_test.csv", "trafic derive")