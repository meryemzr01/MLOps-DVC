import os
import time

import numpy as np
import pandas as pd
import psycopg2
import yaml
from prometheus_client import Gauge, start_http_server

with open("params.yaml") as f:
    cfg = yaml.safe_load(f)["drift"]

WINDOW = int(os.getenv("WINDOW", "500"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "200"))
INTERVAL = int(os.getenv("INTERVAL", "10"))
DB = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", "5432")),
    user=os.getenv("PG_USER", "fraud"),
    password=os.getenv("PG_PASSWORD", "fraud123"),
    dbname=os.getenv("PG_DB", "fraud_db"),
)

# Reference = donnees de test (meme distribution que l'entrainement)
ref = pd.read_csv("data/processed/test.csv").drop(columns=["Class"])
FEATURES = list(ref.columns)

EDGES, REF_PROP = {}, {}
for c in FEATURES:
    e = np.unique(np.quantile(ref[c], np.linspace(0, 1, 11)))
    e[0], e[-1] = -np.inf, np.inf
    EDGES[c] = e
    REF_PROP[c] = np.clip(np.histogram(ref[c], e)[0] / len(ref), 1e-4, None)


def psi(c, values):
    cur = np.clip(np.histogram(values, EDGES[c])[0] / len(values), 1e-4, None)
    return float(np.sum((cur - REF_PROP[c]) * np.log(cur / REF_PROP[c])))


g_psi = Gauge("drift_feature_psi", "PSI par variable", ["feature"])
g_n = Gauge("drift_features_drifted", "Nombre de variables en drift")
g_share = Gauge("drift_share", "Part de variables en drift")
g_flag = Gauge("drift_detected", "1 si drift global detecte")
g_win = Gauge("drift_window_size", "Taille de la fenetre analysee")
g_fraud = Gauge("window_fraud_rate", "Taux de fraudes predites sur la fenetre")


def run_once():
    conn = psycopg2.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT features, is_fraud FROM predictions ORDER BY id DESC LIMIT %s",
                (WINDOW,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    g_win.set(len(rows))
    if len(rows) < MIN_SAMPLES:
        print(f"fenetre trop petite ({len(rows)}/{MIN_SAMPLES})")
        return
    df = pd.DataFrame([r[0] for r in rows])[FEATURES]
    g_fraud.set(sum(r[1] for r in rows) / len(rows))

    n_drift = 0
    for c in FEATURES:
        s = psi(c, df[c].values)
        g_psi.labels(c).set(s)
        n_drift += s > cfg["psi_threshold"]
    share = n_drift / len(FEATURES)
    g_n.set(n_drift)
    g_share.set(share)
    g_flag.set(int(share > cfg["share_threshold"]))
    print(f"fenetre={len(rows)} variables en drift={n_drift}/{len(FEATURES)} "
          f"drift_global={share > cfg['share_threshold']}")


if __name__ == "__main__":
    start_http_server(8001)
    print("Moniteur de drift: metriques sur http://localhost:8001/metrics")
    while True:
        try:
            run_once()
        except Exception as e:
            print("erreur:", e)
        time.sleep(INTERVAL)