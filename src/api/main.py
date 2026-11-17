import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from psycopg2.extras import Json
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "fraud-v7")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DB = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", "5432")),
    user=os.getenv("PG_USER", "fraud"),
    password=os.getenv("PG_PASSWORD", "fraud123"),
    dbname=os.getenv("PG_DB", "fraud_db"),
)

model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)
state = {}

PREDICTIONS = Counter(
    "fraud_predictions_total", "Nombre de predictions", ["model_version", "outcome"]
)
LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Latence de /predict",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
ERRORS = Counter("fraud_prediction_errors_total", "Requetes refusees (422)")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT now(),
    model_version TEXT,
    features JSONB,
    proba DOUBLE PRECISION,
    is_fraud BOOLEAN,
    latency_ms DOUBLE PRECISION
)
"""


@asynccontextmanager
async def lifespan(app):
    conn = psycopg2.connect(**DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(CREATE_SQL)
    state["conn"] = conn
    yield
    conn.close()


app = FastAPI(title="Fraud detection API", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Transaction(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "n_features": len(FEATURES)}


@app.post("/predict")
def predict(t: Transaction):
    missing = [f for f in FEATURES if f not in t.features]
    if missing:
        ERRORS.inc()
        raise HTTPException(status_code=422, detail=f"variables manquantes: {missing}")

    t0 = time.perf_counter()
    X = pd.DataFrame([t.features])[FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    is_fraud = proba >= THRESHOLD
    elapsed = time.perf_counter() - t0

    with state["conn"].cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (model_version, features, proba, is_fraud, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s)",
            (MODEL_VERSION, Json({k: t.features[k] for k in FEATURES}), proba, is_fraud, elapsed * 1000),
        )

    LATENCY.observe(elapsed)
    PREDICTIONS.labels(MODEL_VERSION, "fraud" if is_fraud else "legit").inc()
    return {"is_fraud": is_fraud, "proba": round(proba, 4), "model_version": MODEL_VERSION}