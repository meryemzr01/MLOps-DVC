import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
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


class Transaction(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "n_features": len(FEATURES)}


@app.post("/predict")
def predict(t: Transaction):
    missing = [f for f in FEATURES if f not in t.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"variables manquantes: {missing}")

    t0 = time.perf_counter()
    X = pd.DataFrame([t.features])[FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    is_fraud = proba >= THRESHOLD
    latency = (time.perf_counter() - t0) * 1000

    with state["conn"].cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (model_version, features, proba, is_fraud, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s)",
            (MODEL_VERSION, Json({k: t.features[k] for k in FEATURES}), proba, is_fraud, latency),
        )
    return {
        "is_fraud": is_fraud,
        "proba": round(proba, 4),
        "model_version": MODEL_VERSION,
    }