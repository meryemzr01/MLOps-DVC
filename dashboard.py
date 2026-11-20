import html
import json
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import psycopg2
import requests
import streamlit as st
import yaml

st.set_page_config(page_title="Fraude : suivi en direct", layout="wide")

DB = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", "5432")),
    user=os.getenv("PG_USER", "fraud"),
    password=os.getenv("PG_PASSWORD", "fraud123"),
    dbname=os.getenv("PG_DB", "fraud_db"),
)
PROM = os.getenv("PROM_URL", "http://localhost:9090")

with open("params.yaml", encoding="utf-8-sig") as f:
    cfg = yaml.safe_load(f)
PSI_T = cfg["drift"]["psi_threshold"]
SHARE_T = cfg["drift"]["share_threshold"]

# ---------------------------------------------------------------- design ----
INK = "#14213D"
MUTED = "#6B778C"
LINE = "#E3E8EF"
OK = "#12906B"
BAD = "#D63B52"
STABLE = "#9DB2CE"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"], button, input {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}
.stApp { background: #F3F5F9; color: #14213D; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

/* sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3E8EF; }
section[data-testid="stSidebar"] * { color: #14213D; }

/* top bar */
.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.4rem; gap: 1rem; flex-wrap: wrap; }
.topbar h1 { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; padding: 0; color: #14213D; }
.pill { display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .8rem; border-radius: 999px;
        background: #FFFFFF; border: 1px solid #E3E8EF; font-size: .8rem; font-weight: 600; color: #6B778C; }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #12906B; }
.pill.live .dot { animation: pulse 1.8s ease-out infinite; }
.pill.off .dot { background: #9DB2CE; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(18,144,107,.55); } 100% { box-shadow: 0 0 0 9px rgba(18,144,107,0); } }

/* hero status */
.hero { display: grid; grid-template-columns: 1.3fr 1fr; gap: 2rem; align-items: center;
        padding: 1.6rem 1.9rem; border-radius: 18px; margin-bottom: 1.1rem; color: #fff; }
.hero.ok  { background: linear-gradient(115deg, #0E6F55 0%, #12906B 100%); }
.hero.bad { background: linear-gradient(115deg, #A92439 0%, #D63B52 100%); }
.hero .title { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.015em; margin-bottom: .35rem; }
.hero .sub { font-size: .98rem; line-height: 1.5; opacity: .92; max-width: 46ch; }
.meter-head { display: flex; justify-content: space-between; font-size: .82rem; font-weight: 600; opacity: .95; margin-bottom: .55rem; }
.meter { position: relative; height: 12px; border-radius: 999px; background: rgba(255,255,255,.25); }
.meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: #fff; }
.meter .mark { position: absolute; top: -6px; bottom: -6px; width: 2px; background: #fff; border-radius: 2px; }
.meter-foot { display: flex; justify-content: space-between; font-size: .75rem; opacity: .85; margin-top: .5rem; }
@media (max-width: 860px) { .hero { grid-template-columns: 1fr; gap: 1.2rem; } }

/* KPI cards */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.4rem; }
@media (max-width: 980px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
.kpi { background: #fff; border: 1px solid #E3E8EF; border-radius: 14px; padding: 1.1rem 1.2rem; border-top: 3px solid #9DB2CE; }
.kpi.good { border-top-color: #12906B; }
.kpi.bad  { border-top-color: #D63B52; }
.kpi .label { font-size: .82rem; font-weight: 600; color: #6B778C; margin-bottom: .55rem; }
.kpi .value { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05; color: #14213D; font-variant-numeric: tabular-nums; }
.kpi .value small { font-size: 1rem; font-weight: 600; color: #6B778C; letter-spacing: 0; }
.kpi .hint { font-size: .8rem; color: #6B778C; margin-top: .45rem; }

/* section titles */
.section { font-size: 1.1rem; font-weight: 700; letter-spacing: -0.01em; margin: 1.6rem 0 .8rem; color: #14213D; }
.section.tight { margin-top: 0; }

/* insights */
.notes { background: #fff; border: 1px solid #E3E8EF; border-radius: 14px; padding: .4rem 1.3rem; }
.note { padding: .95rem 0 .95rem 1rem; border-left: 3px solid #9DB2CE; margin: .5rem 0; font-size: .95rem; line-height: 1.55; }
.note.good { border-left-color: #12906B; }
.note.bad  { border-left-color: #D63B52; }
.note b { font-weight: 700; }

/* chart cards */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border: 1px solid #E3E8EF !important; border-radius: 14px; padding: .4rem .6rem; }
.caption { font-size: .82rem; color: #6B778C; margin-top: .2rem; }

/* alerts table */
.empty { background: #fff; border: 1px dashed #C9D3E0; border-radius: 14px; padding: 1.1rem 1.3rem; color: #6B778C; font-size: .95rem; }
div[data-testid="stDataFrame"] { border: 1px solid #E3E8EF; border-radius: 14px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def psi(a, b, bins=10):
    """Même calcul que src/detect_drift.py."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


@st.cache_data
def reference():
    return pd.read_csv("data/processed/test.csv")


@st.cache_data(ttl=3)
def predictions(n):
    conn = psycopg2.connect(**DB)
    try:
        return pd.read_sql(
            "SELECT id, ts, proba, is_fraud, latency_ms, features "
            "FROM predictions ORDER BY id DESC LIMIT %(n)s",
            conn, params={"n": n},
        )
    finally:
        conn.close()


def style_chart(ch):
    return (
        ch.configure_view(stroke=None)
        .configure(background="transparent")
        .configure_axis(
            labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans",
            labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
            titleFontWeight=600, gridColor=LINE, domain=False, tickColor=LINE,
        )
    )


def kpi(label, value, hint="", tone="", unit=""):
    u = f" <small>{unit}</small>" if unit else ""
    return (f'<div class="kpi {tone}"><div class="label">{label}</div>'
            f'<div class="value">{value}{u}</div><div class="hint">{hint}&nbsp;</div></div>')


n = st.sidebar.slider("Fenêtre : dernières prédictions", 100, 2000, 500, 100)
live = st.sidebar.checkbox("Rafraîchir toutes les 5 s", True)


def refresh():
    if live:
        time.sleep(5)
        st.rerun()


pill = ('<span class="pill live"><span class="dot"></span>En direct</span>' if live
        else '<span class="pill off"><span class="dot"></span>Pause</span>')
st.markdown(
    f'<div class="topbar"><h1>Détection de fraude : suivi en direct</h1>{pill}</div>',
    unsafe_allow_html=True,
)

try:
    df = predictions(n)
except Exception as e:
    st.error(f"Postgres inaccessible : {e}. Lance `docker compose up -d postgres`.")
    refresh()
    st.stop()

if df.empty:
    st.info("Aucune prédiction pour l'instant. Lance : python scripts/simulate_traffic.py normal 300 0.05")
    refresh()
    st.stop()

ref = reference()
base_rate = float(ref["Class"].mean())
feats = pd.DataFrame(list(df["features"]))
cols = [c for c in ref.columns if c != "Class" and c in feats.columns]
psis = pd.Series(
    {c: psi(ref[c].values, feats[c].astype(float).values) for c in cols}
).sort_values(ascending=False)
drifted = psis[psis > PSI_T]
share = len(drifted) / len(psis)
drift_on = share > SHARE_T
rate = float(df["is_fraud"].mean())
p95 = float(df["latency_ms"].quantile(0.95))
ratio = rate / base_rate if base_rate else 0

# ------------------------------------------------------------------ hero ----
if drift_on:
    cls = "bad"
    title = "Drift détecté"
    sub = (f"{len(drifted)} variables sur {len(psis)} ont changé "
           f"({share:.1%}, seuil {SHARE_T:.0%}).")
else:
    cls = "ok"
    title = "Pas de drift"
    sub = "Les transactions reçues ressemblent à celles du jeu de test."

fill_w = min(share, 1.0) * 100
mark_x = min(SHARE_T, 1.0) * 100
st.markdown(
    f"""
<div class="hero {cls}">
  <div>
    <div class="title">{title}</div>
    <div class="sub">{sub}</div>
  </div>
  <div>
    <div class="meter-head"><span>Variables en drift</span><span>{share:.1%}</span></div>
    <div class="meter">
      <div class="fill" style="width:{fill_w:.1f}%"></div>
      <div class="mark" style="left:{mark_x:.1f}%"></div>
    </div>
    <div class="meter-foot"><span>0 %</span><span>seuil {SHARE_T:.0%}</span><span>100 %</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ KPIs ----
st.markdown(
    '<div class="kpis">'
    + kpi("Prédictions analysées", f"{len(df):,}".replace(",", " "))
    + kpi("Variables en drift", f"{len(drifted)}", tone="bad" if drift_on else "good",
          unit=f"/ {len(psis)}")
    + kpi("Transactions signalées fraude", f"{rate:.1%}", f"attendu {base_rate:.2%}",
          tone="bad" if ratio > 3 else "good")
    + kpi("Latence p95", f"{p95:.1f}", tone="good" if p95 < 100 else "bad", unit="ms")
    + "</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- notes ----
st.markdown('<div class="section tight">Ce que ça veut dire</div>', unsafe_allow_html=True)
notes = []  # (ton, html)
if drift_on:
    names = html.escape(", ".join(drifted.index[:5]))
    notes.append(("bad", f"Les variables <b>{names}</b> ont le plus changé "
                  f"(PSI max {psis.iloc[0]:.2f}, seuil {PSI_T}). Les transactions reçues ne "
                  "ressemblent plus à celles que le modèle a apprises."))
    notes.append(("bad", "Le modèle devient moins fiable, surtout en précision (plus de fausses alertes). "
                  "Piste : réentraîner avec des données récentes labellisées."))
else:
    notes.append(("good", f"Aucune variable ne dépasse le seuil PSI de {PSI_T} de façon significative : "
                  "le modèle peut rester en production."))
if ratio > 3:
    notes.append(("bad", f"Le modèle signale {rate:.1%} des transactions comme fraude, soit {ratio:.0f} fois plus que "
                  f"le taux attendu ({base_rate:.2%}). Ça peut venir du drift (fausses alertes)."))
else:
    notes.append(("good", f"Le taux d'alertes ({rate:.2%}) reste proche du taux attendu ({base_rate:.2%})."))
if p95 < 100:
    notes.append(("good", f"L'API répond vite (p95 = {p95:.0f} ms)."))
else:
    notes.append(("bad", f"L'API est lente (p95 = {p95:.0f} ms) : vérifier la charge."))
try:
    with open("metrics/metrics.json", encoding="utf-8-sig") as f:
        m = json.load(f)
    pn, pd_ = m["normal"]["precision"], m["drift"]["precision"]
    notes.append(("", f"Hors ligne, la précision passe de {pn:.0%} (données normales) à {pd_:.0%} "
                  f"(données dérivées), pour un recall quasi stable "
                  f"({m['normal']['recall']:.0%} contre {m['drift']['recall']:.0%})."))
except Exception:
    pass
st.markdown(
    '<div class="notes">'
    + "".join(f'<div class="note {t}">{txt}</div>' for t, txt in notes)
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- charts ----
st.markdown('<div class="section">Analyse détaillée</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<div class="section tight" style="font-size:1rem">PSI par variable</div>',
                    unsafe_allow_html=True)
        pdf = psis.rename("psi").rename_axis("variable").reset_index()
        pdf["état"] = np.where(pdf["psi"] > PSI_T, "Dérive", "Stable")
        bars = alt.Chart(pdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("variable:N", sort=None, title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("psi:Q", title="PSI"),
            color=alt.Color("état:N", legend=None,
                            scale=alt.Scale(domain=["Stable", "Dérive"], range=[STABLE, BAD])),
            tooltip=["variable", alt.Tooltip("psi:Q", format=".3f"), "état"],
        )
        rule = alt.Chart(pd.DataFrame({"t": [PSI_T]})).mark_rule(
            color=INK, strokeDash=[5, 4], strokeWidth=1.5).encode(y="t:Q")
        st.altair_chart(style_chart((bars + rule).properties(height=290)), use_container_width=True)
        st.markdown(f'<div class="caption">Au-dessus de {PSI_T} (ligne pointillée) : la variable a dérivé.</div>',
                    unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.markdown('<div class="section tight" style="font-size:1rem">Probabilité de fraude dans le temps</div>',
                    unsafe_allow_html=True)
        ldf = df.sort_values("id")[["id", "proba"]]
        base = alt.Chart(ldf).encode(
            x=alt.X("id:Q", title="Prédiction", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format="d", tickCount=5)),
            y=alt.Y("proba:Q", title="Probabilité", scale=alt.Scale(domain=[0, 1])),
            tooltip=["id", alt.Tooltip("proba:Q", format=".3f")],
        )
        area = base.mark_area(color=INK, opacity=0.07)
        line = base.mark_line(color=INK, strokeWidth=1.6)
        st.altair_chart(style_chart((area + line).properties(height=290)), use_container_width=True)
        st.markdown('<div class="caption">Chaque point est une prédiction reçue par l\'API.</div>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- alerts ----
st.markdown('<div class="section">Alertes Prometheus</div>', unsafe_allow_html=True)
try:
    alerts = requests.get(f"{PROM}/api/v1/alerts", timeout=2).json()["data"]["alerts"]
    if alerts:
        st.dataframe(pd.DataFrame([{"alerte": a["labels"].get("alertname"), "état": a["state"],
                                    "depuis": a.get("activeAt")} for a in alerts]),
                     use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="empty">Aucune alerte active.</div>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Prometheus inaccessible ({e}).")

refresh()