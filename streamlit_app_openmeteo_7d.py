
import os
import re
from datetime import date, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Config paths
# =========================
BASE_KABKOTA_CSV = "data/base_kabkota.csv"
INCIDENCE_CSV = "data/incidence_dummy_weekly.csv"   # << precomputed dummy table
GEOCODE_CACHE = "data/geocode_cache.csv"
OPEN_METEO_FC = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
HORIZON_DAYS = 7  # fixed 7-day horizon

st.set_page_config(page_title="SIGER-DBD – Open‑Meteo (7 Hari) + Insidensi CSV", layout="wide")
st.title("SIGER-DBD – Prediksi Risiko (Open‑Meteo 7 Hari + Insidensi dari CSV)")

# =========================
# Name normalization & aliases (DKI)
# =========================
ALIASES = {
    "KAB. ADM. KEP. SERIBU": "Kepulauan Seribu",
    "KAB ADM KEP SERIBU": "Kepulauan Seribu",
    "KOTA ADM. JAKARTA PUSAT": "Jakarta Pusat",
    "KOTA ADM. JAKARTA UTARA": "Jakarta Utara",
    "KOTA ADM. JAKARTA BARAT": "Jakarta Barat",
    "KOTA ADM. JAKARTA SELATAN": "Jakarta Selatan",
    "KOTA ADM. JAKARTA TIMUR": "Jakarta Timur",
}

def _normalize_name(txt: str) -> str:
    if not txt: return ""
    t = re.sub(r"[^A-Za-z0-9\s\.]", " ", txt.upper())
    t = re.sub(r"\s+", " ", t).strip()
    if t in ALIASES:
        return ALIASES[t]
    t = t.replace("KAB. ", "KABUPATEN ").replace("KOTA ADM. ", "KOTA ")
    t = t.replace("ADM ", "").replace(" KEP ", " KEPULAUAN ")
    t = t.replace(" KEP. ", " KEPULAUAN ")
    t = re.sub(r"\s+", " ", t).strip()
    return t.title()

# =========================
# Base loaders
# =========================
def load_base_kabkota(path=BASE_KABKOTA_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for c in ["prov_code","prov_name","kabkota_code","kabkota_name"]:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' wajib ada di {path}.")
    if "lat" not in df.columns: df["lat"] = np.nan
    if "lon" not in df.columns: df["lon"] = np.nan
    return df.assign(
        prov_code=lambda d: d["prov_code"].str.strip(),
        prov_name=lambda d: d["prov_name"].str.strip(),
        kabkota_code=lambda d: d["kabkota_code"].str.strip(),
        kabkota_name=lambda d: d["kabkota_name"].str.strip(),
    )

@st.cache_data(ttl=24*60*60, show_spinner=False)
def load_geocode_cache(path=GEOCODE_CACHE) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["place","lat","lon"])
    return pd.read_csv(path, dtype={"place": str, "lat": float, "lon": float})

def save_geocode_cache(df_cache: pd.DataFrame, path=GEOCODE_CACHE):
    df_cache.to_csv(path, index=False)

# =========================
# Incidence CSV loader
# =========================
def load_incidence_csv(csv_path: str, kabkota_code: str, take_last_weeks: int = 52) -> pd.DataFrame:
    try:
        df_all = pd.read_csv(csv_path, dtype={"kabkota_code": str, "kabkota_name": str})
    except Exception:
        return pd.DataFrame()
    df = df_all[df_all["kabkota_code"].astype(str) == str(kabkota_code)].copy()
    if df.empty:
        return df
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df = df.sort_values("week_start_date")
    return df.tail(take_last_weeks).reset_index(drop=True)

# =========================
# Geocoding & Forecast
# =========================
@st.cache_data(ttl=6*60*60, show_spinner=False)
def geocode_open_meteo(place_variants: list[str]):
    cache = load_geocode_cache()
    # cache hit
    for p in place_variants:
        hit = cache[cache["place"] == p]
        if not hit.empty:
            return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"]), f"cache:{p}"
    # query
    for p in place_variants:
        params = {"name": p, "count": 1, "language": "id", "format": "json"}
        try:
            r = requests.get(OPEN_METEO_GEO, params=params, timeout=20, headers={"User-Agent":"SIGER-DBD/streamlit"})
            r.raise_for_status()
            j = r.json()
            if j.get("results"):
                res = j["results"][0]
                lat, lon = float(res["latitude"]), float(res["longitude"])
                cache = pd.concat([cache, pd.DataFrame([{"place": p, "lat": lat, "lon": lon}])], ignore_index=True)
                save_geocode_cache(cache)
                return lat, lon, f"new:{p}"
        except Exception:
            continue
    return None, None, "not_found"

@st.cache_data(ttl=2*60*60, show_spinner=False)
def fetch_open_meteo_hourly(lat: float, lon: float, tz: str, days: int = HORIZON_DAYS):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m",
        "forecast_days": int(days),
        "timezone": tz or "Asia/Jakarta"
        # "wind_speed_unit": "ms"  # uncomment if you prefer m/s
    }
    r = requests.get(OPEN_METEO_FC, params=params, timeout=30, headers={"User-Agent":"SIGER-DBD/streamlit"})
    r.raise_for_status()
    j = r.json()
    h = j.get("hourly", {})
    if not h or "time" not in h:
        return pd.DataFrame()
    df = pd.DataFrame({
        "local_datetime": pd.to_datetime(h["time"]),
        "t": h.get("temperature_2m"),
        "hu": h.get("relative_humidity_2m"),
        "ws": h.get("wind_speed_10m"),
        "tcc": h.get("cloud_cover"),
        "precip": h.get("precipitation"),
    })
    df["weather_desc"] = df["precip"].apply(lambda x: "Hujan" if (x or 0) > 0 else "Berawan/Cerah")
    return df

def summarize_open_meteo_7d(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame()
    days = HORIZON_DAYS
    df = df_hourly.copy()
    df["is_rain"] = (df["precip"].fillna(0) > 0).astype(int)
    out = pd.DataFrame({
        "t_mean_7d": [np.nanmean(df["t"])],
        "t_min_7d": [np.nanmin(df["t"])],
        "t_max_7d": [np.nanmax(df["t"])],
        "hu_mean_7d": [np.nanmean(df["hu"])],
        "ws_mean_7d": [np.nanmean(df["ws"])],
        "tcc_mean_7d": [np.nanmean(df["tcc"])],
        "rain_slots_7d": [df["is_rain"].sum()],
        "window_hours": [days * 24],
        "rain_frac": [float(df["is_rain"].sum()) / float(days*24)]
    })
    return out

# =========================
# Incidence features & model (robust)
# =========================
def build_lag_features(df_inc: pd.DataFrame, max_lag=4) -> pd.DataFrame:
    df = df_inc.sort_values("week_start_date").copy()
    for L in range(1, max_lag+1):
        df[f"incidence_per_100k_lag{L}"] = df["incidence_per_100k"].shift(L)
        df[f"outbreak_lag{L}"] = df["outbreak"].shift(L)
    return df

def incidence_model_proba(df_inc_with_lag: pd.DataFrame) -> float:
    df = df_inc_with_lag.dropna().copy()
    if df.shape[0] < 10:
        return 0.5
    train = df.iloc[:-1].copy()
    test = df.iloc[-1:].copy()
    feat_cols = [c for c in df.columns if c.startswith("incidence_per_100k_lag")] + \
                [c for c in df.columns if c.startswith("outbreak_lag")]
    X_tr = train[feat_cols]
    y_tr = train["outbreak"].astype(int)
    X_te = test[feat_cols]

    # Handle single-class training safely
    if y_tr.nunique() < 2:
        return float(y_tr.mean())  # prior probability (0.0..1.0)

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, min_samples_split=4,
            class_weight="balanced", random_state=42
        ))
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)
    classes_ = list(pipe.named_steps["rf"].classes_)
    if 1 in classes_:
        idx = classes_.index(1)
        return float(proba[:, idx][0])
    else:
        return 0.0

def weather_score(df_feats: pd.DataFrame) -> float:
    if df_feats.empty:
        return 0.5
    r = df_feats.iloc[0]
    t = float(r.get("t_mean_7d")) if pd.notna(r.get("t_mean_7d")) else 30.0
    rh = float(r.get("hu_mean_7d")) if pd.notna(r.get("hu_mean_7d")) else 85.0
    rain_frac = float(r.get("rain_frac")) if pd.notna(r.get("rain_frac")) else 0.0
    s_t = min(max((t - 24.0) / (34.0 - 24.0), 0.0), 1.0)
    s_rh = min(max((rh - 60.0) / (95.0 - 60.0), 0.0), 1.0)
    s_rain = min(max(rain_frac, 0.0), 1.0)
    return round(0.4*s_t + 0.35*s_rh + 0.25*s_rain, 3)

def fused_risk_score(weather_s: float, hist_proba: float, w_cuaca=0.4, w_hist=0.6) -> float:
    return round(w_cuaca*weather_s + w_hist*hist_proba, 3)

def risk_label_from_score(score: float) -> str:
    if score >= 0.66: return "High"
    if score >= 0.33: return "Medium"
    return "Low"

# =========================
# UI
# =========================
try:
    df_base = load_base_kabkota(BASE_KABKOTA_CSV)
except Exception as e:
    st.error(f"Gagal baca {BASE_KABKOTA_CSV}: {e}")
    st.stop()

st.sidebar.header("Pilih Wilayah (Horizon tetap 7 hari)")
prov_list = df_base[["prov_code","prov_name"]].drop_duplicates().to_dict("records")
sel_prov = st.sidebar.selectbox("Provinsi", prov_list, format_func=lambda r: f"{r['prov_name']} ({r['prov_code']})")
prov_code, prov_name = sel_prov["prov_code"], sel_prov["prov_name"]

kab_list = df_base[df_base["prov_code"] == prov_code][["kabkota_code","kabkota_name","lat","lon"]].drop_duplicates().to_dict("records")
sel_kab = st.sidebar.selectbox("Kabupaten/Kota", kab_list, format_func=lambda r: f"{r['kabkota_name']} ({r['kabkota_code']})")
kabkota_code, kabkota_name = sel_kab["kabkota_code"], sel_kab["kabkota_name"]
lat_override = sel_kab.get("lat")
lon_override = sel_kab.get("lon")

st.write(f"📍 **Provinsi**: {prov_name} (`{prov_code}`)  \n🏙️ **Kab/Kota**: {kabkota_name} (`{kabkota_code}`)  \n🗓️ **Horizon**: **{HORIZON_DAYS} hari (tetap)**")

# ===== Weather (7d) =====
st.subheader("1) Cuaca (Hourly) – Open‑Meteo, 7 hari")

# 1) Use lat/lon override if provided
lat, lon, source = None, None, None
if pd.notna(lat_override) and pd.notna(lon_override):
    try:
        lat = float(lat_override); lon = float(lon_override); source = "csv"
    except Exception:
        lat, lon, source = None, None, None

# 2) Otherwise try multiple geocoding variants
if lat is None or lon is None:
    kab_norm = _normalize_name(kabkota_name)
    prov_norm = _normalize_name(prov_name)
    candidates = [
        f"{kab_norm}, {prov_norm}, Indonesia",
        f"{kab_norm}, Indonesia",
        f"{kab_norm}, Jakarta, Indonesia" if prov_norm.upper().startswith("DKI") else f"{kab_norm}, {prov_norm}",
        kab_norm
    ]
    lat, lon, source = geocode_open_meteo(candidates)

st.write(f"Geocoding: **{kabkota_name}**, {prov_name} → lat={lat}, lon={lon} (source: {source})")
if lat is None:
    st.error("Lokasi tidak ditemukan oleh geocoding Open-Meteo. Tambahkan kolom lat,lon di data/base_kabkota.csv atau perbaiki nama.")
    st.stop()

@st.cache_data(ttl=2*60*60, show_spinner=False)
def get_hourly(lat, lon):
    return fetch_open_meteo_hourly(lat, lon, tz="Asia/Jakarta", days=HORIZON_DAYS)

df_hourly = get_hourly(lat, lon)
if df_hourly.empty:
    st.error("Open‑Meteo tidak mengembalikan data untuk lokasi ini.")
    st.stop()
st.dataframe(df_hourly.head(24))

st.subheader("2) Fitur Ringkas 7‑Hari (Kab/Kota)")
df_kab_wx = summarize_open_meteo_7d(df_hourly)
st.dataframe(df_kab_wx)

# ===== Incidence + Model =====
st.subheader("3) Riwayat Insidensi (52 minggu — dibaca dari CSV)")
df_inc = load_incidence_csv(INCIDENCE_CSV, kabkota_code, take_last_weeks=52)
if df_inc.empty:
    st.warning("CSV insidensi tidak ditemukan / tidak berisi kab/kota ini. App akan membuat dummy 52 minggu sementara.")
    # fallback
    from math import ceil
    def seed_from_text(txt: str) -> int:
        import hashlib
        h = hashlib.sha256(str(txt).encode("utf-8")).hexdigest()
        return int(h[:8], 16)
    rng = np.random.default_rng(seed_from_text(kabkota_code))
    pop = int(rng.integers(150_000, 2_500_000))
    # simple fallback 52w
    today = date.today()
    last_monday = today - timedelta(days=today.weekday())
    weeks = [last_monday - timedelta(weeks=i) for i in range(1, 53)]
    weeks = sorted(weeks)
    rows = []
    for i, wk in enumerate(weeks):
        lam = max(1.0, 12 + rng.normal(0, 3))
        cases = int(rng.poisson(lam))
        inc = (cases / pop) * 100000.0
        label = "unsafe" if inc*4.0 > 10.0 else ("moderately_safe" if inc*4.0 >= 3.0 else "safe")
        rows.append({"kabkota_code": kabkota_code, "kabkota_name": kabkota_name,
                     "week_start_date": wk, "cases": cases, "population": pop,
                     "incidence_per_100k": round(inc,4), "risk_label": label, "outbreak": 1 if label=="unsafe" else 0})
    df_inc = pd.DataFrame(rows)

st.dataframe(df_inc.tail(10))

st.subheader("4) Model Historis (RF lag 1..4)")
def build_lag_and_model(df_inc):
    df_lag = build_lag_features(df_inc, max_lag=4)
    proba = incidence_model_proba(df_lag)
    return df_lag, proba

df_lag, hist_proba = build_lag_and_model(df_inc)
st.write(f"🧪 Probabilitas outbreak (historis): **{hist_proba:.3f}**")

st.subheader("5) Skor Cuaca & Fusi")
wx_score = weather_score(df_kab_wx)
w_cuaca = st.slider("Bobot Cuaca", 0.0, 1.0, 0.4, 0.05)
w_hist = 1.0 - w_cuaca
final_score = fused_risk_score(wx_score, hist_proba, w_cuaca=w_cuaca, w_hist=w_hist)
label = risk_label_from_score(final_score)

c1, c2, c3 = st.columns(3)
with c1: st.metric("Prob. Outbreak (Historis)", f"{hist_proba:.2f}")
with c2: st.metric("Skor Cuaca (0–1)", f"{wx_score:.2f}")
with c3: st.metric("Risk Score (Final)", f"{final_score:.2f}")
st.success(f"Kategori Risiko: **{label}**")

# Optional: export current summary
if st.button("💾 Export ringkasan CSV"):
    out = df_kab_wx.copy()
    out.insert(0, "kabkota_code", kabkota_code)
    out.insert(1, "kabkota_name", kabkota_name)
    out["hist_proba"] = hist_proba
    out["wx_score"] = wx_score
    out["final_score"] = final_score
    out["risk_label"] = label
    out.to_csv("outputs_summary.csv", index=False, encoding="utf-8")
    st.success("Tersimpan: outputs_summary.csv (di root app)")

st.caption("Sumber cuaca: Open‑Meteo. Insidensi: dibaca dari data/incidence_dummy_weekly.csv (precomputed). Geocoding dicache ke data/geocode_cache.csv.")
