
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
INCIDENCE_CSV = "data/incidence_dummy_weekly.csv"
WEATHER_WEEKLY_CSV = "data/weather_weekly_agg.csv"   # built by tools/build_weather_weekly.py
GEOCODE_CACHE = "data/geocode_cache.csv"
OPEN_METEO_FC = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
HORIZON_DAYS = 7  # fixed 7-day horizon

st.set_page_config(page_title="SIGER-DBD – Early-Fusion RF (7 Hari)", layout="wide")
st.title("SIGER-DBD – Random Forest Early‑Fusion (Historis + Cuaca Mingguan)")

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
# Loaders
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

def load_incidence_csv(csv_path: str, kabkota_code: str) -> pd.DataFrame:
    try:
        df_all = pd.read_csv(csv_path, dtype={"kabkota_code": str, "kabkota_name": str})
    except Exception:
        return pd.DataFrame()
    df = df_all[df_all["kabkota_code"].astype(str) == str(kabkota_code)].copy()
    if df.empty:
        return df
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df.sort_values("week_start_date").reset_index(drop=True)

def load_weather_weekly(csv_path: str, kabkota_code: str) -> pd.DataFrame:
    try:
        df_all = pd.read_csv(csv_path, dtype={"kabkota_code": str, "kabkota_name": str}, parse_dates=["week_start_date"])
    except Exception:
        return pd.DataFrame()
    df = df_all[df_all["kabkota_code"].astype(str) == str(kabkota_code)].copy()
    if df.empty:
        return df
    # expected feature cols: t_mean,t_min,t_max,hu_mean,ws_mean,tcc_mean,rain_hours,rain_frac,hours
    return df.sort_values("week_start_date").reset_index(drop=True)

# =========================
# Geocoding & Forecast (for 7d ahead features)
# =========================
@st.cache_data(ttl=6*60*60, show_spinner=False)
def geocode_open_meteo(place_variants: list[str]):
    cache = load_geocode_cache()
    for p in place_variants:
        hit = cache[cache["place"] == p]
        if not hit.empty:
            return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"]), f"cache:{p}"
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
    }
    r = requests.get(OPEN_METEO_FC, params=params, timeout=30, headers={"User-Agent":"SIGER-DBD/streamlit"})
    r.raise_for_status()
    j = r.json()
    h = j.get("hourly", {})
    if not h or "time" not in h:
        return pd.DataFrame()
    df = pd.DataFrame({
        "local_datetime": pd.to_datetime(h["time"]),
        "temperature_2m": h.get("temperature_2m"),
        "relative_humidity_2m": h.get("relative_humidity_2m"),
        "wind_speed_10m": h.get("wind_speed_10m"),
        "cloud_cover": h.get("cloud_cover"),
        "precipitation": h.get("precipitation"),
    })
    return df

def summarize_7d_like_weekly(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame()
    d = df_hourly.copy()
    d["is_rain"] = (d["precipitation"].fillna(0) > 0).astype(int)
    out = pd.DataFrame({
        "t_mean": [np.nanmean(d["temperature_2m"])],
        "t_min": [np.nanmin(d["temperature_2m"])],
        "t_max": [np.nanmax(d["temperature_2m"])],
        "hu_mean": [np.nanmean(d["relative_humidity_2m"])],
        "ws_mean": [np.nanmean(d["wind_speed_10m"])],
        "tcc_mean": [np.nanmean(d["cloud_cover"])],
        "rain_hours": [int(d["is_rain"].sum())],
        "rain_frac": [float(d["is_rain"].sum()) / float(len(d)) if len(d) else 0.0],
        "hours": [int(len(d))]
    })
    return out

# =========================
# Feature engineering & model
# =========================
def build_lag_features(df_inc: pd.DataFrame, max_lag=4) -> pd.DataFrame:
    df = df_inc.sort_values("week_start_date").copy()
    for L in range(1, max_lag+1):
        df[f"incidence_per_100k_lag{L}"] = df["incidence_per_100k"].shift(L)
        df[f"outbreak_lag{L}"] = df["outbreak"].shift(L)
    return df

def merge_hist_inc_weather(df_inc: pd.DataFrame, df_wx: pd.DataFrame) -> pd.DataFrame:
    if df_inc.empty or df_wx.empty:
        return pd.DataFrame()
    df = pd.merge(df_inc, df_wx, on=["kabkota_code","kabkota_name","week_start_date"], how="inner")
    df = build_lag_features(df, max_lag=4)
    df = df.dropna().reset_index(drop=True)
    return df

def train_rf_earlyfusion(df_hist: pd.DataFrame):
    if df_hist.empty or df_hist.shape[0] < 16:
        return None, None
    feat_cols = [c for c in df_hist.columns if c.startswith("incidence_per_100k_lag")] + \
                [c for c in df_hist.columns if c.startswith("outbreak_lag")] + \
                ["t_mean","hu_mean","ws_mean","tcc_mean","rain_frac"]
    feat_cols = [c for c in feat_cols if c in df_hist.columns]
    X = df_hist[feat_cols]
    y = df_hist["outbreak"].astype(int)
    if y.nunique() < 2:
        return ("PRIOR", float(y.mean())), feat_cols
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500, min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=42
        ))
    ])
    pipe.fit(X, y)
    return pipe, feat_cols

def predict_proba_with_model(model, feat_cols, df_featrow: pd.DataFrame) -> float:
    if model is None:
        return 0.5
    if isinstance(model, tuple) and model[0] == "PRIOR":
        return float(model[1])
    X = df_featrow.reindex(columns=feat_cols, fill_value=np.nan)
    if X.isna().any().any():
        return 0.5
    proba = model.predict_proba(X)
    classes_ = list(model.named_steps["rf"].classes_)
    if 1 in classes_:
        idx = classes_.index(1)
        return float(proba[:, idx][0])
    return 0.0

def risk_label_from_score(score: float) -> str:
    if score >= 0.66: return "High"
    if score >= 0.33: return "Medium"
    return "Low"

def risk_badge(label: str) -> str:
    colors = {
        "High":   ("#e53935", "#fff"),
        "Medium": ("#fdd835", "#000"),
        "Low":    ("#43a047", "#fff"),
    }
    bg, fg = colors.get(label, ("#999","#fff"))
    return '<span style="display:inline-block;padding:6px 12px;border-radius:999px;background:%s;color:%s;font-weight:700;">%s</span>' % (bg, fg, label)

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

st.write(f"📍 **Provinsi**: {prov_name} (`{prov_code}`)  \n🏙️ **Kab/Kota**: {kabkota_name} (`{kabkota_code}`)  \n🗓️ **Horizon Prediksi**: **{HORIZON_DAYS} hari ke depan**")

# ===== Load historical datasets =====
st.subheader("0) Data Historis (Insidensi + Cuaca Mingguan)")
df_inc_all = load_incidence_csv(INCIDENCE_CSV, kabkota_code)
df_wx_all  = load_weather_weekly(WEATHER_WEEKLY_CSV, kabkota_code)

c0a, c0b = st.columns(2)
with c0a:
    st.write("Insidensi (52 mgr terakhir, preview):")
    st.dataframe(df_inc_all.tail(10))
with c0b:
    st.write("Cuaca Mingguan (preview):")
    st.dataframe(df_wx_all.tail(10))

df_hist = merge_hist_inc_weather(df_inc_all, df_wx_all)
if df_hist.empty:
    st.error("Tidak ada irisan minggu antara insidensi & cuaca mingguan. Pastikan kamu sudah membangun `data/weather_weekly_agg.csv` dengan tools dan rentangnya mencakup minggu-minggu insidensi.")
    st.stop()

st.info(f"Hist rows: {len(df_hist)} (setelah lag),  Positive (outbreak=1): {int(df_hist['outbreak'].sum())}  |  Negative: {len(df_hist)-int(df_hist['outbreak'].sum())}")

# ===== Train early-fusion RF =====
st.subheader("1) Model Early‑Fusion (RF)")
model, feat_cols = train_rf_earlyfusion(df_hist)
if model is None:
    st.error("Data historis terlalu sedikit untuk melatih model.")
    st.stop()
st.write(f"Fitur yang dipakai: {', '.join(feat_cols)}")

# ===== Forecast 7d → weekly-like features =====
st.subheader("2) Cuaca 7 Hari ke Depan → Fitur Mingguan")
# lat/lon source
lat, lon, source = None, None, None
if pd.notna(lat_override) and pd.notna(lon_override):
    try:
        lat = float(lat_override); lon = float(lon_override); source = "csv"
    except Exception:
        lat, lon, source = None, None, None
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

st.write(f"Koordinat: lat={lat}, lon={lon} (source: {source})")
if lat is None:
    st.error("Lokasi tidak ditemukan. Tambahkan lat,lon di base_kabkota.csv.")
    st.stop()

@st.cache_data(ttl=2*60*60, show_spinner=False)
def get_hourly(lat, lon):
    return fetch_open_meteo_hourly(lat, lon, tz="Asia/Jakarta", days=HORIZON_DAYS)

df_hourly = get_hourly(lat, lon)
if df_hourly.empty:
    st.error("Open‑Meteo tidak mengembalikan data untuk lokasi ini.")
    st.stop()
st.dataframe(df_hourly.head(24))

df_future_wx = summarize_7d_like_weekly(df_hourly)
st.write("Ringkasan 7‑hari (format mingguan):")
st.dataframe(df_future_wx)

# ===== Build last row with lags + future weather for inference =====
st.subheader("3) Prediksi Risiko Outbreak (Minggu Mendatang)")
last_hist = df_hist.iloc[[-1]].copy()
wx_cols = ["t_mean","t_min","t_max","hu_mean","ws_mean","tcc_mean","rain_frac","rain_hours","hours"]
for c in wx_cols:
    if c in last_hist.columns:
        last_hist = last_hist.drop(columns=[c])
feat_row = pd.concat([last_hist.reset_index(drop=True), df_future_wx.reset_index(drop=True)], axis=1)

proba = predict_proba_with_model(model, feat_cols, feat_row)
label = risk_label_from_score(proba)

c1, c2 = st.columns(2)
with c1:
    st.metric("Probabilitas Outbreak (RF Early‑Fusion)", f"{proba:.2f}")
with c2:
    st.markdown(risk_badge(label), unsafe_allow_html=True)

st.caption("Threshold label: Low <0.33 ≤ Medium <0.66 ≤ High")

# ===== Optional export =====
if st.button("💾 Export ringkasan prediksi (CSV)"):
    out = df_future_wx.copy()
    out.insert(0, "kabkota_code", kabkota_code)
    out.insert(1, "kabkota_name", kabkota_name)
    out["rf_proba"] = proba
    out["risk_label"] = label
    out.to_csv("outputs_prediction_earlyfusion.csv", index=False, encoding="utf-8")
    st.success("Tersimpan: outputs_prediction_earlyfusion.csv (root app)")

st.caption("Mode: Early‑Fusion RF (insidensi + cuaca mingguan historis). Cuaca 7 hari ke depan dipetakan ke format mingguan untuk inferensi.")
