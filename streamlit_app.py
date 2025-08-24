
import hashlib
import math
import re
import time
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================
# Config & constants
# ============================================
BASE_CSV_PATH = "data/base.csv"  # pastikan file ini ada di repo kamu
BMKG_BASE = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
HUJAN = re.compile(r"hujan", re.I)
FEATURES_WEATHER = ["t_mean_3d","t_min_3d","t_max_3d","hu_mean_3d","ws_mean_3d","tcc_mean_3d","rain_slots_3d"]

# ============================================
# Utilities
# ============================================
def level_of(code: str) -> int:
    return code.count(".")

def load_base(path: str = BASE_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"kode": str, "nama": str})
    df["kode"] = df["kode"].str.strip()
    df["nama"] = df["nama"].str.strip()
    return df

def get_provinces(df_base: pd.DataFrame) -> pd.DataFrame:
    return df_base[df_base["kode"].apply(level_of) == 0][["kode","nama"]].sort_values("kode")

def get_kabkota_by_prov(df_base: pd.DataFrame, prov_code: str) -> pd.DataFrame:
    mask = df_base["kode"].str.startswith(prov_code + ".") & (df_base["kode"].apply(level_of) == 1)
    return df_base[mask][["kode","nama"]].sort_values("kode")

def get_adm4_in_kabkota(df_base: pd.DataFrame, kabkota_code: str) -> pd.DataFrame:
    mask = df_base["kode"].str.startswith(kabkota_code + ".") & (df_base["kode"].apply(level_of) == 3)
    return df_base[mask][["kode","nama"]].sort_values("kode")

# ============================================
# BMKG Open Data client
# ============================================
@st.cache_data(ttl=3*60*60, show_spinner=False)
def fetch_forecast_adm4(adm4: str, timeout=20) -> pd.DataFrame:
    url = f"{BMKG_BASE}?adm4={adm4}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    rows = []
    for loc in j.get("data", []):
        adm4_code = loc.get("adm4", adm4)
        for it in loc.get("forecasts", []):
            rows.append({
                "adm4": adm4_code,
                "utc_datetime": it.get("utc_datetime"),
                "local_datetime": it.get("local_datetime"),
                "t": it.get("t"),
                "hu": it.get("hu"),
                "ws": it.get("ws"),
                "wd": it.get("wd"),
                "tcc": it.get("tcc"),
                "weather_desc": it.get("weather_desc"),
                "weather_desc_en": it.get("weather_desc_en"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
        for c in ["t","hu","ws","tcc"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_many_forecasts(adm4_codes: List[str]) -> pd.DataFrame:
    frames = []
    if len(adm4_codes) == 0:
        return pd.DataFrame()
    prog = st.progress(0.0, text="Mengambil data BMKG per-ADM4...")
    n = len(adm4_codes)
    for i, code in enumerate(adm4_codes, start=1):
        try:
            df = fetch_forecast_adm4(code)
            if not df.empty:
                frames.append(df.assign(adm4=code))
        except Exception as e:
            st.warning(f"Gagal ADM4 {code}: {e}")
        prog.progress(i/n, text=f"Ambil ADM4 {i}/{n}")
        if i % 60 == 0:
            time.sleep(1.2)  # jaga rate limit 60 req/menit/IP
    prog.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ============================================
# Feature engineering (3-day aggregation)
# ============================================
def summarize_3day_slots(df_fc: pd.DataFrame) -> pd.DataFrame:
    if df_fc.empty:
        return pd.DataFrame()
    df = df_fc.copy()
    df["is_rain"] = df["weather_desc"].fillna("").apply(lambda s: 1 if HUJAN.search(s) else 0)
    agg_map = {
        "t": ["mean","min","max"],
        "hu": ["mean"],
        "ws": ["mean"],
        "tcc": ["mean"],
        "is_rain": ["sum"]
    }
    g = df.groupby("adm4").agg(agg_map)
    g.columns = ["_".join([c for c in col if c]) for col in g.columns.values]
    g = g.reset_index().rename(columns={
        "t_mean":"t_mean_3d",
        "t_min":"t_min_3d",
        "t_max":"t_max_3d",
        "hu_mean":"hu_mean_3d",
        "ws_mean":"ws_mean_3d",
        "tcc_mean":"tcc_mean_3d",
        "is_rain_sum":"rain_slots_3d"
    })
    for c in FEATURES_WEATHER:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    return g

def aggregate_adm4_to_kabkota(df_adm4_feats: pd.DataFrame, kabkota_name: str) -> pd.DataFrame:
    if df_adm4_feats.empty:
        return pd.DataFrame()
    num_cols = [c for c in df_adm4_feats.columns if c not in ["adm4"]]
    out = df_adm4_feats[num_cols].mean(numeric_only=True).to_frame().T
    out.insert(0, "kabkota", kabkota_name)
    return out

# ============================================
# Incidence dummy (weekly) per reference
# - 52 weeks history
# - risk label via monthly thresholds converted to weekly (x4)
# ============================================
def seed_from_text(txt: str) -> int:
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def generate_dummy_incidence(kabkota_code: str, kabkota_name: str, weeks=52) -> pd.DataFrame:
    today = date.today()
    last_monday = today - timedelta(days=today.weekday())
    week_starts = [last_monday - timedelta(weeks=i) for i in range(1, weeks+1)]
    week_starts = sorted(week_starts)

    rng = np.random.default_rng(seed_from_text(kabkota_code))
    pop = int(rng.integers(150_000, 2_500_000))
    base_lambda = 14 if "KOTA" in kabkota_name.upper() else 10

    rows = []
    for i, wk in enumerate(week_starts):
        seasonal = 1.0 + 0.5*np.sin(2*np.pi*(i/52.0))
        lam = max(1.0, base_lambda * seasonal + rng.normal(0, 2))
        cases = int(rng.poisson(lam))
        inc_per_100k = (cases / pop) * 100000.0
        monthly_equiv = inc_per_100k * 4.0  # weekly -> monthly-equivalent
        if monthly_equiv < 3.0:
            label = "safe"
        elif monthly_equiv <= 10.0:
            label = "moderately_safe"
        else:
            label = "unsafe"
        rows.append({
            "kabkota_code": kabkota_code,
            "kabkota_name": kabkota_name,
            "week_start_date": wk.isoformat(),
            "cases": cases,
            "population": pop,
            "incidence_per_100k": round(inc_per_100k, 4),
            "risk_label": label
        })
    df = pd.DataFrame(rows)
    return df

def build_lag_features(df_inc: pd.DataFrame, max_lag=4) -> pd.DataFrame:
    df = df_inc.sort_values("week_start_date").copy()
    # outbreak target (per reference: unsafe -> 1, else 0)
    df["outbreak"] = (df["risk_label"] == "unsafe").astype(int)
    for L in range(1, max_lag+1):
        df[f"incidence_per_100k_lag{L}"] = df["incidence_per_100k"].shift(L)
        df[f"outbreak_lag{L}"] = df["outbreak"].shift(L)
    return df

# ============================================
# Weather score (0..1) and fusion with incidence model
# ============================================
def weather_score(df_kab_feats: pd.DataFrame) -> float:
    if df_kab_feats.empty:
        return 0.5
    r = df_kab_feats.iloc[0]
    # scale temp 24-34C, RH 60-95%, rain_slots scaled by 24 slots (3 days * 8 slots)
    t = float(r.get("t_mean_3d", 30.0) or 30.0)
    rh = float(r.get("hu_mean_3d", 85.0) or 85.0)
    rain_slots = float(r.get("rain_slots_3d", 0.0) or 0.0)
    s_t = min(max((t - 24.0) / (34.0 - 24.0), 0.0), 1.0)
    s_rh = min(max((rh - 60.0) / (95.0 - 60.0), 0.0), 1.0)
    s_rain = min(max(rain_slots / 24.0, 0.0), 1.0)
    return round(0.4*s_t + 0.35*s_rh + 0.25*s_rain, 3)

def incidence_model_proba(df_inc_with_lag: pd.DataFrame) -> float:
    # Train on history except last row; predict next using latest lags
    df = df_inc_with_lag.dropna().copy()
    if df.shape[0] < 10:
        # not enough data
        return 0.5
    train = df.iloc[:-1].copy()
    test = df.iloc[-1:].copy()
    feat_cols = [c for c in df.columns if c.startswith("incidence_per_100k_lag")] + \
                [c for c in df.columns if c.startswith("outbreak_lag")]
    X_tr = train[feat_cols]
    y_tr = train["outbreak"].astype(int)
    X_te = test[feat_cols]

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, min_samples_split=4,
            class_weight="balanced",
            random_state=42
        ))
    ])
    pipe.fit(X_tr, y_tr)
    proba = float(pipe.predict_proba(X_te)[:,1][0])
    return proba

def fused_risk_score(weather_s: float, hist_proba: float, w_cuaca=0.4, w_hist=0.6) -> float:
    return round(w_cuaca*weather_s + w_hist*hist_proba, 3)

def risk_label_from_score(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="SIGER-DBD – BMKG (3 Hari) + Incidence (Ref) dari base.csv", layout="wide")
st.title("SIGER-DBD – BMKG 3 Hari + Riwayat Insidensi (berdasarkan referensi) – from base.csv")
st.caption("Data cuaca: © BMKG Open Data (prakiraan 3 hari, slot 3-jam, update 2×/hari). Batas 60 req/menit/IP. Label risiko insidensi mengikuti ambang bulanan (<3 aman, 3–10 sedang, >10 tidak aman) yang diekivalenkan ke mingguan (×4).")

# Load base.csv & selections
try:
    df_base = load_base(BASE_CSV_PATH)
except Exception as e:
    st.error(f"Tidak bisa membaca {BASE_CSV_PATH}. Pastikan file ada di repo. Error: {e}")
    st.stop()

st.sidebar.header("Pilih Wilayah (dari base.csv)")
provs = get_provinces(df_base)
default_prov_idx = provs["nama"].tolist().index("DKI JAKARTA") if "DKI JAKARTA" in provs["nama"].tolist() else 0
prov_name = st.sidebar.selectbox("Provinsi", provs["nama"].tolist(), index=default_prov_idx)
prov_code = provs.loc[provs["nama"] == prov_name, "kode"].iloc[0]

kabkota_df = get_kabkota_by_prov(df_base, prov_code)
kabkota_name = st.sidebar.selectbox("Kabupaten/Kota", kabkota_df["nama"].tolist())
kabkota_code = kabkota_df.loc[kabkota_df["nama"] == kabkota_name, "kode"].iloc[0]

st.write(f"📍 **Provinsi**: {prov_name} (`{prov_code}`)  \n🏙️ **Kab/Kota**: {kabkota_name} (`{kabkota_code}`)")

# ADM4 derivation
adm4_df = get_adm4_in_kabkota(df_base, kabkota_code)
total_adm4 = len(adm4_df)
st.write(f"🔎 Ditemukan **{total_adm4}** ADM4 (kel/desa) di bawah {kabkota_name}.")

max_fetch = st.sidebar.slider("Batas ADM4 yang diambil (hemat kuota)", min_value=10, max_value=max(10, total_adm4 or 10), value=min(100, total_adm4 or 10), step=10)
adm4_list = adm4_df["kode"].tolist()[:max_fetch]

# Fetch BMKG
st.subheader("1) Prakiraan Cuaca (3 hari, slot 3-jam)")
if total_adm4 == 0:
    st.warning("Tidak ditemukan ADM4 untuk kab/kota ini pada base.csv.")
    st.stop()

df_fc = fetch_many_forecasts(adm4_list)
if df_fc.empty:
    st.error("Tidak ada data prakiraan yang berhasil diambil dari BMKG.")
    st.stop()
st.dataframe(df_fc.sort_values("local_datetime"))

# Weather features
st.subheader("2) Fitur Ringkas 3-Hari (Agregasi ADM4 → Kab/Kota)")
df_adm4_feats = summarize_3day_slots(df_fc)
df_kab_wx = aggregate_adm4_to_kabkota(df_adm4_feats, kabkota_name)
st.dataframe(df_kab_wx)

# Incidence history (dummy if missing)
st.subheader("3) Riwayat Insidensi (Dummy, 52 minggu ke belakang)")
df_inc = generate_dummy_incidence(kabkota_code, kabkota_name, weeks=52)
st.dataframe(df_inc.tail(10))

# Build lags and train RF on history
st.subheader("4) Model Insidensi Berbasis Riwayat (Random Forest)")
df_inc_lag = build_lag_features(df_inc, max_lag=4)
hist_proba = incidence_model_proba(df_inc_lag)
st.write(f"🧪 Probabilitas outbreak dari **model historis** (berdasarkan lag 1..4): **{hist_proba:.3f}**")

# Weather score
st.subheader("5) Skor Cuaca (3-hari)")
wx_score = weather_score(df_kab_wx)
st.write(f"☁️ Skor cuaca komposit (0–1): **{wx_score:.3f}**  \n*(berdasarkan t_mean_3d, RH, dan rain_slots_3d)*")

# Fusion
st.subheader("6) Fusi Skor (Cuaca + Historis) → Risiko 3-Hari")
w_cuaca = st.slider("Bobot Cuaca", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
w_hist = 1.0 - w_cuaca
final_score = fused_risk_score(wx_score, hist_proba, w_cuaca=w_cuaca, w_hist=w_hist)
label = risk_label_from_score(final_score)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Prob. Outbreak (Historis)", f"{hist_proba:.2f}")
with c2:
    st.metric("Skor Cuaca (0–1)", f"{wx_score:.2f}")
with c3:
    st.metric("Risk Score (Final)", f"{final_score:.2f}")

st.success(f"Kategori Risiko: **{label}**")
st.caption("Catatan: Ambang risiko insidensi menggunakan rujukan bulanan (<3, 3–10, >10 per 100k) yang diekivalensikan ke mingguan (×4). Model historis dilatih pada lag 1..4 minggu. Fusi skor mengikuti bobot yang dapat kamu ubah.")
