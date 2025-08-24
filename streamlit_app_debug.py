
import math
import re
import time
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================
# Config
# =============================
BASE_CSV_PATH = "data/base.csv"
BMKG_BASE = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
HUJAN = re.compile(r"hujan", re.I)

# =============================
# HTTP session with retry
# =============================
def make_session(retries=3, backoff=0.2, timeout=20) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['GET']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({
        "User-Agent": "SIGER-DBD/1.0 (+streamlit)",
        "Accept": "application/json",
    })
    sess.request_timeout = timeout
    return sess

# =============================
# Helpers
# =============================
def level_of(code: str) -> int:
    return (code or "").count(".")

def load_base(path: str = BASE_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"kode": str, "nama": str})
    df["kode"] = df["kode"].astype(str).str.strip()
    df["nama"] = df["nama"].astype(str).str.strip()
    return df

def get_provinces(df_base: pd.DataFrame) -> pd.DataFrame:
    return df_base[df_base["kode"].apply(level_of) == 0][["kode","nama"]].drop_duplicates().sort_values("kode")

def get_kabkota_by_prov(df_base: pd.DataFrame, prov_code: str) -> pd.DataFrame:
    mask = df_base["kode"].astype(str).str.startswith(str(prov_code) + ".") & (df_base["kode"].apply(level_of) == 1)
    return df_base[mask][["kode","nama"]].drop_duplicates().sort_values("kode")

def get_adm4_in_kabkota(df_base: pd.DataFrame, kabkota_code: str) -> pd.DataFrame:
    mask = df_base["kode"].astype(str).str.startswith(str(kabkota_code) + ".") & (df_base["kode"].apply(level_of) == 3)
    return df_base[mask][["kode","nama"]].drop_duplicates().sort_values("kode")

# =============================
# BMKG fetchers
# =============================
def fetch_forecast_adm4_session(s: requests.Session, adm4: str) -> Tuple[pd.DataFrame, str]:
    url = f"{BMKG_BASE}?adm4={adm4}"
    try:
        r = s.get(url, timeout=s.request_timeout)
        status = f"{r.status_code} {r.reason}; elapsed={r.elapsed.total_seconds():.2f}s"
        if r.status_code != 200:
            return pd.DataFrame(), status
        j = r.json()
        rows = []
        for loc in j.get("data", []):
            adm4_code = loc.get("adm4", adm4)
            for it in loc.get("forecasts", []):
                rows.append({
                    "adm4": adm4_code,
                    "utc_datetime": it.get("utc_datetime"),
                    "local_datetime": it.get("local_datetime"),
                    "t": pd.to_numeric(it.get("t"), errors="coerce"),
                    "hu": pd.to_numeric(it.get("hu"), errors="coerce"),
                    "ws": pd.to_numeric(it.get("ws"), errors="coerce"),
                    "wd": it.get("wd"),
                    "tcc": pd.to_numeric(it.get("tcc"), errors="coerce"),
                    "weather_desc": it.get("weather_desc"),
                    "weather_desc_en": it.get("weather_desc_en"),
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
        return df, status
    except Exception as e:
        return pd.DataFrame(), f"EXC: {e}"

@st.cache_data(ttl=2*60*60, show_spinner=False)
def fetch_many(adm4_codes: List[str], retries=3, timeout=20):
    s = make_session(retries=retries, timeout=timeout)
    frames, info = [], []
    for i, code in enumerate(adm4_codes, start=1):
        df, status = fetch_forecast_adm4_session(s, code)
        info.append({"adm4": code, "status": status, "rows": len(df)})
        if not df.empty:
            frames.append(df.assign(adm4=code))
        if i % 60 == 0:
            time.sleep(1.2)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            pd.DataFrame(info))

# =============================
# Feature aggregation
# =============================
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
    for c in ["t_mean_3d","t_min_3d","t_max_3d","hu_mean_3d","ws_mean_3d","tcc_mean_3d","rain_slots_3d"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    return g

def aggregate_to_kab(df_adm4_feats: pd.DataFrame, kabkota_name: str) -> pd.DataFrame:
    if df_adm4_feats.empty:
        return pd.DataFrame()
    num_cols = [c for c in df_adm4_feats.columns if c != "adm4"]
    out = df_adm4_feats[num_cols].mean(numeric_only=True).to_frame().T
    out.insert(0, "kabkota", kabkota_name)
    return out

# =============================
# UI
# =============================
st.set_page_config(page_title="SIGER-DBD – Debug BMKG Fetch", layout="wide")
st.title("SIGER-DBD – Debug BMKG Fetch (ADM4 → 3-day features)")

# Controls
st.sidebar.header("Config")
retries = st.sidebar.slider("HTTP retries", 0, 6, 3)
timeout = st.sidebar.slider("Timeout (detik)", 5, 60, 20)
sample_n = st.sidebar.slider("Batas ADM4", 1, 200, 20)

# Load base
try:
    df_base = load_base(BASE_CSV_PATH)
except Exception as e:
    st.error(f"Gagal baca {BASE_CSV_PATH}: {e}")
    st.stop()

# Select prov/kab
provs = get_provinces(df_base).to_dict("records")
sel_prov = st.selectbox("Provinsi", provs, format_func=lambda r: f"{r['nama']} ({r['kode']})")
kab = get_kabkota_by_prov(df_base, sel_prov["kode"]).to_dict("records")
sel_kab = st.selectbox("Kab/Kota", kab, format_func=lambda r: f"{r['nama']} ({r['kode']})")
adm4_df = get_adm4_in_kabkota(df_base, sel_kab["kode"])
adm4_list = adm4_df["kode"].tolist()[:sample_n]

st.write(f"ADM4 sample: {len(adm4_list)}")

# Test single ADM4
st.subheader("Tes 1 ADM4 Langsung")
test_adm4 = st.selectbox("Pilih 1 ADM4 untuk tes", adm4_df.to_dict("records"), format_func=lambda r: f"{r['nama']} ({r['kode']})")
if st.button("Coba Fetch ADM4 ini"):
    s = make_session(retries=retries, timeout=timeout)
    df_one, status = fetch_forecast_adm4_session(s, test_adm4["kode"])
    st.write("Status:", status)
    if df_one.empty:
        st.error("Kosong untuk ADM4 ini.")
    else:
        st.success(f"Dapat {len(df_one)} baris.")
        st.dataframe(df_one.head(20))

# Fetch many
st.subheader("Fetch Banyak ADM4")
df_fc, df_log = fetch_many(adm4_list, retries=retries, timeout=timeout)
st.dataframe(df_log)

ok = (df_log["rows"] > 0).sum()
err = (df_log["rows"] == 0).sum()
st.write(f"Ringkas: **{ok} ADM4 ada data**, **{err} ADM4 kosong**.")

if df_fc.empty:
    st.error("Tidak ada data prakiraan (semua ADM4 kosong).")
else:
    st.success(f"Total baris prakiraan: {len(df_fc)}")
    st.dataframe(df_fc.head(30))
    st.subheader("Agregasi → Fitur 3-hari")
    df_adm4_feats = summarize_3day_slots(df_fc)
    st.dataframe(df_adm4_feats.head(30))
    df_kab = aggregate_to_kab(df_adm4_feats, sel_kab["nama"])
    st.dataframe(df_kab)
