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
st.title("SIGER-DBD – Random Forest Early-Fusion (Historis + Cuaca Harian→Mingguan)")

# =========================
# Name normalization & aliases (DKI)
# =========================
ALIASES = {
    "KAB. ADM. KEP. SERIBU": "Kepulauan Seribu",
    "KAB ADM KEP SERIBU": "Kepulauan Seribu",
    "KOTA ADM. JAKARTA PUSAT": "Jakarta Pusat",
    "KOTA ADM. JAKARTA UTARA": "Jakarta Utara",
    "KOTA ADM. JAKARTA BARAT": "Jakarta Barat",
    "
