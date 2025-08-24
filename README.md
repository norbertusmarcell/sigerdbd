# SIGER-DBD – Open‑Meteo (7 Hari) + Insidensi

Aplikasi Streamlit untuk memprediksi risiko **outbreak DBD** di level **Kab/Kota**, dengan:
- Cuaca **7 hari** (hourly) dari **Open‑Meteo** → diringkas jadi fitur: `t_mean_7d, hu_mean_7d, rain_frac`, dst.
- Riwayat insidensi **mingguan 52w** (dummy baseline) → **lag(1..4)** → **Random Forest** (probabilitas outbreak).
- **Fusi skor**: `final = 0.6*hist_proba + 0.4*wx_score` → label Low/Medium/High.

## Struktur Repo
```
.
├─ streamlit_app_openmeteo_7d.py        # main app (fixed 7 hari)
├─ requirements.txt
├─ .gitignore
├─ .streamlit/
│  └─ config.toml                       # (opsional) tema
├─ data/
│  ├─ base_kabkota.csv                  # daftar Prov + Kab/Kota
│  └─ README.md
└─ tools/
   └─ convert_kabkota_to_base_kabkota.py
```

## Menjalankan Lokal
```bash
pip install -r requirements.txt
streamlit run streamlit_app_openmeteo_7d.py
```

## Deploy di Streamlit Cloud
- **Main file**: `streamlit_app_openmeteo_7d.py`
- Pastikan file `data/base_kabkota.csv` ikut ter-commit.
- (Tidak perlu API key).

## Ubah Data Wilayah
Jika kamu punya CSV lain berisi `kode,nama` (kab/kota):
```bash
python tools/convert_kabkota_to_base_kabkota.py   --src data/kabkota_jakarta_bali_lampung_from_base.csv   --out data/base_kabkota.csv
```
