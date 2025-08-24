import re, os, io, csv, time, json, random
import requests
import streamlit as st

# ---------- Konfigurasi ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_BASE = os.path.join(DATA_DIR, "base.csv")
BMKG_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
SAMPLE_MAX = 6  # jumlah adm4 yang diambil per kab/kota (hemat kuota)

# ---------- Util & Loader CSV (robust) ----------
def _read_first_two_columns(path):
    """
    Baca base.csv -> kembalikan list (kode, nama)
    - autodetect delimiter (',' atau ';')
    - handle BOM/CRLF
    - header 'kode,nama' bisa ada/ tidak ada
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "rb") as fb:
        raw = fb.read()
    text = raw.decode("utf-8-sig", errors="ignore").replace("\r\n","\n").replace("\r","\n")
    sio = io.StringIO(text)

    sample = text[:2000]
    try:
        snif = csv.Sniffer().sniff(sample, delimiters=";,")
        delimiter = snif.delimiter
        has_header = csv.Sniffer().has_header(sample)
    except Exception:
        delimiter = ","
        has_header = True

    rd = csv.reader(sio, delimiter=delimiter)
    if has_header:
        next(rd, None)

    for r in rd:
        if not r: 
            continue
        kode = (r[0] if len(r)>0 else "").strip()
        nama = (r[1] if len(r)>1 else "").strip()
        if kode and nama:
            rows.append((kode, nama))
    return rows

@st.cache_data(show_spinner=False)
def build_kab_map():
    """
    Susun peta kab/kota (NN.NN) -> list ADM4 (NN.NN.NN.NNNN)
    """
    rows = _read_first_two_columns(CSV_BASE)
    RE_KAB  = re.compile(r"^\d{2}\.\d{2}$")
    RE_ADM4 = re.compile(r"^\d{2}\.\d{2}\.\d{2}\.\d{4}$")

    kab_name, kab_adm4 = {}, {}

    # nama kab/kota
    for kode, nama in rows:
        if RE_KAB.match(kode):
            kab_name[kode] = nama
            kab_adm4.setdefault(kode, set())

    # kumpulkan adm4
    for kode, _ in rows:
        if RE_ADM4.match(kode):
            parts = kode.split(".")
            kab = ".".join(parts[:2])
            kab_adm4.setdefault(kab, set()).add(kode)
            kab_name.setdefault(kab, f"Kab/Kota {kab}")

    kab_map = {}
    for kab, s in kab_adm4.items():
        kab_map[kab] = {
            "name": kab_name.get(kab, f"Kab/Kota {kab}"),
            "adm4": sorted(list(s))
        }
    return kab_map

# ---------- Ambil cuaca BMKG & ringkas ----------
def fetch_bmkg(adm4: str):
    if not adm4:
        return {"suhu": 30.0, "rh": 85.0, "rain": 60.0}
    try:
        r = requests.get(BMKG_URL.format(adm4=adm4), timeout=10)
        if not r.ok:
            return {"suhu": 30.0, "rh": 85.0, "rain": 60.0}
        j = r.json()
        cuaca = j.get("data", [{}])[0].get("cuaca", [])
        win = cuaca[:8] if isinstance(cuaca, list) else []
        temps, rhs, rains_rr, rains_proxy = [], [], [], []
        for slot in win:
            t = slot.get("t"); hu = slot.get("hu"); rr = slot.get("rr")
            if t is not None:
                try: temps.append(float(t))
                except: pass
            if hu is not None:
                try: rhs.append(float(hu))
                except: pass
            if rr is not None:
                try: rains_rr.append(float(rr))
                except: pass
            desc = (slot.get("weather_desc") or "").lower()
            if any(k in desc for k in ["hujan deras","rain heavy","thunderstorm"]):
                rains_proxy.append(50.0)
            elif any(k in desc for k in ["hujan","rain","gerimis","drizzle","shower"]):
                rains_proxy.append(15.0)
            else:
                rains_proxy.append(0.0)
        def mean(vs): return sum(vs)/len(vs) if vs else None
        suhu = mean(temps) or 30.0
        rh = mean(rhs) or 85.0
        rain = mean(rains_rr) if rains_rr else mean(rains_proxy)
        rain = max(0.0, (rain or 0.0))
        return {"suhu": round(suhu,1), "rh": round(min(max(rh,0.0),100.0),1), "rain": round(rain,1)}
    except Exception:
        return {"suhu": 30.0, "rh": 85.0, "rain": 60.0}

# ---------- Rule-based (cuaca + insidensi opsional) ----------
def _score_cuaca(wx):
    suhu = float(wx.get("suhu", 30.0))
    rh   = float(wx.get("rh",   85.0))
    rain = float(wx.get("rain", 60.0))

    if suhu <= 24: suhu_score = 0.2
    elif suhu >= 35: suhu_score = 0.2
    elif 27 <= suhu <= 32: suhu_score = 1.0
    else:
        if 24 < suhu < 27:
            suhu_score = 0.2 + (suhu - 24) * (0.8/3.0)
        else:
            suhu_score = 1.0 - (suhu - 32) * (0.8/3.0)

    rh_score   = 0.0 if rh < 60 else (0.3 if rh < 75 else 0.6)
    rain_score = 0.0 if rain < 10 else (0.2 if rain < 50 else 0.6)

    cuaca_score = max(0.0, min(1.0, 0.5*suhu_score + 0.25*rh_score + 0.25*rain_score))
    return round(cuaca_score, 2)

def _score_insidensi(cases, population):
    try:
        cases = float(cases); pop = float(population)
        ir = (cases / pop) * 100000.0 if pop > 0 else 0.0
    except:
        ir = 0.0
    if ir < 10: ins_score = 0.2
    elif ir < 50: ins_score = 0.5
    else: ins_score = 0.85
    return round(ir,2), round(ins_score,2)

def risk_rule(wx, cases=None, population=None, w_cuaca=0.4, w_ins=0.6):
    cuaca_s = _score_cuaca(wx)
    if cases is not None and population is not None:
        ir, ins_s = _score_insidensi(cases, population)
    else:
        ir, ins_s = None, 0.0
        w_cuaca, w_ins = 1.0, 0.0
    risk = max(0.0, min(1.0, w_cuaca*cuaca_s + w_ins*ins_s))
    if risk >= 0.70: kategori = "TINGGI"
    elif risk >= 0.40: kategori = "SEDANG"
    else: kategori = "RENDAH"
    return {
        "cuaca_score": cuaca_s,
        "ir": ir,
        "ins_score": ins_s,
        "risk": kategori,
        "risk_value": round(risk,2)
    }

# ========== UI ==========
st.set_page_config(page_title="SIGER (Streamlit)", page_icon="🛡️", layout="centered")
st.title("🛡️ SIGER — DBD Early Warning (Streamlit)")

# Sidebar: input insidensi (opsional)
with st.sidebar:
    st.header("📈 Data Insidensi (opsional)")
    use_ins = st.checkbox("Gabungkan insidensi kasus", value=False)
    cases = st.number_input("Kasus periode berjalan", min_value=0, value=0, step=1)
    pop   = st.number_input("Populasi wilayah", min_value=1, value=100000, step=1000)
    st.caption("Jika tidak dicentang, skor dihitung dari cuaca saja.")

# Load data kab/kota
with st.spinner("Memuat daftar kab/kota dari base.csv ..."):
    KAB = build_kab_map()

if not KAB:
    st.error("KAB_MAP kosong. Pastikan file data/base.csv ada & berformat 'kode,nama'.")
    st.stop()

# Dropdown kab/kota
kab_opts = [(k, v["name"]) for k, v in sorted(KAB.items())]
labels = [f"{k} — {name}" for k, name in kab_opts]
idx = st.selectbox("Pilih Kab/Kota", options=range(len(labels)), format_func=lambda i: labels[i])

kab_code, kab_name = kab_opts[idx]
adm4_list = KAB[kab_code]["adm4"]
st.write(f"**{kab_name}** • {len(adm4_list)} kel/desa (ADM4)")

# Tombol prediksi
if st.button("🔮 Prediksi Risiko Kab/Kota"):
    sample = adm4_list[:SAMPLE_MAX]
    suhus, rhs, rains, detail = [], [], [], []
    prog = st.progress(0)
    for i, a in enumerate(sample):
        wx = fetch_bmkg(a)
        suhus.append(wx["suhu"]); rhs.append(wx["rh"]); rains.append(wx["rain"])
        detail.append({"adm4": a, "cuaca": wx})
        prog.progress((i+1)/len(sample))
        time.sleep(0.2)  # sopan ke API

    def mean(vs): return round(sum(vs)/len(vs), 2) if vs else None
    agg = {"suhu": mean(suhus), "rh": mean(rhs), "rain": mean(rains)}
    rr = risk_rule(agg, cases, pop) if use_ins else risk_rule(agg)

    st.subheader("Hasil")
    st.metric("Risiko", rr["risk"], f"Skor {rr['risk_value']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Suhu (°C)", agg["suhu"])
    col2.metric("RH (%)", agg["rh"])
    col3.metric("Hujan (mm)", agg["rain"])
    if use_ins and rr["ir"] is not None:
        st.caption(f"IR per 100.000 = **{rr['ir']}** (skor {rr['ins_score']})")
    with st.expander("Rincian sampel ADM4 yang diambil"):
        st.json(detail)
