# SIGER-DBD (BMKG 3 Hari + Insidensi)

Aplikasi Streamlit untuk estimasi risiko DBD dengan:
- Prakiraan cuaca **BMKG Open Data** (3 hari, slot 3 jam, update 2×/hari)
- Insidensi mingguan (dummy 52 minggu) + lag 1..4 (Random Forest)
- Fusi skor historis dan cuaca → Risk Score (Low/Medium/High)

## Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
