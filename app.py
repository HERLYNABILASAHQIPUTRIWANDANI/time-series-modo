# ==========================================================
# app.py (VERSI MODERN - TEMA BIRU)
# Sistem Prediksi NO2 berbasis Random Forest
# ==========================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import model_utils as mu
from datetime import timedelta

# ==========================================================
# 0. KONFIGURASI STREAMLIT & TEMA
# ==========================================================
st.set_page_config(
    page_title="Prediksi NO₂ Harian",
    page_icon="💧",
    layout="wide"
)

# ==========================================================
# GAYA VISUAL (CSS - Tema Biru)
# ==========================================================
st.markdown("""
    <style>
    body {
        background-color: #f0f6fb;
    }
    .main-title {
        text-align: center;
        font-size: 2.3em;
        color: #1e3a8a;
        font-weight: 800;
        margin-bottom: 0.3em;
    }
    .subtitle {
        text-align: center;
        color: #2563eb;
        font-size: 1.1em;
        margin-bottom: 1.8em;
    }
    .stCard {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6em 1.3em;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1e40af;
        color: white;
    }
    .stMetric {
        background: #e0f2fe;
        padding: 15px;
        border-radius: 10px;
    }
    hr {
        border: 1px solid #c7d2fe;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 1. HEADER
# ==========================================================
st.markdown('<h1 class="main-title">💧 Prediksi Konsentrasi NO₂ Harian</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📘 Model <b>Random Forest</b> digunakan untuk memprediksi konsentrasi NO₂ berdasarkan data historis.</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================================
# 2. KONFIGURASI & PEMUATAN MODEL
# ==========================================================
DATA_FILE = "NO2_Modo.csv"
N_LAGS = mu.N_LAGS
TEST_SIZE = mu.TEST_SIZE_DAYS

@st.cache_resource
def load_or_train_model(file_path, n_lags, test_size):
    if not os.path.exists('model'):
        os.makedirs('model')
    try:
        results = mu.prepare_and_train_all(file_path, n_lags, test_size)
        return results['rf_model'], results['full_df'], results['last_data'], results['metrics_rf']
    except Exception as e:
        st.error(f"❌ Gagal memuat atau melatih model. Error: {e}")
        return None, None, None, None

with st.spinner("🔄 Memuat & melatih model Random Forest..."):
    rf_model, full_data, last_data, metrics = load_or_train_model(DATA_FILE, N_LAGS, TEST_SIZE)

if rf_model is None:
    st.error("Model gagal dimuat. Aplikasi dihentikan.")
    st.stop()

st.success("✅ Model Random Forest berhasil dimuat dan siap digunakan!")

last_historical_date = last_data.index[-1].date()

# ==========================================================
# 3. METRIK KINERJA MODEL
# ==========================================================
st.markdown("### 📊 Evaluasi Model Random Forest")
col1, col2 = st.columns(2)

with col1:
    st.metric("🎯 MAPE", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error (semakin kecil semakin baik)")
with col2:
    st.metric("📈 ACF Residual (Lag 1)", f"{metrics['acf']:.4f}", help="Nilai mendekati nol menunjukkan model sudah baik")

st.markdown("---")

# ==========================================================
# 4. INPUT PERIODE PREDIKSI
# ==========================================================
st.markdown("### 🗓️ Tentukan Periode Prediksi")
col_start, col_end = st.columns([1, 2])

start_date_forecast = last_historical_date + timedelta(days=1)
with col_start:
    st.date_input("Tanggal Mulai Prediksi", value=start_date_forecast, disabled=True)
    st.markdown(f"📅 <b>Data terakhir:</b> `{last_historical_date}`", unsafe_allow_html=True)

with col_end:
    target_date = st.date_input(
        "Pilih Tanggal Akhir Prediksi",
        min_value=start_date_forecast,
        value=start_date_forecast + timedelta(days=7),
        max_value=start_date_forecast + timedelta(days=60),
        help="Maksimal 60 hari dari data historis terakhir."
    )

days_to_forecast = (target_date - last_historical_date).days
st.markdown("---")

# ==========================================================
# 5. PROSES PREDIKSI
# ==========================================================
if st.button(f"🚀 Prediksi Selama {days_to_forecast} Hari (Hingga {target_date})"):
    if days_to_forecast < 1:
        st.warning("⚠️ Pilih tanggal yang lebih besar dari data historis terakhir.")
        st.stop()

    with st.spinner("🔮 Model sedang memprediksi NO₂..."):
        forecast_df = mu.predict_rf_n_days(rf_model, last_data, days_to_forecast, N_LAGS)
        forecast_df.index = pd.to_datetime(forecast_df.index)
        forecast_df = forecast_df.sort_index()

    st.success("✅ Prediksi selesai!")

    # Tambahkan label kualitas udara
    def quality_level(no2):
        if no2 < 40:
            return "🟢 Baik"
        elif no2 < 80:
            return "🟡 Sedang"
        else:
            return "🔴 Tidak Sehat"

    forecast_df["Kualitas Udara"] = forecast_df["NO2_Prediction"].apply(quality_level)
    display_df = forecast_df[["NO2_Prediction", "Kualitas Udara"]].copy()
    display_df.index.name = "Tanggal"
    display_df.columns = ["Prediksi NO₂ (µg/m³)", "Kualitas Udara"]

    # ==========================================================
    # 6. HASIL DALAM TAB
    # ==========================================================
    tab1, tab2, tab3 = st.tabs(["📊 Grafik", "📅 Tabel", "💾 Unduh"])

    with tab1:
        st.markdown("### 📊 Grafik Prediksi NO₂")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(full_data["NO2"].tail(90), label="Data Historis (90 Hari Terakhir)", color="#2563eb", linewidth=2)
        ax.axvline(x=forecast_df.index.min(), color="#475569", linestyle="--", label="Awal Prediksi")
        ax.plot(forecast_df.index, forecast_df["NO2_Prediction"], label="Prediksi NO₂", color="#1d4ed8", linewidth=2.5)
        ax.set_title(f"Prediksi Konsentrasi NO₂ Hingga {target_date}", fontsize=13, color="#1e3a8a")
        ax.set_xlabel("Tanggal"); ax.set_ylabel("Konsentrasi NO₂ (µg/m³)")
        ax.grid(alpha=0.3); ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab2:
        st.markdown("### 📅 Detail Prediksi Harian")
        st.dataframe(display_df.style.format({"Prediksi NO₂ (µg/m³)": "{:.2f}"}), use_container_width=True)

    with tab3:
        st.markdown("### 💾 Unduh Hasil Prediksi")
        csv = display_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Unduh Data (.csv)",
            data=csv,
            file_name=f"Prediksi_NO2_RF_Hingga_{target_date}.csv",
            mime="text/csv"
        )
