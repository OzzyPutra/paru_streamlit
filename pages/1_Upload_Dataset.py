import streamlit as st
import pandas as pd
import os
import requests
from services.ml_service import train_and_evaluate

st.title("📂 Dataset & Training (Server File)")

# --- KONFIGURASI LOKASI FILE ---
DATASET_PATH = "data/dataset_paruparu.csv"  # <--- Ini nama variabel yang benar

# Cek apakah file ada (Gunakan DATASET_PATH, bukan dataset_paruparu)
if os.path.exists(DATASET_PATH):
    # Tampilkan info file
    st.info(f"Menggunakan dataset dari server: **{DATASET_PATH}**")
    
    try:
        # Load Dataset
        if DATASET_PATH.endswith('.csv'):
            df = pd.read_csv(DATASET_PATH) # <--- Perbaikan disini
        else:
            df = pd.read_excel(DATASET_PATH) # <--- Perbaikan disini
            
        # Simpan ke session state
        st.session_state["uploaded_df"] = df
        st.session_state["uploaded_filename"] = os.path.basename(DATASET_PATH) # <--- Perbaikan disini

        # Tampilkan Sample Data
        st.subheader("📊 Data Sample")
        st.dataframe(df.head(), use_container_width=True)

        # Pilih Target Column
        target_col = st.selectbox("Pilih kolom target:", df.columns, index=len(df.columns)-1)
        st.session_state["target_col"] = target_col

        # Tombol Training
        if st.button("Latih & Simpan Model"):
            with st.spinner("Sedang melatih model..."):
                acc, cm, report = train_and_evaluate(df, target_col)

            st.success(f"✅ Model berhasil dilatih & disimpan. Akurasi: {acc:.2%}")

            # Simpan hasil evaluasi
            st.session_state["evaluation"] = {
                "acc": acc,
                "cm": cm.tolist(),
                "report": report,
                "target_col": target_col
            }
            
            # --- Tambahan Logic API Reload & Download (Opsional) ---
            # (Tambahkan kode reload API di sini jika diperlukan seperti kode sebelumnya)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")

else:
    st.error(f"❌ File tidak ditemukan: {DATASET_PATH}")
