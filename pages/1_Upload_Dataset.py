import streamlit as st
import pandas as pd
import os
import requests
from services.ml_service import train_and_evaluate
# from utils.helpers import load_dataset # (Opsional: Kita pakai pandas langsung agar lebih aman membaca path file)

st.title("📂 Dataset & Training (Server File)")

# --- KONFIGURASI LOKASI FILE ---
# Tentukan lokasi file dataset yang "sudah terupload" / tersedia di server
# Pastikan file ini benar-benar ada di folder proyek Anda
DATASET_PATH = "data/dataset_paru_paru.csv" 

# Cek apakah file ada
if os.path.exists(dataset_paruparu):
    # Tampilkan info file
    st.info(f"Menggunakan dataset dari server: **{dataset_paruparu}**")
    
    # 1. Load Dataset langsung dari path
    # Kita gunakan pd.read_csv langsung karena load_dataset bawaan helper 
    # mungkin dirancang khusus untuk objek st.file_uploader, bukan string path.
    try:
        if DATASET_PATH.endswith('.csv'):
            df = pd.read_csv(dataset_paruparu)
        else:
            df = pd.read_excel(dataset_paruparu)
            
        # Simpan ke session state
        st.session_state["uploaded_df"] = df
        st.session_state["uploaded_filename"] = os.path.basename(dataset_paruparu)

        # 2. Tampilkan Sample Data
        st.subheader("📊 Data Sample")
        st.dataframe(df.head(), use_container_width=True)

        # 3. Pilih Target Column
        # (Logika sama seperti sebelumnya)
        target_col = st.selectbox("Pilih kolom target:", df.columns, index=len(df.columns)-1)
        st.session_state["target_col"] = target_col

        # 4. Tombol Training
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
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")

else:
    st.error(f"❌ File tidak ditemukan: {DATASET_PATH}")


