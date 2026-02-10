import streamlit as st
import os
st.set_page_config(page_title="Prediksi Penyakit Paru-Paru", layout="wide")

st.title("🫁 Aplikasi Prediksi Penyakit Paru-Paru")
st.markdown("""
Selamat datang di aplikasi prediksi penyakit paru-paru menggunakan **Naïve Bayes**.

📂 Menu:
- **Upload Dataset** → untuk mengunggah data & melatih model.
- **Dashboard** → untuk melihat hasil evaluasi model (akurasi, confusion matrix, laporan).
""")
st.divider() 

st.subheader("📥 Unduh Sampel Dataset")
st.write("Unduh contoh format file CSV di bawah ini untuk dicoba.")

# Tentukan nama file sampel yang ada di folder proyek Anda
file_path = "dataset_paruparu.csv"  # Ganti dengan nama file CSV Anda yang sebenarnya

# Cek apakah file ada untuk menghindari error
if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label="📄 Download Sampel CSV",
            data=file,
            file_name="dataset_paruparu.csv",
            mime="text/csv"
        )
else:
    st.warning(f"File sampel '{file_path}' tidak ditemukan. Pastikan file ada di folder proyek.")



