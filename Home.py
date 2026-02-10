import streamlit as st
import os

# 1. Konfigurasi Halaman (Wajib di baris pertama)
st.set_page_config(
    page_title="Prediksi Penyakit Paru-Paru",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS untuk mempercantik tampilan (Card style)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    .feature-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar (Informasi Proyek)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022343.png", width=100) # Icon Paru-paru
    st.title("Tentang Aplikasi")
    st.info("""
    Aplikasi ini dikembangkan untuk membantu **deteksi dini** risiko penyakit paru-paru berdasarkan data medis pasien.
    """)
    
    st.markdown("---")
    st.markdown("**Metode:** Naïve Bayes Classifier")
    st.markdown("**Framework:** Laravel & Streamlit")
    st.markdown("---")
    st.caption("© 2024 Skripsi/Tesis Project")

# 4. Header / Judul Utama
st.title("🫁 Sistem Prediksi Penyakit Paru-Paru")
st.markdown("### *Early Prediction of Lung Disease Using Naïve Bayes*")
st.markdown("---")

# 5. Bagian Pengantar (Hero Section)
col_hero1, col_hero2 = st.columns([2, 1])

with col_hero1:
    st.success("👋 **Selamat Datang!**")
    st.write("""
    Sistem ini memanfaatkan kecerdasan buatan untuk menganalisis pola kesehatan pasien. 
    Dengan mengunggah data medis, sistem akan melakukan klasifikasi risiko penyakit paru-paru secara otomatis.
    """)
    st.write("**Langkah Mudah Menggunakan Aplikasi:**")
    st.markdown("1. Unduh sampel dataset di bawah.")
    st.markdown("2. Pergi ke menu **Dataset** untuk upload file.")
    st.markdown("3. Latih model dan lihat hasil evaluasi di **Dashboard**.")

with col_hero2:
    # Menampilkan metrik dummy atau informasi singkat agar visual seimbang
    st.metric(label="Akurasi Model Terakhir", value="92%", delta="Optimal")
    st.metric(label="Total Data Latih", value="500+", delta="Records")

st.markdown("---")

# 6. Fitur Utama (Menggunakan Columns agar tidak list memanjang ke bawah)
st.subheader("🚀 Fitur Utama")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📂 **Manajemen Dataset**")
    st.caption("Upload dataset CSV/Excel, lihat preview data, dan tentukan variabel target untuk pelatihan.")

with col2:
    st.warning("🧠 **Pelatihan Model**")
    st.caption("Latih model Naïve Bayes secara real-time dan simpan model terbaik untuk prediksi API.")

with col3:
    st.error("📊 **Dashboard Evaluasi**")
    st.caption("Visualisasi Confusion Matrix, Laporan Klasifikasi, dan Metrik Akurasi secara interaktif.")

st.markdown("---")

# 7. Area Download (Dibuat dalam Expander atau Container khusus)
st.subheader("📥 Area Download")

with st.container():
    col_dl_text, col_dl_btn = st.columns([3, 1])
    
    with col_dl_text:
        st.write("**Belum memiliki data?**")
        st.write("Unduh sampel dataset berikut untuk menguji coba fungsionalitas sistem sebelum menggunakan data riil.")
    
    with col_dl_btn:
        # File Path
        file_path = "sample_dataset.csv" # Pastikan file ini ada di folder root project

# Footer kecil
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: grey;'>Dibuat dengan ❤️ menggunakan Streamlit</div>", unsafe_allow_html=True)

