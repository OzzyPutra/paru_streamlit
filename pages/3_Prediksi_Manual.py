import streamlit as st
import matplotlib.pyplot as plt
from services.ml_service import load_model, predict_input, save_prediction
from utils.encoding import ENCODING_MAP, get_reverse_map

st.title("🩺 Prediksi Manual Gejala Pasien")

# Load model
pack, error = load_model()
if error:
    st.warning(error)
    st.stop()

st.subheader("📝 Input Gejala Pasien")

# form: opsi sesuai standar encoding
input_data = {
    "Usia": st.selectbox("Usia", list(ENCODING_MAP["Usia"].keys())),
    "Jenis_Kelamin": st.selectbox("Jenis Kelamin", list(ENCODING_MAP["Jenis_Kelamin"].keys())),
    "Merokok": st.selectbox("Merokok", list(ENCODING_MAP["Merokok"].keys())),
    "Bekerja": st.selectbox("Bekerja", list(ENCODING_MAP["Bekerja"].keys())),
    "Rumah_Tangga": st.selectbox("Rumah Tangga", list(ENCODING_MAP["Rumah_Tangga"].keys())),
    "Aktivitas_Begadang": st.selectbox("Aktivitas Begadang", list(ENCODING_MAP["Aktivitas_Begadang"].keys())),
    "Aktivitas_Olahraga": st.selectbox("Aktivitas Olahraga", list(ENCODING_MAP["Aktivitas_Olahraga"].keys())),
    "Asuransi": st.selectbox("Asuransi", list(ENCODING_MAP["Asuransi"].keys())),
    "Penyakit_Bawaan": st.selectbox("Penyakit Bawaan", list(ENCODING_MAP["Penyakit_Bawaan"].keys()))
}

# (opsional) debug: tampilkan kelas yang dikenal encoder
with st.expander("🔧 Lihat kelas yang dikenali encoder"):
    st.json({col: get_reverse_map(col) for col in ENCODING_MAP})

if st.button("🔮 Prediksi"):
    try:
        hasil, proba, proba_dict, kelas = predict_input(input_data, pack)
        st.success(f"✅ Hasil Prediksi: **{hasil}**")

        st.write("📊 Probabilitas:", proba_dict)

        # Visualisasi probabilitas
        st.subheader("📈 Visualisasi Probabilitas Prediksi")
        fig, ax = plt.subplots()
        ax.bar(kelas, proba)
        ax.set_ylabel("Probabilitas")
        ax.set_xlabel("Kelas")
        ax.set_title("Probabilitas Prediksi per Kelas")
        for i, v in enumerate(proba):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
        st.pyplot(fig)

        # Simpan riwayat
        file_path = save_prediction(input_data, hasil, proba_dict)
        st.info(f"📁 Hasil prediksi disimpan ke: **{file_path}**")
        with open(file_path, "rb") as f:
            st.download_button("💾 Download Semua Prediksi (CSV)", f, file_name="predictions.csv")

    except ValueError as e:
        # Pesan ramah jika masih ada label tak dikenal
        st.error(f"❌ Input tidak dikenali: {e}")
        st.info("💡 Solusi: pilih nilai sesuai daftar pada 'Lihat kelas yang dikenali encoder', atau retrain model dengan dataset yang nilainya konsisten.")
