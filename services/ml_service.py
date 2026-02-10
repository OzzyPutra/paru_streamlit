from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import pandas as pd
from datetime import datetime
from utils.encoding import ENCODING_MAP, encode_input_dict, encode_series, get_reverse_map

def normalize_columns(df):
    """
    Menormalkan nama kolom agar konsisten.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def drop_non_feature_columns(df):
    """
    Menghapus kolom ID/nomor yang tidak informatif.
    """
    df = normalize_columns(df)
    drop_cols = [c for c in ["No", "ID", "Id", "index"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    return df, drop_cols

def build_encoding_table(encoding_map=None):
    """
    Membuat tabel mapping encoding: Atribut | Kategori | Nilai Encoding.
    """
    if encoding_map is None:
        encoding_map = ENCODING_MAP
    rows = []
    for col, mapping in encoding_map.items():
        for label, value in mapping.items():
            rows.append({
                "Atribut": col,
                "Kategori": str(label),
                "Nilai Encoding": int(value),
            })
    return pd.DataFrame(rows)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data dengan stratify.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def train_model(X_train, y_train):
    """
    Melatih model CategoricalNB.
    """
    model = CategoricalNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model dan kembalikan prediksi dan metrik.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return y_pred, acc, cm, report

def save_model_pack(model, encoders, target_col, feature_cols, model_path):
    """
    Menyimpan model beserta encoder dan metadata.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pack = {
        "model": model,
        "encoders": encoders,
        "target_col": target_col,
        "feature_cols": feature_cols
    }
    joblib.dump(pack, model_path)
    return pack

def preprocess_data(df, target_col):
    """
    Melakukan encoding data kategorikal & memisahkan fitur dan target.
    """
    target_col = str(target_col).strip()
    df, _ = drop_non_feature_columns(df)

    # (opsional) normalisasi teks seperti sebelumnya kalau kamu pakai
    # df = _normalize_df(df)

    # --- PISAH FITUR & TARGET ---
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan.")
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # --- ENCODING ---
    for col in X.columns:
        if col in ENCODING_MAP:
            X[col] = encode_series(X[col], col)

    if target_col in ENCODING_MAP:
        y = encode_series(y, target_col)

    return X, y, ENCODING_MAP

def train_and_evaluate(df, target_col, model_path="models/naive_bayes_model.pkl"):
    """
    Training Naive Bayes dan evaluasi dengan train-test split.
    """
    X, y, encoders = preprocess_data(df, target_col)
    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    _, acc, cm, report = evaluate_model(model, X_test, y_test)

    save_model_pack(model, encoders, target_col, feature_cols, model_path)

    return acc, cm, report

def load_model(model_path="models/naive_bayes_model.pkl"):
    """
    Memuat model Naive Bayes beserta encoder.
    """
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        return None, "Model belum ada atau file kosong. Silakan latih ulang model."
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            return None, "Format model lama (tuple) terdeteksi. Silakan latih ulang model."
        if not isinstance(loaded, dict):
            return None, f"Format model tidak dikenal: {type(loaded)}. Silakan latih ulang model."
        required_keys = {"model", "encoders", "target_col", "feature_cols"}
        missing = required_keys - set(loaded.keys())
        if missing:
            return None, f"Model tidak lengkap, kunci hilang: {sorted(missing)}. Silakan latih ulang model."
        return loaded, None
    except Exception as e:
        return None, f"Gagal memuat model: {e}"

def predict_input(input_dict, pack):
    """
    Melakukan prediksi dari input pasien (dict).
    Mengembalikan hasil prediksi (label), probabilitas, dan dict probabilitas.
    """
    if not isinstance(pack, dict):
        raise ValueError("Model pack tidak valid. Silakan latih ulang model.")
    model = pack.get("model")
    target_col = pack.get("target_col")
    feature_cols = pack.get("feature_cols", [])
    if model is None or not feature_cols or target_col is None:
        raise ValueError("Model pack tidak lengkap. Silakan latih ulang model.")

    encoded_input = encode_input_dict(input_dict, feature_cols)
    df_input = pd.DataFrame([encoded_input]).reindex(columns=feature_cols)

    # Prediksi
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    # Decode hasil prediksi
    reverse_map = get_reverse_map(target_col)
    if reverse_map:
        hasil = reverse_map.get(int(pred), pred)
        kelas = [reverse_map.get(int(k), k) for k in model.classes_]
    else:
        hasil = pred
        kelas = model.classes_
    proba_dict = {str(kelas[i]): float(proba[i]) for i in range(len(kelas))}

    return hasil, proba, proba_dict, kelas

def save_prediction(input_dict, hasil, proba_dict, file_path="results/predictions.csv"):
    """
    Menyimpan hasil prediksi pasien ke file CSV.
    Jika file belum ada → buat baru dengan header.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Tambahkan hasil prediksi & timestamp ke input
    record = input_dict.copy()
    record["Prediksi"] = hasil
    record.update({f"Prob_{k}": v for k, v in proba_dict.items()})
    record["Waktu"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_new = pd.DataFrame([record])

    if not os.path.exists(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, mode="a", index=False, header=False)

    return file_path
