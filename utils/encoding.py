ENCODING_MAP = {
    "Usia": {"Muda": 0, "Tua": 1},
    "Jenis_Kelamin": {"Wanita": 0, "Pria": 1},
    "Merokok": {"Pasif": 0, "Aktif": 1},
    "Bekerja": {"Tidak": 0, "Ya": 1},
    "Rumah_Tangga": {"Tidak": 0, "Ya": 1},
    "Aktivitas_Begadang": {"Tidak": 0, "Ya": 1},
    "Aktivitas_Olahraga": {"Jarang": 0, "Sering": 1},
    "Asuransi": {"Tidak": 0, "Ada": 1},
    "Penyakit_Bawaan": {"Tidak": 0, "Ada": 1},
    "Hasil": {"Tidak": 0, "Ya": 1},
}


def get_reverse_map(col):
    return {v: k for k, v in ENCODING_MAP.get(col, {}).items()}


def encode_value(col, value):
    mapping = ENCODING_MAP.get(col)
    if mapping is None:
        return value
    if value not in mapping:
        raise ValueError(f"Unknown category in {col}: {value}")
    return mapping[value]


def encode_series(series, col):
    mapping = ENCODING_MAP.get(col)
    if mapping is None:
        return series
    mapped = series.map(mapping)
    unknown_mask = mapped.isna() & series.notna()
    if unknown_mask.any():
        unknown_val = series[unknown_mask].iloc[0]
        raise ValueError(f"Unknown category in {col}: {unknown_val}")
    return mapped


def encode_dataframe(df, columns=None):
    df_encoded = df.copy()
    cols = columns if columns is not None else [c for c in df.columns if c in ENCODING_MAP]
    for col in cols:
        if col in df_encoded.columns:
            df_encoded[col] = encode_series(df_encoded[col], col)
    return df_encoded


def encode_input_dict(input_dict, feature_cols):
    encoded = {}
    for col in feature_cols:
        value = input_dict.get(col)
        encoded[col] = encode_value(col, value)
    return encoded


def decode_values(values, col):
    reverse_map = get_reverse_map(col)
    if not reverse_map:
        return list(values)
    decoded = []
    for value in values:
        try:
            key = int(value)
        except (TypeError, ValueError):
            decoded.append(value)
            continue
        decoded.append(reverse_map.get(key, value))
    return decoded
