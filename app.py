import streamlit as st
import pandas as pd
from joblib import load  # Ganti pickle dengan joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Kredit",
    page_icon="💳",
    layout="centered"
)

# Load model pakai joblib
model = load("loan_default_model.joblib")  # pastikan nama file sesuai hasil training

# Judul
st.title("💳 Prediksi Risiko Gagal Bayar Kredit")
st.markdown(
    """
    Aplikasi ini membantu **memprediksi risiko gagal bayar**  
    berdasarkan data yang diisi oleh calon peminjam.
    """
)

st.divider()

# Sidebar
st.sidebar.header("📌 Petunjuk Pengisian")
st.sidebar.write(
    """
    - Isi data sesuai kondisi sebenarnya  
    - Klik **Prediksi Risiko**  
    - Hasil akan muncul di bawah
    """
)

# Layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur (tahun)", 18, 100, 30)
    income = st.number_input("Pendapatan Bulanan (Rp)", 0, value=5000000)
    credit_score = st.number_input("Skor Kredit", 300, 850, 650)
    employment_type = st.selectbox(
        "Jenis Pekerjaan",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    )

with col2:
    loan_amount = st.number_input("Jumlah Pinjaman (Rp)", 0, value=10000000)
    months_employed = st.number_input("Lama Bekerja (bulan)", 0, value=24)
    has_dependents = st.selectbox("Memiliki Tanggungan?", ["Yes", "No"])
    has_cosigner = st.selectbox("Memiliki Penjamin?", ["Yes", "No"])

st.divider()

# Tombol prediksi
if st.button("🔍 Prediksi Risiko", use_container_width=True):

    # --- Pastikan numeric ---
    age = int(age)
    income = float(income)
    loan_amount = float(loan_amount)
    credit_score = float(credit_score)
    months_employed = int(months_employed)

    # --- Ubah binary ke 0/1 ---
    has_dependents = 1 if has_dependents == "Yes" else 0
    has_cosigner = 1 if has_cosigner == "Yes" else 0

    # --- Buat dataframe input ---
    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "EmploymentType": employment_type,  # tetap string
        "HasDependents": has_dependents,
        "HasCoSigner": has_cosigner
    }])

    # --- Prediksi ---
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
    except Exception as e:
        st.error(f"⚠️ Terjadi error saat prediksi: {e}")
        st.stop()

    st.subheader("📊 Hasil Prediksi")

    # Indikator warna risiko
    if probability <= 30:
        risk_level = "🟢 Rendah"
        risk_desc = "Risiko gagal bayar tergolong rendah."
    elif probability <= 60:
        risk_level = "🟠 Sedang"
        risk_desc = "Risiko gagal bayar tergolong sedang."
    else:
        risk_level = "🔴 Tinggi"
        risk_desc = "Risiko gagal bayar tergolong tinggi."

    st.metric(
        label="Persentase Risiko Gagal Bayar",
        value=f"{probability:.2f}%"
    )
    st.info(f"**Level Risiko:** {risk_level}\n\n{risk_desc}")

    if prediction == 1:
        st.error(
            f"""
            ⚠️ **BERISIKO GAGAL BAYAR**

            Model memprediksi risiko gagal bayar sebesar **{probability:.2f}%**.

            Hal ini bisa dipengaruhi oleh:
            - Skor kredit yang rendah
            - Jumlah pinjaman relatif besar
            - Pendapatan atau lama bekerja yang kurang stabil
            """
        )
    else:
        st.success(
            f"""
            ✅ **TIDAK BERISIKO GAGAL BAYAR**

            Model memprediksi risiko gagal bayar sebesar **{probability:.2f}%**.

            Hal ini menunjukkan:
            - Kemampuan bayar yang cukup baik
            - Riwayat kredit yang relatif aman
            - Kondisi keuangan yang lebih stabil
            """
        )

st.caption("© Aplikasi Prediksi Kredit | Machine Learning + Streamlit")
