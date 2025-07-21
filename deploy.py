import streamlit as st
import joblib
import pandas as pd

# Load model dan encoding maps
model = joblib.load('model_terbaik_baru.pkl')
mean_encoding_maps = joblib.load('mean_encoding_maps.pkl')

# Ambil encoding map
customer_type_map = mean_encoding_maps['customer_type']
channel_map = mean_encoding_maps['distribution_channel']
pax_segment_map = mean_encoding_maps['pax_segment']

# Threshold (jika pakai)
threshold = 0.35

# Fungsi segmentasi pax
def classify_pax_segment(adults, children, babies):
    total = adults + children + babies
    if adults == 0 and (children > 0 or babies > 0):
        return 'Unaccompanied Children'
    elif total <= 2:
        return 'Solo/Couple'
    elif total <= 5:
        return 'Family'
    elif total <= 8:
        return 'Big Family'
    else:
        return 'Group'

# Streamlit UI
st.title("Prediksi Upgrade Kamar")

# Input user
lead_time = st.number_input("Jarak Reservasi (hari)", min_value=0)
adults = st.number_input("Jumlah Dewasa", min_value=0, step=1)
children = st.number_input("Jumlah Anak-anak", min_value=0, step=1)
babies = st.number_input("Jumlah Bayi", min_value=0, step=1)
stays_in_week_nights = st.number_input("Lama Menginap Weekday", min_value=0)
stays_in_weekend_nights = st.number_input("Lama Menginap Weekend", min_value=0)
distribution_channel = st.selectbox("Distribusi Channel", ['Corporate', 'Direct', 'GDS', 'TA/TO'])
customer_type = st.selectbox("Tipe Customer", ['Contract', 'Group', 'Transient', 'Transient-Party'])

if st.button("Prediksi Upgrade"):
    # Encoding
    customer_type_encoded = customer_type_map.get(customer_type, 0.0)
    channel_encoded = channel_map.get(distribution_channel, 0.0)
    pax_segment = classify_pax_segment(adults, children, babies)
    pax_segment_encoded = pax_segment_map.get(pax_segment, 0.0)

    # Buat DataFrame dengan urutan kolom yang sesuai
    input_df = pd.DataFrame([{
        'lead_time': lead_time,
        'customer_type_mean_encoded': customer_type_encoded,
        'stays_in_week_nights': stays_in_week_nights,
        'distribution_channel_mean_encoded': channel_encoded,
        'stays_in_weekend_nights': stays_in_weekend_nights,
        'pax_segment_mean_encoded': pax_segment_encoded
    }])

    # Prediksi
    prob = model.predict_proba(input_df)[0][1]
    result = "Upgrade" if prob >= threshold else "No Upgrade"
    st.subheader(f"Hasil Prediksi: {result}")
    st.caption(f"Probabilitas Upgrade: {prob:.2f}")