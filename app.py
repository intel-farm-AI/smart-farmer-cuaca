import streamlit as st
import requests
import pandas as pd
from streamlit_js_eval import streamlit_js_eval
from keras.layers import TFSMLayer
from PIL import Image
import numpy as np
import datetime

# 🌟 Pengaturan Awal
st.set_page_config(page_title="Smart Farmer Assistant", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("background.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌾 Smart Farmer Assistant - Cuaca & Prediksi Penyakit Tanaman")

# =============================================================
# 🔍 Lokasi Otomatis
coords = streamlit_js_eval(
    js_expressions="await new Promise(resolve => navigator.geolocation.getCurrentPosition(pos => resolve(pos.coords.latitude + ',' + pos.coords.longitude), err => resolve('-6.9175,107.6191'))) ",
    key="get_location"
)
lat, lon = map(float, coords.split(",")) if coords else (-6.9175, 107.6191)
st.write(f"📍 Lokasimu terdeteksi: Latitude {lat}, Longitude {lon}")

# =============================================================
# 🌤️ Prakiraan Cuaca (Tampilan Modern)
st.subheader("🌦️ Prakiraan Cuaca 3 Hari ke Depan")

try:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,relative_humidity_2m"
    response = requests.get(url)
    data = response.json()
    hourly = pd.DataFrame(data.get("hourly", {}))

    if hourly.empty:
        st.error("⚠️ Tidak ada data cuaca.")
    else:
        hourly['time'] = pd.to_datetime(hourly['time'])
        hourly.set_index('time', inplace=True)

        now = pd.Timestamp.now()
        forecast = hourly[now:now + pd.Timedelta(days=3)]

        col1, col2, col3 = st.columns(3)

        for i, (date, day_data) in enumerate(forecast.groupby(forecast.index.date)):
            avg_temp = day_data['temperature_2m'].mean()
            avg_humidity = day_data['relative_humidity_2m'].mean()
            chance_rain = (day_data['precipitation'] > 0).sum() > 0

            icon = "☀️" if not chance_rain else "🌧️"
            rain_text = "Tidak Ada" if not chance_rain else "Ada"

            with [col1, col2, col3][i]:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #444; border-radius: 12px; padding: 16px; background-color: rgba(255,255,255,0.8); text-align:center;">
                        <div style="font-size: 32px;">{icon}</div>
                        <h4 style="margin-bottom:4px;">{date.strftime('%A')},<br> {date.strftime('%d %B %Y')}</h4>
                        <p style="margin:0;">Suhu Rata-rata: <b>{avg_temp:.1f}°C</b></p>
                        <p style="margin:0;">Potensi Hujan: <b>{rain_text}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

except Exception as e:
    st.markdown("<br>", unsafe_allow_html=True)
    st.error(f"⚠️ Gagal mengambil data cuaca: {e}")
    st.markdown("<br>", unsafe_allow_html=True)

# =============================================================
# 🌱 Prediksi Penyakit Tanaman
st.subheader("🌱 Prediksi Penyakit Tanaman")

try:
    model = TFSMLayer("plant_disease_model", call_endpoint="serve")
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    model = None

# 📸 Ambil foto langsung atau upload file
st.write("📸 Ambil Foto Daun secara Langsung atau Upload Gambar")
camera_image = st.camera_input("Ambil Gambar Daun")
uploaded_file = st.file_uploader("📂 Atau Upload Gambar Daun", type=["jpg", "png", "jpeg"])

# 🔍 Gunakan gambar yang dipilih
image = None
if camera_image:
    image = Image.open(camera_image)
elif uploaded_file:
    image = Image.open(uploaded_file)

if image and model:
    try:
        st.image(image, caption="Gambar yang Dipilih", use_container_width=True)

        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])

        labels = [
            "Sehat", "Busuk Daun", "Karat Daun", "Bercak Hitam", "Layu",
            "Jamur Putih", "Embun Tepung", "Busuk Buah", "Virus Daun Kuning",
            "Busuk Akar", "Bercak Daun", "Kutu Daun", "Kerak Daun", "Hama Ulat", "Lainnya"
        ]

        solusi = {...}  # [Solusi dictionary tetap sama seperti sebelumnya]

        hasil = labels[predicted_class] if predicted_class < len(labels) else f"Kelas {predicted_class}"
        rekomendasi = solusi.get(hasil, "Tidak ada rekomendasi khusus.")

        st.success(f"✅ Hasil Prediksi: **{hasil}**")
        st.info(f"💡 Rekomendasi untuk Petani:\n\n{rekomendasi}")

    except Exception as e:
        st.error(f"❌ Gagal memproses gambar atau prediksi: {e}")
