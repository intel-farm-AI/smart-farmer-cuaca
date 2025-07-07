import streamlit as st
import requests
import pandas as pd
from streamlit_js_eval import streamlit_js_eval
from tensorflow.keras.models import load_model
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
    model = load_model("plant_disease_model")
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

        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])

        labels = [
            "Sehat", "Busuk Daun", "Karat Daun", "Bercak Hitam", "Layu",
            "Jamur Putih", "Embun Tepung", "Busuk Buah", "Virus Daun Kuning",
            "Busuk Akar", "Bercak Daun", "Kutu Daun", "Kerak Daun", "Hama Ulat", "Lainnya"
        ]
        solusi = {
            "Sehat": "✅ Tanaman dalam kondisi baik.\n\n1️⃣ Lanjutkan penyiraman rutin.\n2️⃣ Pastikan tanaman mendapatkan sinar matahari yang cukup.\n3️⃣ Bersihkan gulma dan sampah di sekitar tanaman secara berkala.",
            "Busuk Daun": "⚠️ Daun mengalami pembusukan.\n\n1️⃣ Pangkas daun yang busuk dan buang jauh dari kebun.\n2️⃣ Semprotkan fungisida sesuai dosis anjuran.\n3️⃣ Kurangi penyiraman berlebih dan tingkatkan sirkulasi udara.\n4️⃣ Pastikan area sekitar tanaman tidak terlalu lembab.",
            "Karat Daun": "⚠️ Daun terkena karat.\n\n1️⃣ Semprotkan pestisida organik secara merata.\n2️⃣ Hindari penyiraman langsung ke daun.\n3️⃣ Pangkas bagian tanaman yang sangat terinfeksi.\n4️⃣ Jaga kelembaban lingkungan tetap stabil.",
            "Bercak Hitam": "⚠️ Muncul bercak hitam pada daun.\n\n1️⃣ Pangkas daun yang terinfeksi.\n2️⃣ Semprotkan fungisida tembaga sesuai petunjuk.\n3️⃣ Jangan menyiram tanaman dari atas (hindari membasahi daun).\n4️⃣ Bersihkan sisa tanaman yang jatuh di tanah.",
            "Layu": "⚠️ Tanaman terlihat layu.\n\n1️⃣ Periksa akar tanaman, pastikan tidak busuk.\n2️⃣ Kurangi intensitas penyiraman sementara.\n3️⃣ Tambahkan pupuk organik untuk memperkuat akar.\n4️⃣ Pastikan tanah tidak tergenang air.",
            "Jamur Putih": "⚠️ Terlihat jamur putih pada batang atau daun.\n\n1️⃣ Semprotkan fungisida sulfur secara berkala.\n2️⃣ Perbaiki sirkulasi udara di sekitar tanaman.\n3️⃣ Pangkas bagian yang tertutup jamur.\n4️⃣ Kurangi kelembaban di sekitar tanaman.",
            "Embun Tepung": "⚠️ Muncul lapisan putih seperti tepung pada daun.\n\n1️⃣ Semprotkan larutan baking soda atau fungisida khusus embun tepung.\n2️⃣ Pangkas daun yang terlalu banyak terinfeksi.\n3️⃣ Pastikan tanaman terkena sinar matahari cukup.\n4️⃣ Jangan menyiram daun secara langsung.",
            "Busuk Buah": "⚠️ Buah membusuk.\n\n1️⃣ Petik buah yang matang agar tidak membusuk.\n2️⃣ Buang buah yang busuk jauh dari tanaman.\n3️⃣ Semprotkan fungisida pada buah yang masih kecil.\n4️⃣ Pastikan area kebun tidak terlalu lembab.",
            "Virus Daun Kuning": "⚠️ Daun menguning akibat virus.\n\n1️⃣ Cabut tanaman yang terinfeksi berat.\n2️⃣ Bakar atau musnahkan tanaman yang terinfeksi.\n3️⃣ Jauhkan tanaman sehat dari tanaman sakit.\n4️⃣ Semprotkan pestisida alami untuk cegah penyebaran vektor.",
            "Busuk Akar": "⚠️ Akar mengalami pembusukan.\n\n1️⃣ Perbaiki drainase agar air tidak menggenang.\n2️⃣ Kurangi penyiraman berlebih.\n3️⃣ Tambahkan media tanam yang lebih porous (berongga).\n4️⃣ Gunakan fungisida khusus akar jika perlu.",
            "Bercak Daun": "⚠️ Daun muncul bercak.\n\n1️⃣ Pangkas daun yang terinfeksi ringan.\n2️⃣ Semprotkan fungisida alami sesuai dosis.\n3️⃣ Hindari penyiraman langsung ke daun.\n4️⃣ Bersihkan kebun dari daun yang gugur.",
            "Kutu Daun": "⚠️ Daun diserang kutu.\n\n1️⃣ Semprotkan air sabun atau insektisida nabati.\n2️⃣ Basuh daun dengan air bersih secara berkala.\n3️⃣ Pangkas bagian yang parah.\n4️⃣ Jaga kebersihan sekitar tanaman.",
            "Kerak Daun": "⚠️ Daun berkerak.\n\n1️⃣ Pangkas bagian yang terinfeksi kerak.\n2️⃣ Semprotkan fungisida sesuai anjuran.\n3️⃣ Bersihkan permukaan daun dengan air hangat.\n4️⃣ Periksa secara rutin untuk mencegah penyebaran.",
            "Hama Ulat": "⚠️ Daun dimakan ulat.\n\n1️⃣ Ambil ulat secara manual.\n2️⃣ Semprotkan insektisida organik jika perlu.\n3️⃣ Jaga kebersihan area sekitar tanaman.\n4️⃣ Pasang perangkap serangga sederhana.",
            "Lainnya": "⚠️ Gejala tidak dikenali.\n\n1️⃣ Periksa lebih lanjut dengan ahli pertanian.\n2️⃣ Isolasi tanaman agar tidak menular ke tanaman lain.\n3️⃣ Awasi perkembangan gejala setiap hari.\n4️⃣ Hindari penggunaan pestisida tanpa anjuran ahli."
        }

        hasil = labels[predicted_class] if predicted_class < len(labels) else f"Kelas {predicted_class}"
        rekomendasi = solusi.get(hasil, "Tidak ada rekomendasi khusus.")

        st.success(f"✅ Hasil Prediksi: **{hasil}**")
        st.info(f"💡 Rekomendasi untuk Petani:\n\n{rekomendasi}")

    except Exception as e:
        st.error(f"❌ Gagal memproses gambar atau prediksi: {e}")
