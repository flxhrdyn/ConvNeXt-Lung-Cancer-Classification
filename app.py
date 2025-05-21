import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import json
import requests
import time

# Konfigurasi halaman
st.set_page_config(page_title="🫁 Klasifikasi Kanker Paru", layout="wide")

# Memuat model dari huggingface
@st.cache_resource
def load_model():
    start_time = time.time()
    url = "https://huggingface.co/felixhrdyn/convnextv1-lung-cancer/resolve/main/convnext_lung_82.keras"
    model_path = "convnext_lung_82.keras"
    with open(model_path, "wb") as f:
        f.write(requests.get(url).content)
    model = tf.keras.models.load_model(model_path)
    elapsed_time = time.time() - start_time
    return model, elapsed_time

with st.spinner("Mengunduh dan memuat model..."):
    model_saved, elapsed = load_model()

# Local inference
# def load_model():
#     model_path = "models/convnext_lung_82.keras"
#     return tf.keras.models.load_model(model_path)
# model = load_model()

# Navigasi Sidebar
with st.sidebar:
    st.title("📋 Menu")
    page = st.radio("", ["🫁 Klasifikasi Citra", "📑 Performa Model", "🧬 Kanker Paru"], label_visibility="collapsed")

# Halaman Klasifikasi Citra
import time  # Pastikan sudah di-import di bagian atas

# Halaman Klasifikasi Citra
if "Klasifikasi" in page:
    st.title("🫁 Klasifikasi Citra Kanker Paru")
    st.markdown("📤 Unggah citra histopatologi untuk memprediksi jenis kanker paru.")

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Format yang didukung: JPG, PNG")

    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # ⏱️ Hitung waktu inference
        start_time = time.time()
        prediction = model_saved.predict(img_array)
        inference_time = time.time() - start_time

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        accuracy = float(np.max(prediction) * 100)

        # Simpan hasil ke session_state
        st.session_state["uploaded_image"] = uploaded_file
        st.session_state["predicted_class"] = predicted_class
        st.session_state["accuracy"] = accuracy
        st.session_state["inference_time"] = inference_time

    # Tampilkan hasil jika sudah ada di session_state
    if "predicted_class" in st.session_state and "uploaded_image" in st.session_state:
        img = image.load_img(st.session_state["uploaded_image"], target_size=(224, 224))
        class_labels = ['Adenokarsinoma', 'Jinak', 'Karsinoma Sel Skuamosa']
        colors = ["#f7dc6f", "#2ecc71", "#f7dc6f"]

        with st.container():
            st.subheader("📑 Hasil Prediksi")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption="Citra Histopatologi", use_container_width=True)
            with col2:
                st.markdown("### Jenis Kanker Paru:")
                st.markdown(
                    f"<h3 style='color:{colors[st.session_state['predicted_class']]}'>{class_labels[st.session_state['predicted_class']]}</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(f"### Akurasi: `{st.session_state['accuracy']:.2f}%`")
                st.success(f"✅ Prediksi Selesai! Waktu Inference: `{st.session_state['inference_time']:.4f} detik`")

    # Tampilkan waktu memuat model (satu kali)
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = True
        st.success(f"Model berhasil dimuat dalam {elapsed:.2f} detik.")
    
# Halaman Performa Model
elif "Model" in page:
    st.title("📑 Performa Model ConvNeXt")
    st.markdown("""Model ini menggunakan arsitektur **ConvNeXt**, dilatih pada 3000 citra histopatologi kanker paru-paru. 
                Dataset ini kemudian dibagi dengan rasio 80:10:10, menjadi 2400 citra data latih, 300 citra data validasi, 
                dan 300 citra data uji.""")

    with st.expander("📌 Arsitektur Model"):
        st.markdown("""
        ConvNeXt merupakan arsitektur modern berbasis Convolutional Neural Network (CNN) yang mengadopsi prinsip desain dari Vision Transformer (ViT).
        Pada penelitian ini, model `ConvNeXt_Base` dimodifikasi dengan penambahan lapisan fully connected dan metode transfer learning untuk meningkatkan 
        performa klasifikasi citra histopatologis.
        """)
        st.image("assets/model_performace_82split/model_sum_82.png", caption="Arsitektur Model ConvNeXt Final", width=700)

    with st.expander("📈 Performa Model"):

        # Perfroma Sementara!!!!!!
        st.subheader("🎯 Akurasi Model")
        st.markdown("""
                    Berikut ini adalah akurasi tertinggi yang dicapai oleh Model ConvNeXt setelah 60 epoch pelatihan dan validasi, 
                    serta pengujian dengan menggunakan test set.
                    """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("**Akurasi Latih**", "96.67%", "+96.67%")
        col2.metric("**Akurasi Validasi**", "96.67%", "+0%")
        col3.metric("**Akurasi Uji**", "93.67%", "-3%")

        st.image("assets/model_performace_82split/train_acc_loss_82.png", caption="Plot Akurasi dan Loss pada Pelatihan dan Validasi", width=700)

        st.markdown("### 📝 Classification Report")
        with open("assets/model_performace_82split/class_report_82.json", "r") as f:
            performance_data = json.load(f)
        class_report = performance_data.get("classification_report", {})

        # Konversi ke DataFrame
        report_df = pd.DataFrame(class_report).transpose()
        
        # Tampilkan sebagai tabel
        st.table(report_df)

    with st.expander("📊 Confusion Matrix"):
        st.markdown("""
        Model ini terkadang mengalami kesalahan klasifikasi antara **adenokarsinoma** (`adenocarcinoma`) dengan **karsinoma sel skuamosa** (`squamous_cell_carcinoma`) 
        karena kemiripan visualnya. Namun, jaringan paru jinak (`benign`) sebagian besar dapat diklasifikasikan dengan baik.
        """)
        st.image("assets/model_performace_82split/conf_matrix_82.png", caption="Evaluasi Confusion Matrix", width=700)
    
    with st.expander("🔮 Prediksi Test Set"):
        st.markdown("""
        Model ini diuji dengan menggunakan 300 citra histpatologi dari test set. 
        Berikut ini adalah sampel hasil prediksi citra pada tahap evaluasi model.
        """)
        st.image("assets/model_performace_82split/test_predict_82.png", caption="Hasil Prediksi Test Set", width=700)

# Halaman Kanker Paru
elif "Kanker" in page:
    st.title("🧬 Kanker Paru")
    st.markdown("""<div style='text-align: justify'>
                Kanker paru secara umum mencakup seluruh bentuk keganasan yang terjadi di paru-paru, 
                baik yang berasal dari jaringan paru itu sendiri (primer) maupun yang merupakan penyebaran dari keganasan di organ lain (metastasis). 
                Dalam konteks klinis, istilah kanker paru primer secara khusus merujuk pada tumor ganas yang berasal dari epitel bronkus, yang dikenal sebagai karsinoma bronkus (Joseph dan Rotty, 2020).
                Berikut ini adalah dua jenis kanker yang dapat diklasifikasikan pada dashboard ini:</div>""", unsafe_allow_html=True)

    # LUAD
    with st.container():
        st.subheader("1. **Adenokarsinoma Paru (LUAD)**")
        st.markdown("""<div style='text-align: justify'>
                    Adenokarsinoma paru (Lung Adenocarcinoma/LUAD) merupakan subtipe kanker paru yang paling umum, mencakup sekitar 50% dari seluruh diagnosis kanker paru, dan prevalensinya terus meningkat. 
                    Peningkatan ini diduga berkaitan dengan meningkatnya proporsi perokok perempuan, yang lebih rentan mengalami adenokarsinoma, serta perubahan dalam desain dan komposisi rokok selama 50 tahun terakhir. 
                    Perubahan tersebut menyebabkan perokok menghirup asap lebih dalam, yang diyakini meningkatkan paparan sel saluran napas perifer terhadap zat karsinogenik, tempat di mana adenokarsinoma biasanya berkembang. 
                    Selain faktor merokok, risiko adenokarsinoma paru juga meningkat akibat paparan gas radon, asap rokok pasif, polutan dalam ruangan, dan pencemaran lingkungan (Succony, Rassl, Barker, McCaughan, dan Rintoul, 2021).
                    </div>""", unsafe_allow_html=True)
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image("assets/luad.jpg", caption="Citra Histopatologi LUAD", width=300)

        # LUSC
        st.subheader("2. **Karsinoma Sel Skuamosa Paru (LUSC)**")
        st.markdown("""<div style='text-align: justify'>
                    Karsinoma sel skuamosa paru (Lung Squamous Cell Carcinoma/LUSC) merupakan tipe histologis NSCLC paling umum kedua setelah adenokarsinoma, dan menyumbang sekitar 20% dari seluruh kasus kanker paru. 
                    Pada sebagian besar pasien, karsinoma sel skuamosa paru memiliki hubungan kuat dengan kebiasaan merokok. Meskipun secara umum karsinoma sel skuamosa paru dikenal sebagai tumor yang berlokasi sentral, 
                    jenis karsinoma sel skuamosa paru yang berada di perifer paru juga telah dilaporkan, dan kini mencakup sekitar sepertiga dari seluruh kasus karsinoma sel skuamosa paru paru. 
                    Jenis ini lebih sering ditemukan pada pasien lanjut usia dan perempuan, serta dapat berasosiasi dengan penyakit paru interstisial fibrotic (Berezowska, Maillard, Keyter, dan Bisig, 2024).
                    </div>""", unsafe_allow_html=True)
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image("assets/lusc.jpg", caption="Citra Histopatologi LUSC", width=300)

        # Benign
        st.subheader("3. **Jaringan Paru Jinal (Benign Lung Tissue)**")
        st.markdown("""<div style='text-align: justify'>
                    Meskipun kanker paru seperti adenokarsinoma paru dan karsinoma sel skuamosa paru merupakan salah satu keganasan yang paling umum ditemui oleh ahli patologi, 
                    penting bagi ahli patologi untuk memahami ciri-ciri diagnostik jaringan paru jinak (benign lung tissue) agar tidak keliru mengenalinya sebagai kanker paru yang secara morfologis serupa.
                    </div>""", unsafe_allow_html=True)
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image("assets/benign.jpg", caption="Citra Histopatologi Benign Lung Tissue", width=300)

        st.markdown("""<div style='text-align: justify'>
                    <b>Sumber Referensi:</b> </br>
                    1. Joseph, J. and Rotty, L.W., 2020. Kanker paru: laporan kasus. Medical Scope Journal, 2(1). </br>
                    2. Succony, L., Rassl, D.M., Barker, A.P., McCaughan, F.M. and Rintoul, R.C., 2021. Adenocarcinoma spectrum lesions of the lung: Detection, 
                    pathology and treatment strategies. Cancer treatment reviews, 99, p.102237.</br>
                    3. Berezowska, S., Maillard, M., Keyter, M. and Bisig, B., 2024. Pulmonary squamous cell carcinoma and lymphoepithelial carcinoma–morphology, 
                    molecular characteristics and differential diagnosis. Histopathology, 84(1), pp.32-49.</br>
                    </div>""", unsafe_allow_html=True)

# Footer Opsional
st.markdown("---")
st.markdown("<center>© 2025 Klasifikasi Kanker Paru | Felix Hardyan</center>", unsafe_allow_html=True)
