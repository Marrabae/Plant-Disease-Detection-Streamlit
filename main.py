import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import preprocessing

# Dictionary penyakit dan solusinya
disease_solutions = {
    'Apple___Apple_scab': "Lakukan penyemprotan fungisida yang sesuai, seperti mancozeb atau captan, pada awal musim semi.",
    'Apple___Black_rot': "Pangkas dan hancurkan bagian tanaman yang terinfeksi, serta semprot dengan fungisida berbahan dasar tembaga.",
    'Apple___Cedar_apple_rust': "Gunakan fungisida seperti mancozeb atau myclobutanil dan hapus pohon cedar yang terdekat.",
    'Apple___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Blueberry___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Cherry_(including_sour)___Powdery_mildew': "Semprotkan fungisida berbahan sulfur atau kalium bikarbonat.",
    'Cherry_(including_sour)___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Gunakan fungisida seperti azoxystrobin dan lakukan rotasi tanaman.",
    'Corn_(maize)___Common_rust_': "Semprotkan fungisida yang sesuai, seperti mancozeb atau chlorothalonil.",
    'Corn_(maize)___Northern_Leaf_Blight': "Gunakan fungisida berbahan aktif seperti mancozeb dan tanam varietas tahan penyakit.",
    'Corn_(maize)___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Grape___Black_rot': "Semprotkan fungisida berbahan aktif seperti myclobutanil dan pangkas bagian yang terinfeksi.",
    'Grape___Esca_(Black_Measles)': "Hindari stres air dan lakukan pemangkasan yang tepat. Tidak ada pengobatan kimia yang efektif.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Pangkas bagian yang terinfeksi dan gunakan fungisida berbahan dasar tembaga.",
    'Grape___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Orange___Haunglongbing_(Citrus_greening)': "Tidak ada pengobatan efektif, segera cabut dan hancurkan pohon yang terinfeksi.",
    'Peach___Bacterial_spot': "Gunakan semprotan tembaga pada awal musim semi dan pilih varietas tahan penyakit.",
    'Peach___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Pepper,_bell___Bacterial_spot': "Gunakan semprotan tembaga dan hindari irigasi overhead.",
    'Pepper,_bell___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Potato___Early_blight': "Semprotkan fungisida berbahan dasar mancozeb atau chlorothalonil.",
    'Potato___Late_blight': "Gunakan fungisida berbahan aktif seperti mancozeb dan lakukan rotasi tanaman.",
    'Potato___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Raspberry___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Soybean___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Squash___Powdery_mildew': "Semprotkan fungisida berbahan sulfur atau neem oil.",
    'Strawberry___Leaf_scorch': "Gunakan fungisida berbahan dasar tembaga dan pastikan sirkulasi udara yang baik di sekitar tanaman.",
    'Strawberry___healthy': "Tidak diperlukan tindakan, tanaman sehat.",
    'Tomato___Bacterial_spot': "Gunakan semprotan tembaga dan hindari irigasi overhead.",
    'Tomato___Early_blight': "Semprotkan fungisida berbahan dasar mancozeb atau chlorothalonil.",
    'Tomato___Late_blight': "Gunakan fungisida berbahan aktif seperti mancozeb dan lakukan rotasi tanaman.",
    'Tomato___Leaf_Mold': "Pangkas daun yang terinfeksi dan gunakan fungisida yang sesuai.",
    'Tomato___Septoria_leaf_spot': "Semprotkan fungisida berbahan dasar chlorothalonil atau mancozeb.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Gunakan insektisida berbahan aktif seperti abamectin atau neem oil.",
    'Tomato___Target_Spot': "Gunakan fungisida berbahan aktif seperti chlorothalonil dan pangkas bagian yang terinfeksi.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Cabut dan hancurkan tanaman yang terinfeksi, serta kontrol vektor serangga seperti whiteflies.",
    'Tomato___Tomato_mosaic_virus': "Cabut dan hancurkan tanaman yang terinfeksi, serta sterilisasi peralatan berkebun.",
    'Tomato___healthy': "Tidak diperlukan tindakan, tanaman sehat."
}

# Fungsi prediksi model
def model_prediction(test_image):
    model = load_model('model_train.h5')
    image = preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    model.summary()
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("Sistem Pendeteksi Penyakit Tanaman 🪴")
    st.markdown("""
    ## Selamat Datang di Sistem Pendeteksi Penyakit Tanaman!

    Sistem ini memanfaatkan teknik Machine Learning yang canggih untuk mengidentifikasi berbagai penyakit tanaman dari gambar daun tanaman. Berikut adalah fitur utama dari aplikasi kami:

    - **Deteksi Penyakit**: Unggah gambar daun tanaman, dan model kami akan memprediksi penyakit yang menyerang tanaman tersebut.
    - **Rekomendasi Solusi**: Bersamaan dengan prediksi penyakit, sistem memberikan solusi praktis untuk mengelola atau mengobati penyakit yang teridentifikasi.

    ### Cara Menggunakan
    1. Navigasikan ke halaman "Disease Recognition" pada sidebar.
    2. Unggah gambar daun tanaman yang ingin Anda analisis.
    3. Klik tombol "Prediksi" untuk mendapatkan prediksi penyakit dan rekomendasi solusi.

    Kami berharap alat ini membantu Anda dalam menjaga kesehatan tanaman secara efektif.
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
    ## Tentang Sistem Pendeteksi Penyakit Tanaman

    Sistem Pendeteksi Penyakit Tanaman adalah proyek yang bertujuan untuk membantu petani dan pekebun dalam mengidentifikasi dan mengelola penyakit tanaman menggunakan kekuatan kecerdasan buatan.

    ### Tujuan
    - **Deteksi Penyakit yang Akurat**: Menyediakan prediksi yang andal untuk berbagai penyakit tanaman berdasarkan gambar daun.
    - **Solusi Efektif**: Menawarkan solusi yang dapat diterapkan untuk mengobati atau mengelola penyakit yang terdeteksi.
    - **Teknologi yang Mudah Diakses**: Memastikan teknologi ini mudah diakses dan digunakan oleh semua orang, dari petani profesional hingga pekebun hobi.

    ### Teknologi yang Digunakan
    - **Model Machine Learning**: Dibangun menggunakan Jaringan Saraf Konvolusional (CNN) dengan TensorFlow.
    - **Dataset**: Menggunakan 'New Plant Datasets' dari Kaggle yang dibuat oleh Vipoool. Dataset ini menyediakan lebih dari 80 ribu gambar daun dari 14 jenis tanaman.
    - **Deploy**: Di-deploy menggunakan Streamlit untuk antarmuka berbasis web yang interaktif dan sederhana.
    """)

# Prediction Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Pilih gambar berjenis .jpg, .jpeg, .png", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, use_column_width=True)
    if st.button("Mulai Prediksi"):
        if test_image is not None:
            with st.spinner("Model sedang memprediksi..."):
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_Powdery_mildew',
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                result_index = model_prediction(test_image)
                predicted_class = class_name[result_index]
                st.success("Hasil dari prediksi adalah {}".format(predicted_class))
                solution = disease_solutions.get(predicted_class, "Tidak ada solusi untuk penyakit ini")
                st.success("Rekomendasi penanganan: {}".format(solution))
        else:
            st.warning("Anda belum mengunggah gambar, silahkan unggah gambar.")