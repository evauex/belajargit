import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
from pydub import AudioSegment
import io
import pandas as pd

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_resource()
def load_model():
    """Memuat model Keras .h5 yang sudah di-cache untuk performa."""
    try:
        model = tf.keras.models.load_model("Trained_model.h5")
        return model
    except Exception as e:
        st.error(f"Error: Tidak dapat memuat model 'Trained_model.h5'. Pastikan file ada di folder yang sama. Detail: {e}")
        return None

def load_and_preprocess_data(uploaded_file, target_shape=(150, 150)):
    """
    VERSI BARU: Memastikan sample rate (SR) diatur ke 22050 Hz agar
    konsisten dengan konfigurasi saat training model.
    """
    data = []
    try:
        # TENTUKAN SAMPLE RATE TARGET (HARUS SAMA DENGAN SAAT TRAINING)
        TARGET_SR = None
        
        # 1. Gunakan pydub untuk membuka format apa pun dan mengubahnya menjadi WAV di memori
        audio_segment = AudioSegment.from_file(uploaded_file)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # 2. Gunakan librosa.load untuk membaca dari buffer. Ini memastikan
        #    sample rate diubah ke TARGET_SR dan audio dinormalisasi dengan benar.
        audio_data, sample_rate = librosa.load(wav_io, sr=TARGET_SR)

        # 3. Sisa logika chunking Anda tetap sama dan sudah benar
        chunk_duration = 4
        overlap_duration = 2
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]

            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')

            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)

        if not data:
            st.error("Gagal memproses file audio. Mungkin file terlalu pendek.")
            return None
            
        return np.array(data)

    except Exception as e:
        st.error(f"Error saat memproses file audio: {e}")
        return None

def model_prediction(X_test, model):
    """
    MODIFIKASI: Melakukan prediksi pada semua chunk dan mengembalikan
    VEKTOR RATA-RATA PROBABILITAS untuk semua genre.
    """
    if model is None:
        return None

    # y_pred akan memiliki shape (jumlah_chunk, jumlah_genre)
    y_pred = model.predict(X_test)

    # Hitung rata-rata probabilitas di semua chunk.
    # axis=0 berarti kita merata-ratakan secara vertikal (per kolom/genre)
    mean_probabilities = np.mean(y_pred, axis=0)
    
    return mean_probabilities

# --- TAMPILAN UTAMA APLIKASI ---

# Atur konfigurasi halaman
st.set_page_config(page_title="Music Genre Classifier", layout="wide")

# Judul Utama
st.title("🎵 Music Genre Classification")
st.write("Upload an audio file (MP3, WAV, OGG) and the AI will predict its genre.")

# Muat model di awal
model = load_model()

# Widget untuk upload file
uploaded_file = st.file_uploader("Upload your audio file here:", type=["mp3", "wav", "ogg"])

# Proses otomatis setelah file diunggah
if uploaded_file is not None and model is not None:
    # Tampilkan audio player
    st.audio(uploaded_file, format='audio/wav')

    # Proses dan prediksi secara otomatis
    with st.spinner("Analyzing the music... Please wait."):
        
        processed_chunks = load_and_preprocess_data(uploaded_file)
        
        if processed_chunks is not None:
            # Dapatkan vektor probabilitas rata-rata dari model
            prediction_probabilities = model_prediction(processed_chunks, model)
            
            if prediction_probabilities is not None:
                st.balloons()
                
                # Daftar genre harus sesuai dengan urutan saat training
                genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                
                # --- TAMPILAN HASIL BARU ---
                
                col1, col2 = st.columns([0.6, 0.4]) # Buat 2 kolom

                with col1:
                    # Tentukan Genre Utama (indeks dengan probabilitas tertinggi)
                    top_genre_index = np.argmax(prediction_probabilities)
                    predicted_genre = genre_labels[top_genre_index].capitalize()

                    # Tampilkan hasil utama
                    st.subheader("Predicted Genre")
                    result_style = f"<div style='font-size: 48px; font-weight: bold; color: #FF4B4B; '>{predicted_genre}</div>"
                    st.markdown(result_style, unsafe_allow_html=True)
                
                with col2:
                    # Tampilkan Rincian Probabilitas
                    st.subheader("Confidence Breakdown")
                    
                    # Buat DataFrame untuk tampilan yang rapi
                    df_probs = pd.DataFrame({
                        'Genre': [label.capitalize() for label in genre_labels],
                        'Probability': prediction_probabilities
                    })
                    df_probs = df_probs.sort_values(by='Probability', ascending=False).reset_index(drop=True)
                    
                    # Tampilkan tabel
                    st.dataframe(df_probs,
                                 column_config={
                                     "Genre": "Genre",
                                     "Probability": st.column_config.ProgressColumn(
                                         "Confidence",
                                         format="%.2f%%",
                                         min_value=0,
                                         max_value=1,
                                     ),
                                 },
                                 hide_index=True,
                                 use_container_width=True)
                                 
                st.write("---")
                st.subheader("Probability Visualization")
                # Tampilkan juga bar chart untuk visualisasi cepat
                # Kita perlu dictionary: {'genre': probability}
                chart_data = {label.capitalize(): prob for label, prob in zip(genre_labels, prediction_probabilities)}
                st.bar_chart(chart_data)