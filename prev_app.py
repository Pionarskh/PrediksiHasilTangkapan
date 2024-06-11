# import beberapa modul yang diperlukan
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# load LSTM model yang sudah disimpan di tahap sebelumnya
model = load_model('lstm_hasil_tangkap_ikan.keras')
image_url = "https://images.pexels.com/photos/2163234/pexels-photo-2163234.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

# persentasi ikan yang didapat dari training model
fish_percentage = {
    "Cucut": 0.293325,
    "Kembung": 22.804557,
    "Kerapu": 0.372108,
    "Layur": 0.585560,
    "Lemuru": 0.869104,
    "Selar": 5.189827,
    "Tembang": 24.830376,
    "Tenggiri": 3.208910,
    "Teri": 38.177588,
    "Tongkol": 3.668646
}

# load dataset yang digunakan, yaitu hasil gabungan dari semua ikan
df = pd.read_excel('data_produksi_ikan_merged.xlsx')

# preprocessing dataset
df.set_index('Tanggal', inplace=True)
df = df.drop(columns=['Tahun', 'Bulan'])

# normalisasi data menggunakan minmax scaller
scaler = MinMaxScaler()
df['Produksi'] = scaler.fit_transform(df)

# fungsi untuk membuat dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# fungsi utama
def main():
    st.title("Prediksi Hasil Tangkapan Ikan")
    st.image(image_url, use_column_width=True)
    
    st.header("Masukkan beberapa input di sini")
    
    # text input untuk memilih jenis ikan
    jenis_ikan = st.selectbox("Pilih jenis ikan yang akan diprediksi", 
                              ("Cucut", "Kembung", "Kerapu", "Layur", "Lemuru", "Selar", "Tembang", "Tenggiri", "Teri", "Tongkol"))
    
    # text input untuk memilih tanggal
    tanggal = st.date_input("Pilih tanggal yang akan diprediksi", value=datetime.today())
    
    # lakukan preprocessing untuk input tanggal
    def preprocess_input(date, df, time_step=12):        
        # ekstrak 'time_step' terakhir pada data sesuai dengan tanggal yang dipilih
        recent_data = df.loc[:date].tail(time_step).values
        
        # jika data yang dibutuuhkan untuk prediksi tidak cukup maka tampilkan error
        if len(recent_data) < time_step:
            st.error("Data tidak cukup untuk prediksi. Silakan pilih tanggal yang berbeda.")
            return None
        
        # reshape dataset agar sesuai dengan LSTM model
        recent_data = recent_data.reshape(1, time_step, 1)
        return recent_data
    
    # bagian prediksi
    if st.button("Prediksi Sekarang"):
        input_data = preprocess_input(tanggal, df)
        if input_data is not None:
            prediction = model.predict(input_data)
            
            # kembalikan ke skala semula untuk melihat data dalam skala asli
            predicted_value = scaler.inverse_transform(prediction)
            # kalikan dengan persentase berdasarkan jenis ikan
            result = fish_percentage[jenis_ikan] / 100 * predicted_value[0][0]
        
            # tampilan hasil prediksi
            st.divider()
            st.header("Hasil")
            st.write(f"Prediksi hasil tangkapan ikan {jenis_ikan} pada tanggal {tanggal} adalah {result:.0f}")

if __name__ == '__main__':
    main()