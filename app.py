import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

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

# load LSTM model yang disimpan
model = load_model('lstm_hasil_tangkap_ikan.keras')

# load dataset gabungan
df = pd.read_excel('data_produksi_ikan_merged.xlsx')

# preprocess data
df.set_index('Tanggal', inplace=True)
df = df.drop(columns=['Tahun', 'Bulan'])

# normalisasi data
scaler = MinMaxScaler()
df['Produksi'] = scaler.fit_transform(df[['Produksi']])

# fungsi rekursif untuk memprediksi hasil
def recursive_predict(model, input_data, future_steps, time_step, scaler):
    predictions = []
    current_input = input_data

    for _ in range(future_steps):
        prediction = model.predict(current_input)
        predictions.append(prediction[0, 0])
        # update data berdasarkan hasil prediksi
        new_input = np.append(current_input[:, 1:, :], [[[prediction[0, 0]]]], axis=1)
        current_input = new_input

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

# kode utama untuk streamlit app
def main():
    st.title("Prediksi Hasil Tangkapan Ikan")
    # url gambar cover
    image_url = "https://images.pexels.com/photos/2163234/pexels-photo-2163234.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    st.image(image_url, use_column_width=True)
    
    # input field untuk parameter model
    st.header("Masukkan beberapa input di sini")
    jenis_ikan = st.selectbox("Pilih jenis ikan yang akan diprediksi", 
                              ("Cucut", "Kembung", "Kerapu", "Layur", "Lemuru", "Selar", "Tembang", "Tenggiri", "Teri", "Tongkol"))
    
    # input select untuk bulan dan tahun
    months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    month = st.selectbox("Pilih bulan yang akan diprediksi", months)
    
    years = [str(year) for year in range(2013, 2034)]
    year = st.selectbox("Pilih tahun yang akan diprediksi", years)
    
    # mengubah menjadi format date
    selected_date = datetime(int(year), months.index(month) + 1, 1)
    
    # kode untuk menghitung data ketika lebih dari data yang ada di dataset
    last_date_in_data = df.index[-1]
    future_steps = (selected_date.year - last_date_in_data.year) * 12 + (selected_date.month - last_date_in_data.month)
    
    # preprocess input untuk prediksi
    def preprocess_input(df, time_step=12):
        recent_data = df.tail(time_step).values
        recent_data = recent_data.reshape(1, time_step, 1)
        return recent_data
    
    # fungsi untuk mendapatkan potongan data berdasarkan input tanggal
    def get_data_up_to_date(df, date):
        if date in df.index:
            return df.loc[:date].tail(12).values
        else:
            return df.tail(12).values
    
    # bagian untuk prediksi
    st.header("Prediksi")
    if st.button("Prediksi Sekarang"):
        if selected_date <= last_date_in_data:
            # menggunakan data terbaru sesuai tanggal yang dipilih
            recent_data = get_data_up_to_date(df, selected_date)
            recent_data = recent_data.reshape(1, 12, 1)
            prediction = model.predict(recent_data)
            predicted_value = scaler.inverse_transform(prediction)[0, 0]
            result = fish_percentage[jenis_ikan] / 100 * predicted_value
        else:
            # menggunakan rekursif ketika data yang dipilih lebih dari data yang ada di dataset
            input_data = preprocess_input(df)
            prediction = recursive_predict(model, input_data, future_steps, 12, scaler)
            predicted_value = prediction[-1, 0]
            result = fish_percentage[jenis_ikan] / 100 * predicted_value

        st.header("Hasil")
        st.write(f"Prediksi hasil tangkapan ikan untuk {jenis_ikan} pada bulan {month} {year} adalah {result:.0f} kg")

if __name__ == '__main__':
    main()