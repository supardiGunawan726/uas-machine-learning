import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import streamlit as st

st.header("Prediksi Harga Saham Dengan Algoritma LSTM", divider=True)

data = pd.read_csv('meta-stock.csv')

st.subheader("Dataset")
st.table(data.head())
df = data['Close']

st.subheader("Visualisasi data yang akan digunakan")
fig = plt.figure(figsize=(12,6))
plt.plot(df)
st.pyplot(fig)
plt.clf()

st.subheader("Moving Average")
ma100 = df.rolling(100).mean()
plt.plot(df)
plt.plot(ma100, "r")
plt.title("Meta Stock Moving Average")
st.pyplot(fig)
plt.clf()

# Membuat training data dan test data

# 80% training data
train_size = int(len(df)*0.80)
# 20% test data
test_size = len(df) - train_size

train_data = pd.DataFrame(df[0:train_size])
test_data = pd.DataFrame(df[train_size:len(df)])

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.fit_transform(test_data)

# fungsi untuk membuat dataset

def create_dataset(dataset, time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append([dataset[i + time_step,0]])
    return np.array(dataX),np.array(dataY)

time_step = 100
x_train, y_train =  create_dataset(scaled_train_data, time_step)
x_test, y_test =  create_dataset(scaled_test_data, time_step)

model = load_model("./stock-lstm.h5")

test_predict = model.predict(x_test)

scaled_test_predict = scaler.inverse_transform(test_predict)
scaled_y_test = scaler.inverse_transform(y_test)

testPlot = np.empty(shape = (len(df), 1))
testPlot[:,:] = np.nan
testPlot[len(y_train)+(100)*2 + 1 : len(df) - 1,:] = scaled_y_test

testPredictPlot = np.empty(shape = (len(df), 1))
testPredictPlot[:,:] = np.nan
testPredictPlot[len(y_train)+(100)*2 + 1 : len(df) - 1,:] = scaled_test_predict

converted_dates = list(map(datetime.datetime.strptime, data.Date, len(data.Date)*['%Y-%m-%d']))

st.subheader("Hasil prediksi")
plt.plot(converted_dates, df, 'b', label='Original')
plt.plot(converted_dates, testPlot, 'g', label='Test')
plt.plot(converted_dates, testPredictPlot, 'r', label="Predicted")
plt.legend()
st.pyplot(fig)
plt.clf()