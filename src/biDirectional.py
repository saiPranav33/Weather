from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
import joblib

df = pd.read_csv("data//data.csv")

df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')
df.set_index('datetime', inplace=True)

numerical_cols = ["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
joblib.dump(scaler, 'pkl//scale.pkl')

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  
    return np.array(X), np.array(y)

seq_length = 365

X, y = create_sequences(df.values, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

np.save('npy/X_test.npy', X_test)
np.save('npy/y_test.npy', y_test)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model = Sequential([
    Bidirectional(LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2]))),
    Dropout(0.2),
    Bidirectional(LSTM(50, return_sequences=False)),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(X_train.shape[2])  
])



model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save('h5//bilstm_weather_model_1.h5')

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('images/biLstm_training.png')
plt.show()
