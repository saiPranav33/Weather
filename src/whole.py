import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow.keras.losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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



custom_objects = {'mse': tensorflow.keras.losses.MeanSquaredError()}
model = load_model('h5//bilstm_weather_model.h5', custom_objects=custom_objects)

features = ["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]

scaler = joblib.load('pkl//scale.pkl')

def get_past_1_year_data(df, given_date):
    given_date = pd.to_datetime(given_date, format='%d-%m-%Y', errors='coerce')
    
    if pd.isna(given_date):
        raise ValueError("Invalid date format. Use 'DD-MM-YYYY'.")
    
    
    start_date = given_date - pd.DateOffset(years=1)
    
    past_year_data = df.loc[(df.index >= start_date) & (df.index < given_date), features]
    
    if past_year_data.empty:
        raise ValueError("No data available for the past 1-year period.")
    
    return past_year_data


def train_random_forest(X_lstm, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=47)
    rf.fit(X_lstm, y)
    joblib.dump(rf, "pkl//random_forest_model.pkl")
    
        
        
def train_hybrid_model():
    X_train = np.array([scaler.transform(df[features].iloc[i - 365:i]) for i in range(365, len(df))])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
    lstm_predictions_scaled = model.predict(X_train)
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
    y_train = df[features].iloc[365:].values
    train_random_forest(lstm_predictions, y_train)
    print("Hybrid Model Trained Successfully.")
    
def predict_weather(given_date):
    try:
        past_data = get_past_1_year_data(df, given_date)
        past_data_scaled = scaler.transform(past_data)
        required_timesteps = 365
        if past_data_scaled.shape[0] < required_timesteps:
            padding = np.zeros((required_timesteps - past_data_scaled.shape[0], len(features)))
            past_data_scaled = np.vstack((padding, past_data_scaled))
        X_test_lstm = np.array([past_data_scaled])
        X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], len(features)))
        lstm_prediction_scaled = model.predict(X_test_lstm)
        lstm_output = scaler.inverse_transform(lstm_prediction_scaled)
        print("\nLSTM Model Prediction is \n")
        for feature, value in zip(features, lstm_output[0]):
            print(f"{feature}: {value:.2f}")   
        rf_model = joblib.load("pkl//random_forest_model.pkl")
        final_prediction = rf_model.predict(lstm_output)
        print("\nFinal Hybrid Model Prediction for", given_date,"\n")
        for feature, value in zip(features, final_prediction[0]):
            print(f"{feature}: {value:.2f}")
        actual_weather = df.loc[df['datetime'] == given_date, features].values 
        print("\nActual Data for", given_date,"\n")
        for feature, value in zip(features, actual_weather[0]):
            print(f"{feature}: {value:.2f}")     
        mae = mean_absolute_error(actual_weather, final_prediction)
        rmse = np.sqrt(mean_squared_error(actual_weather, final_prediction))
        print(f"\nMean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    except ValueError as e:
        print(f"Error: {e}")
        
def evaluate_hybrid_model_from_saved_data(model_bilstm_path, rf_model_path, scaler_path, feature_names):
    X_test = np.load('npy/X_test.npy')
    y_test = np.load('npy/y_test.npy')
    model_bilstm = load_model(model_bilstm_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    rf_model = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    lstm_predictions_scaled = model_bilstm.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test)
    hybrid_predictions = rf_model.predict(lstm_predictions)
    print("\n🔎 Evaluation Metrics for Hybrid LSTM + Random Forest Model:\n")
    overall_mae = mean_absolute_error(y_test_actual, hybrid_predictions)
    overall_rmse = np.sqrt(mean_squared_error(y_test_actual, hybrid_predictions))
    overall_r2 = r2_score(y_test_actual, hybrid_predictions)
    print(f"✅ Overall MAE  : {overall_mae:.4f}")
    print(f"✅ Overall RMSE : {overall_rmse:.4f}")
    print(f"✅ Overall R²   : {overall_r2:.4f}")
    print("\n📊 Per Feature Metrics:")
    for i, feature in enumerate(feature_names):
        y_true = y_test_actual[:, i]
        y_pred = hybrid_predictions[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n🔸 {feature}")
        print(f"   • MAE  : {mae:.4f}")
        print(f"   • RMSE : {rmse:.4f}")
        print(f"   • R²   : {r2:.4f}")
        
def evaluate_bilstm_model(model_path, scaler_path, feature_names):
    X_test = np.load('npy/X_test.npy')
    y_test = np.load('npy/y_test.npy')
    model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load(scaler_path)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    print("\n🔍 Evaluation Metrics for BiLSTM Model (Without Hybrid):\n")
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)
    print(f"✅ Overall MAE  : {overall_mae:.4f}")
    print(f"✅ Overall RMSE : {overall_rmse:.4f}")
    print(f"✅ Overall R²   : {overall_r2:.4f}")
    print("\n📊 Per Feature Metrics:")
    for i, feature in enumerate(feature_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"\n🔸 {feature}")
        print(f"   • MAE  : {mae:.4f}")
        print(f"   • RMSE : {rmse:.4f}")
        print(f"   • R²   : {r2:.4f}")
        
        
predict_weather("19-01-2025")
evaluate_bilstm_model(
    model_path='h5//bilstm_weather_model.h5',
    scaler_path='pkl//scale.pkl',
    feature_names=["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]
)
evaluate_hybrid_model_from_saved_data(
    model_bilstm_path='h5//bilstm_weather_model.h5',
    rf_model_path='pkl//random_forest_model.pkl',
    scaler_path='pkl//scale.pkl',
    feature_names=["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]
)