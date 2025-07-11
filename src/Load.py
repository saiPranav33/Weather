import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
import tensorflow.keras.losses
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("data//data.csv")
data['datetime'] = pd.to_datetime(data['datetime'], format='%d-%m-%Y', errors='coerce')
data.dropna(subset=['datetime'], inplace=True) 
data = data.sort_values(by='datetime')

custom_objects = {'mse': tensorflow.keras.losses.MeanSquaredError()}

model = load_model('h5//bilstm_weather_model.h5', custom_objects=custom_objects)

df = pd.read_csv("data//data.csv")
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')
df.set_index('datetime', inplace=True)

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
        
        actual_weather = data.loc[data['datetime'] == given_date, features].values 
        
        print("\nActual Data for", given_date,"\n")
        for feature, value in zip(features, actual_weather[0]):
            print(f"{feature}: {value:.2f}")   
            
        mae = mean_absolute_error(actual_weather, final_prediction)
        rmse = np.sqrt(mean_squared_error(actual_weather, final_prediction))

        print(f"\nMean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
    except ValueError as e:
        print(f"Error: {e}")

def train_hybrid_model():
    X_train = np.array([scaler.transform(df[features].iloc[i - 365:i]) for i in range(365, len(df))])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
    
    lstm_predictions_scaled = model.predict(X_train)

    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

    # Prepare Data for Random Forest (Only LSTM Outputs)
    y_train = df[features].iloc[365:].values
    train_random_forest(lstm_predictions, y_train)

    print("Hybrid Model Trained Successfully.")


def evaluate_hybrid_model_from_saved_data(model_bilstm_path, rf_model_path, scaler_path, feature_names):
    # Load saved X_test and y_test
    X_test = np.load('npy/X_test.npy')
    y_test = np.load('npy/y_test.npy')

    # Load models and scaler
    model_bilstm = load_model(model_bilstm_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    rf_model = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    
    # LSTM prediction (scaled)
    lstm_predictions_scaled = model_bilstm.predict(X_test)
    
    # Inverse transform predictions and actual values
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    # Hybrid prediction via Random Forest
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
    """
    Evaluates the BiLSTM model independently on the saved test data.

    Parameters:
    - model_path: Path to the BiLSTM model (.h5)
    - scaler_path: Path to the saved StandardScaler (.pkl)
    - feature_names: List of feature names for evaluation

    Outputs:
    - MAE, RMSE, and R² scores overall and per feature
    """
    # Load test data
    X_test = np.load('npy/X_test.npy')
    y_test = np.load('npy/y_test.npy')

    # Load model and scaler
    model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load(scaler_path)

    # Predict using BiLSTM
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions and actuals to original scale
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    print("\n🔍 Evaluation Metrics for BiLSTM Model (Without Hybrid):\n")

    # Overall metrics
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
        
        
# train_hybrid_model()

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



















# from tensorflow.keras.models import load_model
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import joblib
# from sklearn.metrics import mean_absolute_error, mean_squared_error


# # Load your trained model
# model = load_model("weather_prediction_model.h5", compile=False)
# model.compile(optimizer='adam', loss='mse')

# # Load dataset (same dataset used for training)
# data = pd.read_csv("data.csv")  # Replace with actual data file

# # Ensure 'datetime' column is in proper date format
# data['datetime'] = pd.to_datetime(data['datetime'], format='%d-%m-%Y', errors='coerce')
# data.dropna(subset=['datetime'], inplace=True)  # Drop rows with invalid datetime

# # Sort data by date (if not already sorted)
# data = data.sort_values(by='datetime')

# # Define features used for training
# features = ["temperature_celsius", "wind_kph", "cloud", "humidity", "precip_mm"]

# train_data = pd.read_csv("data.csv")

# # Fit the MinMaxScaler on the training data only (to ensure same scaling is used)
# scaler = joblib.load('scale.pkl')
# scaler.fit(train_data[features])

# # Function to get past 1-year data
# def get_past_1_year_data(df, given_date):
#     given_date = pd.to_datetime(given_date, format='%d-%m-%Y', errors='coerce')
#     if pd.isna(given_date):
#         raise ValueError("Invalid date format. Use 'DD-MM-YYYY'.")

#     # Convert index to datetime if not already
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df = df.set_index(pd.to_datetime(df['datetime'], format='%d-%m-%Y', errors='coerce'))
    
#     # Define start date (1 year before the given date)
#     start_date = given_date - pd.DateOffset(years=1)

#     # Select past 1-year data for the defined features
#     past_year_data = df.loc[(df.index >= start_date) & (df.index <= given_date), features]
    
#     if past_year_data.empty:
#         raise ValueError("No data available for the past 1-year period.")

#     return past_year_data


# # Input date for prediction
# given_date = "30-05-2021"
# # given_date = "24-06-2021"
# # given_date = "04-05-2021"
# # given_date = "17-06-2021"


# # Fetch past 1-year data for model input
# try:
#     past_data = get_past_1_year_data(train_data, given_date)
    
#     # Scale past data using the same scaler from training
#     past_data_scaled = scaler.transform(past_data)  
    
#     # Reshape for LSTM input
#     X_test = np.array([past_data_scaled])  
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))  # Ensure correct shape

# except ValueError as e:
#     print(f"Error: {e}")
#     exit()

# # Predict using LSTM model
# prediction_scaled = model.predict(X_test)

# # **Inverse Transform to get Actual Values**
# predicted_weather = scaler.inverse_transform(prediction_scaled)[0]  

# from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# # Load actual weather data for the given date
# actual_weather = data.loc[data['datetime'] == given_date, features].values

# if actual_weather.size == 0:
#     raise ValueError(f"No actual data available for {given_date} to compute accuracy.")

# # Avoid division by zero for MAPE
# actual_weather[actual_weather == 0] = 1e-5  

# # Reshape predicted_weather to match actual_weather
# predicted_weather = predicted_weather.reshape(1, -1)

# # Calculate Mean Absolute Percentage Error (MAPE)
# mape = mean_absolute_percentage_error(actual_weather, predicted_weather) * 100

# # Calculate Mean Absolute Error (MAE)
# mae = mean_absolute_error(actual_weather, predicted_weather)

# # Calculate Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mean_squared_error(actual_weather, predicted_weather))


# # Print results
# print(f"\nPredicted Weather for {given_date}:")
# print(f"Temperature (°C): {predicted_weather[0][0]:.2f}")
# print(f"Wind Speed (kph): {predicted_weather[0][1]:.2f}")
# print(f"Cloud Cover (%): {predicted_weather[0][2]:.2f}")
# print(f"Humidity (%): {predicted_weather[0][3]:.2f}")
# print(f"Precipitation (mm): {predicted_weather[0][4]:.2f}")

# print("\n**Performance Metrics:**")
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")





