import pandas as pd


df = pd.read_csv("data.csv")

# Convert to datetime format
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')

# Extract the date part and time part in 24-hour format
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.strftime('%H:%M')
specific_times = ['12:00']
df_filtered = df[df['time'].isin(specific_times)]
df_filtered.to_csv('data.csv', index=False)