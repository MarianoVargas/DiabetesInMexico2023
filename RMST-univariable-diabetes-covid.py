import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path, encoding="ISO-8859-1")


# Load the dataset
file_path = "COVID19MEXICO.csv"
df = load_data(file_path)
if df is None:
    raise ValueError("Data loading failed. Check the file path or data loading process.")

# Filter rows where DIABETES is equal to 1 and CLASIFICACION_FINAL is in [1, 2, 3]
df_filtered = df[(df['DIABETES'] == 1) & (df['CLASIFICACION_FINAL'].isin([1, 2, 3]))]

# Filter out rows where FECHA_DEF is not equal to '9999-99-99'
df_filtered = df_filtered[df_filtered['FECHA_DEF'] != '9999-99-99']

# Convert date columns to datetime format
df_filtered['FECHA_DEF'] = pd.to_datetime(df_filtered['FECHA_DEF'], format='%d/%m/%Y', errors='coerce')
df_filtered['FECHA_INGRESO'] = pd.to_datetime(df_filtered['FECHA_INGRESO'], format='%d/%m/%Y', errors='coerce')

# Calculate survival time in days
df_filtered['TIEMPO_SUP'] = (df_filtered['FECHA_DEF'] - df_filtered['FECHA_INGRESO']).dt.days

# Set the time horizon (365 days for 1 year)
time_horizon = 365

# Create a list to store RMST values for each time interval
rmst_values = []

# Define the time intervals (e.g., 2 or 3 days at a time)
time_interval = 10  # Change this value as needed

# Iterate through the dataset in time intervals
start_date = df_filtered['FECHA_INGRESO'].min()
end_date = df_filtered['FECHA_INGRESO'].max()
current_date = start_date

while current_date <= end_date:
    interval_df = df_filtered[
        (df_filtered['FECHA_INGRESO'] >= current_date) &
        (df_filtered['FECHA_INGRESO'] < current_date + pd.DateOffset(days=time_interval))
        ]

    if not interval_df.empty:
        interval_rmst = np.mean(np.minimum(interval_df['TIEMPO_SUP'], time_horizon))
        rmst_values.append(interval_rmst)

    current_date += pd.DateOffset(days=time_interval)

# Create a time series plot
dates = pd.date_range(start=start_date, periods=len(rmst_values), freq=f'{time_interval}D')
plt.figure(figsize=(10, 6))
plt.plot(dates, rmst_values, marker='o', linestyle='-')
plt.title('RMST Over Time')
plt.xlabel('Time')
plt.ylabel('RMST (days)')
plt.grid(True)
plt.show()
