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

# Filter rows where CLASIFICACION_FINAL is in [1, 2, 3]
df_filtered = df[df['CLASIFICACION_FINAL'].isin([1, 2, 3])]

# Filter out rows where FECHA_DEF is not equal to '9999-99-99'
df_filtered = df_filtered[df_filtered['FECHA_DEF'] != '9999-99-99']

# Convert date columns to datetime format
df_filtered['FECHA_DEF'] = pd.to_datetime(df_filtered['FECHA_DEF'], format='%d/%m/%Y', errors='coerce')
df_filtered['FECHA_INGRESO'] = pd.to_datetime(df_filtered['FECHA_INGRESO'], format='%d/%m/%Y', errors='coerce')

# Calculate survival time in days
df_filtered['TIEMPO_SUP'] = (df_filtered['FECHA_DEF'] - df_filtered['FECHA_INGRESO']).dt.days

# Set the time horizon (4 days)
time_horizon = 4

# Create a list to store RMST values for DIABETES and NON-DIABETES
rmst_values_diabetes = []
rmst_values_no_diabetes = []

# Define the time intervals (e.g., 2 or 3 days at a time)
time_interval = 5  # Change this value as needed

# Iterate through the dataset in time intervals
start_date = df_filtered['FECHA_INGRESO'].min()
end_date = df_filtered['FECHA_INGRESO'].max()
current_date = start_date

while current_date <= end_date:
    interval_df_diabetes = df_filtered[
        (df_filtered['FECHA_INGRESO'] >= current_date) &
        (df_filtered['FECHA_INGRESO'] < current_date + pd.DateOffset(days=time_interval)) &
        (df_filtered['DIABETES'] == 1)
    ]

    interval_df_no_diabetes = df_filtered[
        (df_filtered['FECHA_INGRESO'] >= current_date) &
        (df_filtered['FECHA_INGRESO'] < current_date + pd.DateOffset(days=time_interval)) &
        (df_filtered['DIABETES'] == 2)
    ]

    if not interval_df_diabetes.empty:
        interval_rmst_diabetes = np.mean(np.minimum(interval_df_diabetes['TIEMPO_SUP'], time_horizon))
        rmst_values_diabetes.append(interval_rmst_diabetes)
    else:
        rmst_values_diabetes.append(np.nan)

    if not interval_df_no_diabetes.empty:
        interval_rmst_no_diabetes = np.mean(np.minimum(interval_df_no_diabetes['TIEMPO_SUP'], time_horizon))
        rmst_values_no_diabetes.append(interval_rmst_no_diabetes)
    else:
        rmst_values_no_diabetes.append(np.nan)

    current_date += pd.DateOffset(days=time_interval)

# Create separate time series plots for DIABETES and NON-DIABETES
dates = pd.date_range(start=start_date, periods=len(rmst_values_diabetes), freq=f'{time_interval}D')
plt.figure(figsize=(12, 6))

# Plot for DIABETES
plt.subplot(1, 2, 1)
plt.plot(dates, rmst_values_diabetes, marker='o', linestyle='-', label='DIABETES', color='blue')
plt.title('RMST Over Time (DIABETES)')
plt.xlabel('Time')
plt.ylabel('RMST (days)')
plt.grid(True)
plt.legend()

# Plot for NON-DIABETES
plt.subplot(1, 2, 2)
plt.plot(dates, rmst_values_no_diabetes, marker='o', linestyle='-', label='NON-DIABETES', color='orange')
plt.title('RMST Over Time (NON-DIABETES)')
plt.xlabel('Time')
plt.ylabel('RMST (days)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Create DataFrames for DIABETES and NON-DIABETES RMST values
result_df_diabetes = pd.DataFrame({'Date': dates, 'RMST_DIABETES': rmst_values_diabetes})
result_df_no_diabetes = pd.DataFrame({'Date': dates, 'RMST_NON_DIABETES': rmst_values_no_diabetes})

# Display the DataFrames
print("DIABETES RMST Results:")
print(result_df_diabetes)

print("\nNON-DIABETES RMST Results:")
print(result_df_no_diabetes)