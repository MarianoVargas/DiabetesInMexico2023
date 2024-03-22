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

# Set the time horizon (365 days for 1 year)
time_horizon = 365

# Create a list to store RMST values for each time interval
rmst_values = []

# Create lists to store RMST estimates, 95% CI, and p-values
rmst_estimates = []
ci_lower_values = []
ci_upper_values = []
p_values = []

# Define the time intervals (e.g., 2 or 3 days at a time)
time_interval = 20  # Change this value to 4 for a 4-day interval

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

        # Apply multivariable filter (only DIABETES and NEUMONIA)
        interval_df = interval_df[(interval_df['DIABETES'] == 1) & (interval_df['HIPERTENSION'] == 1)]

        if not interval_df.empty:
            interval_rmst_multivariable = np.mean(np.minimum(interval_df['TIEMPO_SUP'], time_horizon))
            rmst_values.append(interval_rmst_multivariable)

            # Calculate confidence interval and p-value
            se = np.sqrt((np.var(interval_df['TIEMPO_SUP']) / len(interval_df)))
            ci_lower = interval_rmst_multivariable - 1.96 * se
            ci_upper = interval_rmst_multivariable + 1.96 * se
            p_value = 0.01  # Change this to the actual p-value

            rmst_estimates.append(interval_rmst_multivariable)
            ci_lower_values.append(ci_lower)
            ci_upper_values.append(ci_upper)
            p_values.append(p_value)
        else:
            rmst_values.append(np.nan)
    else:
        rmst_values.append(np.nan)

    current_date += pd.DateOffset(days=time_interval)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Time Interval Start': pd.date_range(start=start_date, periods=len(rmst_values), freq=f'{time_interval}D'),
    'RMST': rmst_values,
    'CI Lower': ci_lower_values,
    'CI Upper': ci_upper_values,
    'p-value': p_values
})

# Print the results table
print(results_df)

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(results_df['Time Interval Start'], results_df['RMST'], marker='o', linestyle='-', label='RMST', color='blue')

plt.title('RMST Over Time (4-Day Intervals)')
plt.xlabel('Time')
plt.ylabel('RMST (days)')
plt.grid(True)
plt.legend()
plt.show()
