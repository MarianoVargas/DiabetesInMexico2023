import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    return pd.read_csv(file_path, encoding="ISO-8859-1")

# Load the dataset
file_path = "COVID19MEXICO.csv"
df = load_data(file_path)
if df is None:
    raise ValueError("Data loading failed. Check the file path or data loading process.")

# Filter rows where DIABETES is equal to 1 and CLASIFICACION_FINAL is in [1, 2, 3]
df_filtered_diabetes = df[(df['DIABETES'] == 1) & (df['CLASIFICACION_FINAL'].isin([1, 2, 3]))]

# Filter rows where DIABETES is equal to 2 (no diabetes) and CLASIFICACION_FINAL is in [1, 2, 3]
df_filtered_no_diabetes = df[(df['DIABETES'] == 2) & (df['CLASIFICACION_FINAL'].isin([1, 2, 3]))]

# Filter out rows where FECHA_DEF is not equal to '9999-99-99'
df_filtered_diabetes = df_filtered_diabetes[df_filtered_diabetes['FECHA_DEF'] != '9999-99-99']
df_filtered_no_diabetes = df_filtered_no_diabetes[df_filtered_no_diabetes['FECHA_DEF'] != '9999-99-99']

# Convert date columns to datetime format
df_filtered_diabetes['FECHA_DEF'] = pd.to_datetime(df_filtered_diabetes['FECHA_DEF'], format='%d/%m/%Y', errors='coerce')
df_filtered_diabetes['FECHA_INGRESO'] = pd.to_datetime(df_filtered_diabetes['FECHA_INGRESO'], format='%d/%m/%Y', errors='coerce')

df_filtered_no_diabetes['FECHA_DEF'] = pd.to_datetime(df_filtered_no_diabetes['FECHA_DEF'], format='%d/%m/%Y', errors='coerce')
df_filtered_no_diabetes['FECHA_INGRESO'] = pd.to_datetime(df_filtered_no_diabetes['FECHA_INGRESO'], format='%d/%m/%Y', errors='coerce')

# Calculate survival time in days for people with diabetes and without diabetes
df_filtered_diabetes['TIEMPO_SUP'] = (df_filtered_diabetes['FECHA_DEF'] - df_filtered_diabetes['FECHA_INGRESO']).dt.days
df_filtered_no_diabetes['TIEMPO_SUP'] = (df_filtered_no_diabetes['FECHA_DEF'] - df_filtered_no_diabetes['FECHA_INGRESO']).dt.days

# Set the time horizon (365 days for 1 year)
time_horizon = 365

# Calculate RMST for people with diabetes
rmst_diabetes = np.mean(np.minimum(df_filtered_diabetes['TIEMPO_SUP'], time_horizon))

# Calculate RMST for people without diabetes
rmst_no_diabetes = np.mean(np.minimum(df_filtered_no_diabetes['TIEMPO_SUP'], time_horizon))

# Calculate the difference in RMST and its confidence interval
rmst_difference = rmst_diabetes - rmst_no_diabetes
se = np.sqrt(
    (np.var(df_filtered_diabetes['TIEMPO_SUP']) / len(df_filtered_diabetes)) +
    (np.var(df_filtered_no_diabetes['TIEMPO_SUP']) / len(df_filtered_no_diabetes))
)
ci_lower = rmst_difference - 1.96 * se
ci_upper = rmst_difference + 1.96 * se
p_value = 0.01  # Change this to the actual p-value

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'RMST estimate': [rmst_difference],
    '95% CI': [(ci_lower, ci_upper)],
    'p Value': [f'p < {p_value}']
})

# Print the results table
print(results_df)
