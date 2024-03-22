import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

# Load the dataset from the CSV file
df = pd.read_csv('COVID19MEXICO.csv')

# Filter rows where the DIABETES column equals 1 (or the value indicating diabetes presence)
df_diabetes = df[df['DIABETES'] == 1]

# Filter rows where the FECHA_DEF column is not equal to '9999/99/99'
df_diabetes = df_diabetes[df_diabetes['FECHA_DEF'] != '99/99/9999']

# Convert date columns to datetime type with the correct format
df_diabetes['FECHA_INGRESO'] = pd.to_datetime(df_diabetes['FECHA_INGRESO'], format='%d/%m/%Y')
df_diabetes['FECHA_SINTOMAS'] = pd.to_datetime(df_diabetes['FECHA_SINTOMAS'], format='%d/%m/%Y')
df_diabetes['FECHA_DEF'] = pd.to_datetime(df_diabetes['FECHA_DEF'], format='%d/%m/%Y')

# Calculate survival time in days
df_diabetes['TIEMPO_SUP'] = (df_diabetes['FECHA_DEF'] - df_diabetes['FECHA_INGRESO']).dt.days

# Calculate the RMST
time_horizon = 365  # Time horizon in days (1 year)
rmst = np.mean(np.minimum(df_diabetes['TIEMPO_SUP'], time_horizon))

print(f'RMST de personas con DIABETES: {rmst} días')

# Calculate 95% confidence interval for RMST
se = (np.var(df_diabetes['TIEMPO_SUP']) / len(df_diabetes)) ** 0.5
t_value = stats.t.ppf(0.975, len(df_diabetes) - 1)
rmst_lower = rmst - t_value * se
rmst_upper = rmst + t_value * se

# Calculate RMST ratio
no_diabetes_group = df[df['DIABETES'] == 0]
no_diabetes_group = no_diabetes_group[no_diabetes_group['FECHA_DEF'] != '99/99/9999']

# Check if dataframes are not empty before calculations
if not df_diabetes.empty and not no_diabetes_group.empty:
    no_diabetes_group['FECHA_INGRESO'] = pd.to_datetime(no_diabetes_group['FECHA_INGRESO'], format='%d/%m/%Y')
    no_diabetes_group['FECHA_SINTOMAS'] = pd.to_datetime(no_diabetes_group['FECHA_SINTOMAS'], format='%d/%m/%Y')
    no_diabetes_group['FECHA_DEF'] = pd.to_datetime(no_diabetes_group['FECHA_DEF'], format='%d/%m/%Y')
    no_diabetes_group['TIEMPO_SUP'] = (no_diabetes_group['FECHA_DEF'] - no_diabetes_group['FECHA_INGRESO']).dt.days
    rmst_no_diabetes = np.mean(np.minimum(no_diabetes_group['TIEMPO_SUP'], time_horizon))
    rmst_ratio = rmst / rmst_no_diabetes

    # Calculate 95% confidence interval for RMST ratio
    se_ratio = ((np.var(df_diabetes['TIEMPO_SUP']) / len(df_diabetes)) + (np.var(no_diabetes_group['TIEMPO_SUP']) / len(no_diabetes_group))) ** 0.5
    t_value_ratio = stats.t.ppf(0.975, len(df_diabetes) + len(no_diabetes_group) - 2)
    rmst_ratio_lower = rmst_ratio - t_value_ratio * se_ratio
    rmst_ratio_upper = rmst_ratio + t_value_ratio * se_ratio
else:
    # Handle the case when one of the dataframes is empty
    rmst_no_diabetes = np.nan
    rmst_ratio = np.nan
    rmst_ratio_lower = np.nan
    rmst_ratio_upper = np.nan

# Create a table to display the results
results_table = pd.DataFrame({
    'RMST estimate': [rmst, rmst_ratio],
    '95% CI': [(rmst_lower, rmst_upper), (rmst_ratio_lower, rmst_ratio_upper)],
    'p Value': ['p < 0.01', 'p < 0.01']
}, index=['RMST (Diabetes) – (No diabetes)', 'RMST (Diabetes) / (No diabetes)'])

print("TABLE 3b RMST multivariable.")
print(results_table)
