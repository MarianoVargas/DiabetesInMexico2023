import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def load_data(file_path):
    return pd.read_csv(file_path, encoding="ISO-8859-1")

# Load the dataset
file_path = "COVID19MEXICO.csv"
df = load_data(file_path)
if df is None:
    raise ValueError("Data loading failed. Check the file path or data loading process.")

# Filter rows where the DIABETES column equals 1 (or the value indicating diabetes presence)
df_diabetes = df[df['DIABETES'] == 1]

# Filter rows where the FECHA_DEF column is not equal to '9999-99-99'
df_diabetes = df_diabetes[df_diabetes['FECHA_DEF'] != '99/99/9999']

# Convert date columns to datetime type
df_diabetes['FECHA_DEF'] = pd.to_datetime(df_diabetes['FECHA_DEF'], format='%d/%m/%Y', dayfirst=True)
df_diabetes['FECHA_INGRESO'] = pd.to_datetime(df_diabetes['FECHA_INGRESO'], format='%d/%m/%Y', dayfirst=True)

# Calculate survival time in days
df_diabetes['TIEMPO_SUP'] = (df_diabetes['FECHA_DEF'] - df_diabetes['FECHA_INGRESO']).dt.days

# Set the time horizon (in days)
time_horizon = 365  # 1 year

# Filter rows where the DIABETES column equals 0 (no diabetes)
no_diabetes_group = df[df['DIABETES'] == 0]

no_diabetes_group = no_diabetes_group[no_diabetes_group['FECHA_DEF'] != '99/99/9999']

# Convert date columns to datetime type
no_diabetes_group['FECHA_DEF'] = pd.to_datetime(no_diabetes_group['FECHA_DEF'], format='%d/%m/%Y', dayfirst=True)
no_diabetes_group['FECHA_INGRESO'] = pd.to_datetime(no_diabetes_group['FECHA_INGRESO'], format='%d/%m/%Y', dayfirst=True)

# Calculate survival time in days for people without diabetes
no_diabetes_group['TIEMPO_SUP'] = (no_diabetes_group['FECHA_DEF'] - no_diabetes_group['FECHA_INGRESO']).dt.days

# Calculate the RMST for people without diabetes
rmst_no_diabetes = np.mean(np.minimum(no_diabetes_group['TIEMPO_SUP'], time_horizon))

# Calculate the RMST for people with diabetes
rmst_diabetes = np.mean(np.minimum(df_diabetes['TIEMPO_SUP'], time_horizon))

# Perform t-test to calculate p-value
_, p_value = ttest_ind(df_diabetes['TIEMPO_SUP'], no_diabetes_group['TIEMPO_SUP'], equal_var=False)

# Create arrays for 'Group,' 'RMST estimate,' and 'p Value'
groups = ['Diabetes', 'No Diabetes']
rmst_estimates = [rmst_diabetes, rmst_no_diabetes]
p_value_formatted = [f'p < {0.01 if p_value < 0.01 else p_value:.2f}']

# Print the lengths of the arrays for debugging
print(f'Length of groups: {len(groups)}')
print(f'Length of rmst_estimates: {len(rmst_estimates)}')
print(f'Length of p_value_formatted: {len(p_value_formatted)}')

# Convert date columns to datetime type
no_diabetes_group['FECHA_DEF'] = pd.to_datetime(no_diabetes_group['FECHA_DEF'], format='%d/%m/%Y', dayfirst=True)
no_diabetes_group['FECHA_INGRESO'] = pd.to_datetime(no_diabetes_group['FECHA_INGRESO'], format='%d/%m/%Y', dayfirst=True)

# Calculate survival time in days for people without diabetes
no_diabetes_group['TIEMPO_SUP'] = (no_diabetes_group['FECHA_DEF'] - no_diabetes_group['FECHA_INGRESO']).dt.days

# Calculate the RMST for people without diabetes
rmst_no_diabetes = np.mean(np.minimum(no_diabetes_group['TIEMPO_SUP'], time_horizon))

# Perform t-test to calculate p-value
_, p_value = ttest_ind(df_diabetes['TIEMPO_SUP'], no_diabetes_group['TIEMPO_SUP'], equal_var=False)

# Create arrays for 'Group,' 'RMST estimate,' and 'p Value'
groups = ['Diabetes', 'No Diabetes']
rmst_estimates = [rmst_diabetes, rmst_no_diabetes]
p_values = [p_value, p_value]  # Assign the p-value to both groups

# Format p-values
p_value_formatted = [f'p < {0.01 if p_val < 0.01 else p_val:.2f}' for p_val in p_values]

# Ensure that all arrays have the same length (2 in this case)
assert len(groups) == len(rmst_estimates) == len(p_value_formatted) == 2, "Arrays must have the same length"

# Create a table to display the results
results_table = pd.DataFrame({
    'Group': groups,
    'RMST estimate': rmst_estimates,
    'p Value': p_value_formatted
})

print(results_table)