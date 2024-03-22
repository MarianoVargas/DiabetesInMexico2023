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

# Filtrar las filas donde la columna DIABETES es igual a 1 (o el valor que indique la presencia de diabetes)
df_diabetes = df[df['DIABETES'] == 1]

# Filtrar las filas donde la columna FECHA_DEF no sea igual a '9999-99-99'
df_diabetes = df_diabetes[df_diabetes['FECHA_DEF'] != '99/99/9999']

# Convertir las columnas de fecha a tipo datetime
df_diabetes['FECHA_DEF'] = pd.to_datetime(df_diabetes['FECHA_DEF'], format='%d/%m/%Y', dayfirst=True)
df_diabetes['FECHA_INGRESO'] = pd.to_datetime(df_diabetes['FECHA_INGRESO'], format='%d/%m/%Y', dayfirst=True)

# Calcular el tiempo de supervivencia en días
df_diabetes['TIEMPO_SUP'] = (df_diabetes['FECHA_DEF'] - df_diabetes['FECHA_INGRESO']).dt.days

# Calcular el RMST
time_horizon = 365  # Horizonte de tiempo en días (1 año)
rmst = np.mean(np.minimum(df_diabetes['TIEMPO_SUP'], time_horizon))

print(f'RMST univariable de personas con DIABETES: {rmst} días')
