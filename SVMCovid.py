import pandas as pd

# Define the columns of interest
columns_of_interest = [
    "FECHA_ACTUALIZACION", "ID_REGISTRO", "ORIGEN", "SECTOR", "ENTIDAD_UM",
    "SEXO", "ENTIDAD_NAC", "ENTIDAD_RES", "MUNICIPIO_RES", "TIPO_PACIENTE",
    "FECHA_INGRESO", "FECHA_SINTOMAS", "FECHA_DEF", "INTUBADO", "NEUMONIA",
    "EDAD", "NACIONALIDAD", "EMBARAZO", "HABLA_LENGUA_INDIG", "INDIGENA",
    "DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION", "OTRA_COM",
    "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", "TABAQUISMO", "OTRO_CASO",
    "TOMA_MUESTRA_LAB", "RESULTADO_LAB", "TOMA_MUESTRA_ANTIGENO", "RESULTADO_ANTIGENO",
    "CLASIFICACION_FINAL", "MIGRANTE", "PAIS_NACIONALIDAD", "PAIS_ORIGEN", "UCI"
]

# Read the CSV file with pandas
file_path = "COVID19MEXICO.csv"  # Replace with your actual file path
data = pd.read_csv(file_path, usecols=columns_of_interest, encoding="ISO-8859-1")

# Find the start and end of the data
start_date = data["FECHA_INGRESO"].min()
end_date = data["FECHA_ACTUALIZACION"].max()

# Print the start and end dates
print("Start Date:", start_date)
print("End Date:", end_date)