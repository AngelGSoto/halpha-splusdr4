import pandas as pd
import re

# Ruta al archivo .dat
file_path = '../Be-stars/Be-tars.dat'
error_log_path = 'errores.log'

# Función para inspeccionar el archivo y mostrar las primeras líneas
def inspect_file(file_path, num_lines=20):
    with open(file_path, 'r', encoding='utf-8') as file:
        print(f"Inspección de las primeras {num_lines} líneas del archivo:")
        for i in range(num_lines):
            print(file.readline().strip())

# Función para registrar errores en un archivo separado
def log_error(line, fields):
    with open(error_log_path, 'a', encoding='utf-8') as error_file:
        error_file.write(f"Error en la línea: {line.strip()}\n")
        error_file.write(f"Campos detectados: {fields}\n\n")

# Función para leer el archivo y procesar los datos
def read_file(file_path):
    data = []
    columns = ['Number', 'Be star', 'Category', 'RA', 'DEC', 'V', 'Type', 'vsini', 'NbBeSS']
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Saltar la primera línea que contiene el encabezado
        header_line = file.readline().strip()
        print("Encabezado detectado:", header_line)
        
        for line in file:
            # Dividir la línea en campos utilizando tabuladores y espacios, preservando los valores vacíos
            fields = re.split(r'\t+', line.strip())
            if len(fields) == 1:
                fields = re.split(r'\s+', line.strip())
                
            # Revisar y ajustar los campos para que coincidan con el número de columnas
            if len(fields) != len(columns):
                log_error(line, fields)
                # Ajustar los campos para que coincidan con el número de columnas
                while len(fields) > len(columns):
                    fields[-2] += ' ' + fields.pop()
                while len(fields) < len(columns):
                    fields.append('')
            
            data.append(fields)
    
    return columns, data

# Función para convertir RA (h m s) a grados decimales
def ra_to_decimal(ra):
    if not ra.strip():
        return None
    h, m, s = map(float, ra.split())
    return 15 * (h + m / 60 + s / 3600)

# Función para convertir DEC (deg m s) a grados decimales
def dec_to_decimal(dec):
    if not dec.strip():
        return None
    # Manejar el signo de la declinación
    sign = -1 if dec.strip()[0] == '-' else 1
    # Eliminar el signo para evitar problemas en la separación de los componentes
    dec = dec.strip().lstrip('-').lstrip('+')
    parts = re.split('[^\d.]+', dec)
    if len(parts) != 3:
        return None
    try:
        d, m, s = map(float, parts)
    except ValueError:
        return None
    return sign * (d + m / 60 + s / 3600)

# Inspeccionar las primeras líneas del archivo
inspect_file(file_path)

# Leer el archivo y obtener encabezado y datos
columns, data = read_file(file_path)

# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(data, columns=columns)

# Convertir las columnas RA y DEC a grados decimales
df['RA_decimal'] = df['RA'].apply(ra_to_decimal)
df['DEC_decimal'] = df['DEC'].apply(dec_to_decimal)

# Verificar las primeras filas del DataFrame para asegurarnos de que se ha leído correctamente
print("Primeras filas del DataFrame:")
print(df.head())

# Guardar el DataFrame en un archivo CSV
df.to_csv('bess_catalog_converted.csv', index=False)
print("Datos guardados en 'bess_catalog_converted.csv'.")
