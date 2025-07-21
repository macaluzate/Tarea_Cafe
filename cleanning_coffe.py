import pandas as pd
import numpy as np

def clean_columns(df):
    """
    Función para limpiar nombres de columnas
    """
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # quitar columnas sin nombre
    df.columns = df.columns.str.strip().str.upper()       # espacios y mayúsculas
    return df

def cargar_datasets():
    """
    Función para cargar y limpiar los datasets desde archivos Excel

    Returns:
        df17: DataFrame del dataset 17 limpio
        df18: DataFrame del dataset 18 limpio
    """
    # Rutas exactas de los archivos
    file_17 = '/content/CC FT 17   Formato de Control de Calidad Café de Trillado (1).xlsx'
    file_18 = '/content/CC FT 18  Formato de  Tostión (1).xlsx'

    # Cargar los datos del archivo 17
    df1 = pd.read_excel(file_17, sheet_name='CONTROL CALIDAD CAFE TRILLADO J', skiprows=5)
    df2 = pd.read_excel(file_17, sheet_name='Sheet2', skiprows=5)
    df17 = pd.concat([df1, df2], ignore_index=True)

    # Cargar los datos del archivo 18
    df3 = pd.read_excel(file_18, sheet_name='TOSTIÓN JERICÓ L', skiprows=5)
    df4 = pd.read_excel(file_18, sheet_name='TOSTIÓN JERICÓ', skiprows=5)
    df18 = pd.concat([df3, df4], ignore_index=True)

    # Limpiar columnas
    df17 = clean_columns(df17)
    df18 = clean_columns(df18)

    # Normalizar nombres de columnas
    df17.columns = df17.columns.str.strip().str.upper()
    df18.columns = df18.columns.str.strip().str.upper()

    # Limpiar filas vacías y reinicializar índices
    df17 = df17.drop(index=[0,1])               # borra la fila vacía y la de 'N°'
    df17 = df17.reset_index(drop=True)          # reindexa
    df17 = df17.drop(index=range(125, 139))     # excluye filas 125-138
    df17 = df17.reset_index(drop=True)          # reindexa después de eliminar

    # Eliminar filas completamente vacías o con datos insuficientes
    df17 = df17.dropna(how='all')  # elimina filas completamente vacías
    df17 = df17.dropna(subset=['DENOMINACIÓN/     MARCA'], how='any')  # elimina filas sin denominación
    df17 = df17.reset_index(drop=True)  # reindexa después de limpiar

    # Convertir tiempo de tueste a minutos en df18
    col = 'TIEMPO DE TUESTE'

    # Homogeneizar a string y parsear como timedelta
    df18['TIEMPO_DELTA'] = pd.to_timedelta(df18[col].astype(str), errors='coerce')

    # Extraer minutos
    df18['TIEMPO DE TUESTE EN MIN'] = df18['TIEMPO_DELTA'].dt.total_seconds() / 60

    # Eliminar columnas intermedias y la original
    df18 = df18.drop(columns=[col, 'TIEMPO_DELTA'])

    print(df18[['TIEMPO DE TUESTE EN MIN']].head())

    return df17, df18

def unificar_datasets(df17, df18):
    """
    Función para unificar y limpiar los datasets 17 y 18

    Args:
        df17: DataFrame del dataset 17
        df18: DataFrame del dataset 18

    Returns:
        df: DataFrame unificado y limpio
    """

    # Definir el diccionario base de mapeo de variedades
    mapeo_variedades = {
        # Mapeos específicos
        'Tabi Natural': 'Tabi',
        'Don Mario': 'Dos mil',
        'Monteverde - Wush Wush': 'Wush Wush',
        'Don Felix': 'Dos mil',
        'Doña Dolly': 'Dos mil',
        'Madre Laura Natural': 'Dos mil',
        'Madre Laura': 'Dos mil',

        # Todo lo Gesha
        'Gesha Villabernarda': 'Gesha',
        'Gesha Villa - Natural': 'Gesha',
        'Gesha Blue - Monteverde': 'Gesha',

        # Todo lo Don -> Dos mil
        **{nombre: 'Dos mil' for nombre in df17['DENOMINACIÓN/     MARCA'].unique()
           if isinstance(nombre, str) and nombre.startswith('Don')},

        # Bourbon a Bourbon Rojo
        **{nombre: 'Bourbon Rojo' for nombre in df17['DENOMINACIÓN/     MARCA'].unique()
           if isinstance(nombre, str) and 'Bourbon' in nombre},

        # Caso especial
        'El Ocaso - Caturron': 'Caturra'
    }

    # Añadir el caso "Otros" por separado
    otros_mapes = {
        nombre: 'Otros' for nombre in df17['DENOMINACIÓN/     MARCA'].unique()
        if isinstance(nombre, str) and nombre not in mapeo_variedades
    }

    # Combinar ambos diccionarios
    mapeo_variedades.update(otros_mapes)

    # Crear una copia de df17 para no modificarlo
    df = df17.copy()

    # Aplicar el mapeo al nuevo DataFrame 'df'
    df['VARIEDAD_ESTANDAR'] = df['DENOMINACIÓN/     MARCA'].map(mapeo_variedades)

    # Manejar valores no mapeados
    df['VARIEDAD_ESTANDAR'] = df['VARIEDAD_ESTANDAR'].fillna('Otros')

    # Limpiar espacios en RESPONSABLE
    df['RESPONSABLE'] = df['RESPONSABLE'].str.strip()

    # Limpiar espacios en la columna 'DENOMINACIÓN/MARCA'
    df['DENOMINACIÓN/     MARCA'] = df['DENOMINACIÓN/     MARCA'].str.strip()

    # Limpiar espacios en VARIEDAD del df18
    df18['VARIEDAD'] = df18['VARIEDAD'].str.strip()

    # Extraer pares únicos de VARIEDAD y PROCESO de df18
    proceso_por_variedad = (
        df18.drop_duplicates('VARIEDAD')[['VARIEDAD', 'PROCESO']]
        .set_index('VARIEDAD')['PROCESO']
        .to_dict()
    )

    # Mapear PROCESO al DataFrame principal
    df['PROCESO'] = df['VARIEDAD_ESTANDAR'].map(proceso_por_variedad)

    # Extraer pares únicos de VARIEDAD y ORIGEN de df18
    origen_por_variedad = (
        df18.drop_duplicates('VARIEDAD')[['VARIEDAD', 'ORIGEN']]
        .set_index('VARIEDAD')['ORIGEN']
        .to_dict()
    )

    # Mapear ORIGEN al DataFrame principal
    df['ORIGEN'] = df['VARIEDAD_ESTANDAR'].map(origen_por_variedad)

    # Limpiar y unificar ORIGEN
    df['ORIGEN'] = (
        df['ORIGEN']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Mapear las variantes "HERRRA" y "HERRERA" a una sola
    df['ORIGEN'] = df['ORIGEN'].replace({
        'HERRRA': 'HERRERA',
        'HERRERA': 'HERRERA'
    })

    # Calcular la mediana del tiempo por PROCESO en df18
    tiempo_por_proceso = (
        df18.groupby('PROCESO')['TIEMPO DE TUESTE EN MIN']
        .median()
        .to_dict()
    )

    # Mapear tiempo de tueste median
    df['TIEMPO_TUESTE_MEDIAN'] = df['PROCESO'].map(tiempo_por_proceso)

    # Rellenar NaN con la mediana global (si hay procesos sin datos)
    tiempo_global = df18['TIEMPO DE TUESTE EN MIN'].median()
    df['TIEMPO_TUESTE_MEDIAN'] = df['TIEMPO_TUESTE_MEDIAN'].fillna(tiempo_global)

    # Extraer pares únicos de PROCESO y BENEFICIO de df18
    beneficio_por_proceso = (
        df18.drop_duplicates('PROCESO')[['PROCESO', 'BENEFICIO']]
        .set_index('PROCESO')['BENEFICIO']
        .to_dict()
    )

    # Mapear BENEFICIO al DataFrame principal
    df['BENEFICIO'] = df['PROCESO'].map(beneficio_por_proceso)

    return df

def calcular_correlaciones(df):
    """
    Función para calcular la matriz de correlación de las columnas numéricas

    Args:
        df: DataFrame unificado

    Returns:
        corr_matrix: Matriz de correlación
    """
    # Seleccionar todas las columnas numéricas del DataFrame fusionado
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Calcular la matriz de correlación
    corr_matrix = df[num_cols].corr()

    return corr_matrix

# Ejecutar el proceso de unificación
print("Iniciando proceso de unificación de datasets...")

# Cargar los datasets desde archivos Excel
df17, df18 = cargar_datasets()
print(f"Dataset 17 cargado: {len(df17)} filas, {len(df17.columns)} columnas")
print(f"Dataset 18 cargado: {len(df18)} filas, {len(df18.columns)} columnas")

# Ejecutar la unificación
df_unificado = unificar_datasets(df17, df18)
print(f"Unificación completada: {len(df_unificado)} filas, {len(df_unificado.columns)} columnas")

# Calcular correlaciones
matriz_correlacion = calcular_correlaciones(df_unificado)
print("Matriz de correlación calculada")

# Guardar el dataset unificado en CSV
df_unificado.to_csv('dataset_unificado.csv', index=False)
print("Dataset unificado guardado en 'dataset_unificado.csv'")

print("Proceso completado exitosamente!")
