import pandas as pd
from pathlib import Path

# =========================================================
# 0. RUTAS DE ARCHIVOS (ADAPTA SI CAMBIAN)
# =========================================================
FILE_17 = '/content/CC FT 17   Formato de Control de Calidad Café de Trillado (1).xlsx'
FILE_18 = '/content/CC FT 18  Formato de  Tostión (1).xlsx'

# =========================================================
# 1. UTILIDADES
# =========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas Unnamed y normaliza a MAYÚSCULAS sin espacios extremos."""
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', case=False, na=False)]
    df.columns = (df.columns
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.upper())
    return df

def normalizar_texto(serie: pd.Series, keep_single_space=True, remove_all_spaces=False, upper=True):
    s = serie.astype(str).str.strip()
    if keep_single_space:
        s = s.str.replace(r'\s+', ' ', regex=True)
    if remove_all_spaces:
        s = s.str.replace(' ', '', regex=False)
    if upper:
        s = s.str.upper()
    return s

def asegurar_col(df, col):
    if col not in df.columns:
        raise KeyError(f"❌ Falta la columna requerida: '{col}'")

def buscar_variantes(df, posibles):
    """Devuelve el primer nombre de columna que exista entre una lista de posibles."""
    for p in posibles:
        if p in df.columns:
            return p
    raise KeyError(f"Ninguna de las variantes {posibles} se encontró en columnas: {list(df.columns)}")

# =========================================================
# 2. LECTURA DE EXCEL Y CONCATENACIÓN
# =========================================================
# Hojas 2017
df1 = pd.read_excel(FILE_17, sheet_name='CONTROL CALIDAD CAFE TRILLADO J', skiprows=5)
df2 = pd.read_excel(FILE_17, sheet_name='Sheet2', skiprows=5)
df17 = pd.concat([df1, df2], ignore_index=True)

# Hojas 2018
df3 = pd.read_excel(FILE_18, sheet_name='TOSTIÓN JERICÓ L', skiprows=5)
df4 = pd.read_excel(FILE_18, sheet_name='TOSTIÓN JERICÓ', skiprows=5)
df18 = pd.concat([df3, df4], ignore_index=True)

# Limpieza de columnas
df17 = clean_columns(df17)
df18 = clean_columns(df18)

# Eliminar filas iniciales vacías / numeración si corresponde

if 1 in df17.index and 0 in df17.index:
    df17 = df17.drop(index=[0, 1]).reset_index(drop=True)

# =========================================================
# 3. NORMALIZACIÓN DE COLUMNAS CLAVE
# =========================================================
# Posibles nombres de la columna de denominación / marca (ajusta si ves otra)
COL_DENOM = buscar_variantes(
    df17,
    ['DENOMINACIÓN/ MARCA', 'DENOMINACIÓN/MARCA', 'DENOMINACION/ MARCA', 'DENOMINACION/ MARCA', 'DENOMINACIÓN/     MARCA']
)


# Limpieza de texto en denominación
df17[COL_DENOM] = normalizar_texto(df17[COL_DENOM])
if COL_DENOM in df18.columns:
    df18[COL_DENOM] = normalizar_texto(df18[COL_DENOM])

# =========================================================
# 4. CREACIÓN DEL MAPEO DE VARIEDADES
# =========================================================
variedades_unicas = df17[COL_DENOM].dropna().unique()

mapeo_variedades = {
    'TABI NATURAL': 'TABI',
    'DON MARIO': 'DOS MIL',
    'MONTEVERDE - WUSH WUSH': 'WUSH WUSH',
    'DON FELIX': 'DOS MIL',
    'DOÑA DOLLY': 'DOS MIL',
    'MADRE LAURA NATURAL': 'DOS MIL',
    'MADRE LAURA': 'DOS MIL',
    'GESHA VILLABERNARDA': 'GESHA',
    'GESHA VILLA - NATURAL': 'GESHA',
    'GESHA BLUE - MONTEVERDE': 'GESHA',
    'EL OCASO - CATURRON': 'CATURRA'
}

# Prefijo DON*
mapeo_variedades.update({
    nombre: 'DOS MIL'
    for nombre in variedades_unicas
    if nombre.startswith('DON ') and nombre not in mapeo_variedades
})

# Contiene BOURBON
mapeo_variedades.update({
    nombre: 'BOURBON ROJO'
    for nombre in variedades_unicas
    if 'BOURBON' in nombre and nombre not in mapeo_variedades
})

# Resto → OTROS
mapeo_variedades.update({
    nombre: 'OTROS'
    for nombre in variedades_unicas
    if nombre not in mapeo_variedades
})

df17['VARIEDAD_ESTANDAR'] = df17[COL_DENOM].map(mapeo_variedades)
if COL_DENOM in df18.columns:
    df18['VARIEDAD_ESTANDAR'] = df18[COL_DENOM].map(mapeo_variedades)

# =========================================================
# 5. DEFINIR df BASE (puedes decidir: usar df17, df18 o merge)
# =========================================================
# Aquí empiezo con df = df17. Luego podemos añadir info de df18 por LOTE / VARIEDAD, etc.
df = df17.copy()

# =========================================================
# 6. MAPEO PROCESO Y TIEMPOS
# =========================================================
# TODO: Rellena según tu realidad
proceso_por_variedad = {
    'GESHA': 'LAVADO',
    'TABI': 'HONEY',
    'DOS MIL': 'NATURAL',
    'BOURBON ROJO': 'LAVADO',
    'CATURRA': 'LAVADO',
    'OTROS': 'NATURAL'
}

tiempo_por_proceso = {
     'LAVADO': 48,
     'HONEY': 60,
     'NATURAL': 72
}

if proceso_por_variedad:
    df['PROCESO'] = df['VARIEDAD_ESTANDAR'].map(proceso_por_variedad)
else:
    print("⚠️ Define 'proceso_por_variedad' para crear la columna PROCESO.")

if tiempo_por_proceso and 'PROCESO' in df.columns:
    df['TIEMPO_DE_TUESTE_MIN'] = df['PROCESO'].map(tiempo_por_proceso)

# =========================================================
# 7. PARSEAR TIEMPO DE TUESTE EN df18
# =========================================================
# Posibles variantes (verifica exacto)
COL_TIEMPO_TUESTE = buscar_variantes(df18, ['TIEMPO DE TUESTE'])
if COL_TIEMPO_TUESTE in df18.columns:
    df18['TIEMPO_DELTA'] = pd.to_timedelta(df18[COL_TIEMPO_TUESTE].astype(str), errors='coerce')
    df18['TIEMPO DE TUESTE EN MIN'] = df18['TIEMPO_DELTA'].dt.total_seconds() / 60
    df18 = df18.drop(columns=[COL_TIEMPO_TUESTE, 'TIEMPO_DELTA'], errors='ignore')

# =========================================================
# 8. BENEFICIO POR PROCESO DESDE df18
# =========================================================
if 'PROCESO' in df18.columns and 'BENEFICIO' in df18.columns:
    beneficio_por_proceso = (
        df18.dropna(subset=['PROCESO'])
             .drop_duplicates('PROCESO')[['PROCESO', 'BENEFICIO']]
             .set_index('PROCESO')['BENEFICIO']
             .to_dict()
    )
    if 'PROCESO' in df.columns:
        df['BENEFICIO'] = df['PROCESO'].map(beneficio_por_proceso)

# =========================================================
# 9. LIMPIEZA DE RESPONSABLE / ORIGEN
# =========================================================
for col_resp in ['RESPONSABLE']:
    if col_resp in df.columns:
        df[col_resp] = normalizar_texto(df[col_resp],
                                        keep_single_space=True,
                                        remove_all_spaces=True,
                                        upper=True)

if 'ORIGEN' in df.columns:
    df['ORIGEN'] = normalizar_texto(df['ORIGEN'], keep_single_space=True, upper=True)
    df['ORIGEN'] = df['ORIGEN'].replace({'HERRRA': 'HERRERA'})

# =========================================================
# 10. MERGE OPCIONAL DE INFORMACIÓN DE df18
# =========================================================
if all(col in df18.columns for col in ['PROCESO', 'TIEMPO DE TUESTE EN MIN']) and 'PROCESO' in df.columns:
    # 1. Diccionario: mediana por proceso (solo procesos con datos no nulos)
    tiempo_por_proceso = (
        df18.dropna(subset=['TIEMPO DE TUESTE EN MIN'])
            .groupby('PROCESO')['TIEMPO DE TUESTE EN MIN']
            .median()
            .to_dict()
    )

    # 2. Mapear a df
    df['TIEMPO_TUESTE_MEDIAN'] = df['PROCESO'].map(tiempo_por_proceso)

    # 3. Mediana global como fallback
    tiempo_global = df18['TIEMPO DE TUESTE EN MIN'].median()
    df['TIEMPO_TUESTE_MEDIAN'] = df['TIEMPO_TUESTE_MEDIAN'].fillna(tiempo_global)

    # (Opcional) Si quiere redondear:
    # df['TIEMPO_TUESTE_MEDIAN'] = df['TIEMPO_TUESTE_MEDIAN'].round(2)



# =========================================================
# 11. INGENIERÍA DE CARACTERÍSTICAS
# =========================================================
# Ejemplo: estadísticos por VARIEDAD_ESTANDAR
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    stats = (df.groupby('VARIEDAD_ESTANDAR')[numeric_cols]
               .agg(['mean', 'median', 'std']))
    stats.columns = [f"{c}_{f.upper()}" for c, f in stats.columns]
    stats = stats.reset_index()
    df = df.merge(stats, on='VARIEDAD_ESTANDAR', how='left')

# =========================================================
# 12. ORDEN DE COLUMNAS Y EXPORTACIÓN
# =========================================================
prefer = [c for c in [
    'VARIEDAD_ESTANDAR', 'PROCESO', 'BENEFICIO', 'TIEMPO_DE_TUESTE_MIN',
    'TIEMPO DE TUESTE EN MIN', 'RESPONSABLE', 'ORIGEN'
] if c in df.columns]
otros = [c for c in df.columns if c not in prefer]
df = df[prefer + otros]

OUTPUT = "dataset_final_procesado.csv"
df.to_csv(OUTPUT, index=False)
print(f"✅ Exportado: {OUTPUT}")
print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
