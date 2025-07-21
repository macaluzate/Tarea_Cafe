#!/usr/bin/env python3
"""
Script de preprocesamiento y modelado para dataset_unificado.csv
Implementa Random Forest, Lasso y Ridge con preprocesamiento completo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(filepath):
    """Carga y muestra información básica del dataset"""
    print("="*60)
    print("CARGANDO Y EXPLORANDO DATOS")
    print("="*60)

    df = pd.read_csv(filepath)
    print(f"Dimensiones del dataset: {df.shape}")
    print(f"\nColumnas disponibles:\n{list(df.columns)}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    print(f"\nValores nulos por columna:\n{df.isnull().sum()}")

    return df

def clean_numeric_columns(df):
    """Limpia y convierte columnas numéricas que tienen formato incorrecto"""
    print("\n" + "="*60)
    print("LIMPIEZA DE COLUMNAS NUMÉRICAS")
    print("="*60)

    # Columnas que deben ser numéricas pero pueden tener formato incorrecto
    numeric_cols_to_clean = ['%H', 'PUNTAJE']

    for col in numeric_cols_to_clean:
        if col in df.columns:
            print(f"Limpiando columna {col}...")
            # Mostrar algunos valores originales
            print(f"  Valores únicos antes: {df[col].unique()[:10]}")

            # Convertir comas a puntos y luego a float
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

            # Convertir a float explícitamente
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

            print(f"  Tipo después de conversión: {df[col].dtype}")
            print(f"  Valores nulos después de conversión: {df[col].isnull().sum()}")

            if not df[col].isnull().all():
                print(f"  Rango: {df[col].min():.2f} - {df[col].max():.2f}")
            else:
                print(f"  ⚠️  Todos los valores son nulos después de la conversión")

    print(f"\nTipos de datos después de limpieza:\n{df.dtypes}")
    print(f"\nEstadísticas descriptivas de columnas numéricas:\n{df.select_dtypes(include=['float64', 'int64']).describe()}")

    return df

def define_variables(df):
    """Define las variables numéricas y categóricas para el modelo"""
    print("\n" + "="*60)
    print("DEFINICIÓN DE VARIABLES")
    print("="*60)

    # Variables numéricas predefinidas
    numeric_cols = [
        '%H',
        'TIEMPO_TUESTE_MEDIAN',
        'CANTIDAD'
    ]

    # Variables categóricas predefinidas
    categorical_cols = [
        'ORIGEN',
        'RESPONSABLE',
        'VARIEDAD_ESTANDAR',
        'PROCESO',
        'BENEFICIO'
    ]

    # Verificar que las columnas existen en el dataset
    available_numeric = [col for col in numeric_cols if col in df.columns]
    available_categorical = [col for col in categorical_cols if col in df.columns]

    missing_numeric = set(numeric_cols) - set(available_numeric)
    missing_categorical = set(categorical_cols) - set(available_categorical)

    if missing_numeric:
        print(f"⚠️  Columnas numéricas no encontradas: {missing_numeric}")
    if missing_categorical:
        print(f"⚠️  Columnas categóricas no encontradas: {missing_categorical}")

    print(f"✅ Variables numéricas disponibles: {available_numeric}")
    print(f"✅ Variables categóricas disponibles: {available_categorical}")

    # Verificar que PUNTAJE existe como variable objetivo
    if 'PUNTAJE' not in df.columns:
        print("❌ ERROR: Columna 'PUNTAJE' no encontrada en el dataset")
        return None, None, None

    return available_numeric, available_categorical, df

def prepare_data(df, numeric_cols, categorical_cols):
    """Prepara los datos para el modelado"""
    print("\n" + "="*60)
    print("PREPARACIÓN DE DATOS")
    print("="*60)

    # Verificar que PUNTAJE existe y es numérico
    if 'PUNTAJE' not in df.columns:
        print("❌ ERROR: Columna 'PUNTAJE' no encontrada")
        return None, None, None, None

    if df['PUNTAJE'].dtype == 'object':
        print("⚠️  PUNTAJE aún es tipo object. Intentando conversión...")
        df['PUNTAJE'] = pd.to_numeric(df['PUNTAJE'], errors='coerce')

    # Separar características y variable objetivo
    X = df[numeric_cols + categorical_cols]
    y = df['PUNTAJE']

    print(f"Características (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")
    print(f"Valores nulos en y: {y.isnull().sum()}")
    print(f"Rango de y: {y.min():.2f} - {y.max():.2f}")

    # Eliminar filas donde y es nulo
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    print(f"Después de eliminar nulos en y: X={X.shape}, y={y.shape}")

    if len(X) == 0:
        print("❌ ERROR: No hay datos válidos después de la limpieza")
        return None, None, None, None

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")

    return X_train, X_test, y_train, y_test

def create_preprocessor(numeric_cols, categorical_cols):
    """Crea el preprocesador con imputación y escalado"""
    print("\n" + "="*60)
    print("CREANDO PREPROCESADOR")
    print("="*60)

    # Imputador para variables numéricas (llenar con -1)
    num_imputer = SimpleImputer(strategy='constant', fill_value=-1)

    # Imputador para variables categóricas (llenar con "MISSING")
    cat_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')

    # Preprocesador completo
    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', cat_imputer),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols),
        ('num', Pipeline([
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
        ]), numeric_cols)
    ])

    print("✅ Preprocesador creado:")
    print("   - Variables categóricas: Imputación + OneHotEncoder")
    print("   - Variables numéricas: Imputación + StandardScaler")

    return preprocessor

def train_random_forest(preprocessor, X_train, X_test, y_train, y_test):
    """Entrena y evalúa modelo Random Forest"""
    print("\n" + "="*60)
    print("MODELO RANDOM FOREST")
    print("="*60)

    # Pipeline Random Forest (sin escalado adicional)
    pipe_rf = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    # Entrenamiento
    pipe_rf.fit(X_train, y_train)

    # Predicciones
    y_pred_train = pipe_rf.predict(X_train)
    y_pred_test = pipe_rf.predict(X_test)

    # Métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(pipe_rf, X_train, y_train, cv=5, scoring='r2')

    print("Random Forest - Resultados:")
    print(f"  R² Train: {r2_train:.3f}")
    print(f"  R² Test:  {r2_test:.3f}")
    print(f"  MAE Train: {mae_train:.3f}")
    print(f"  MAE Test:  {mae_test:.3f}")
    print(f"  R² CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return pipe_rf, y_pred_train, y_pred_test

def train_lasso(preprocessor, X_train, X_test, y_train, y_test):
    """Entrena y evalúa modelo Lasso"""
    print("\n" + "="*60)
    print("MODELO LASSO")
    print("="*60)

    # Pipeline Lasso
    pipe_lasso = Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('lasso', Lasso(alpha=0.1, random_state=42))
    ])

    # Entrenamiento
    pipe_lasso.fit(X_train, y_train)

    # Predicciones
    y_pred_train = pipe_lasso.predict(X_train)
    y_pred_test = pipe_lasso.predict(X_test)

    # Métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("Lasso - Resultados:")
    print(f"  R² Train: {r2_train:.3f}")
    print(f"  R² Test:  {r2_test:.3f}")
    print(f"  MAE Train: {mae_train:.3f}")
    print(f"  MAE Test:  {mae_test:.3f}")

    return pipe_lasso, y_pred_train, y_pred_test

def train_ridge(preprocessor, X_train, X_test, y_train, y_test):
    """Entrena y evalúa modelo Ridge"""
    print("\n" + "="*60)
    print("MODELO RIDGE")
    print("="*60)

    # Pipeline Ridge
    pipe_ridge = Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('ridge', Ridge(alpha=0.1, random_state=42))
    ])

    # Entrenamiento
    pipe_ridge.fit(X_train, y_train)

    # Predicciones
    y_pred_train = pipe_ridge.predict(X_train)
    y_pred_test = pipe_ridge.predict(X_test)

    # Métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("Ridge - Resultados:")
    print(f"  R² Train: {r2_train:.3f}")
    print(f"  R² Test:  {r2_test:.3f}")
    print(f"  MAE Train: {mae_train:.3f}")
    print(f"  MAE Test:  {mae_test:.3f}")

    return pipe_ridge, y_pred_train, y_pred_test

def plot_predictions(y_train, y_test, rf_pred_train, rf_pred_test):
    """Crea gráfico de predicciones vs valores reales"""
    print("\n" + "="*60)
    print("GENERANDO GRÁFICOS")
    print("="*60)

    plt.figure(figsize=(12, 5))

    # Gráfico Random Forest
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, rf_pred_train, alpha=0.5, label='Train', color='blue')
    plt.scatter(y_test, rf_pred_test, alpha=0.5, label='Test', color='red')
    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Línea perfecta')
    plt.xlabel("Valor Real")
    plt.ylabel("Predicción")
    plt.title("Random Forest: Predicciones vs Reales")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histograma de residuos
    plt.subplot(1, 2, 2)
    residuos_test = y_test - rf_pred_test
    plt.hist(residuos_test, bins=30, alpha=0.7, color='orange')
    plt.xlabel("Residuos (Real - Predicción)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Residuos (Test)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    """Función principal que ejecuta todo el pipeline"""
    try:
        # 1. Cargar datos
        df = load_and_inspect_data('dataset_unificado.csv')

        # 2. Limpiar columnas numéricas con formato incorrecto
        df = clean_numeric_columns(df)

        # 3. Definir variables
        numeric_cols, categorical_cols, df = define_variables(df)
        if df is None:
            return

        # 4. Preparar datos
        X_train, X_test, y_train, y_test = prepare_data(df, numeric_cols, categorical_cols)
        if X_train is None:
            return

        # 5. Crear preprocesador
        preprocessor = create_preprocessor(numeric_cols, categorical_cols)

        # 6. Entrenar modelos
        pipe_rf, rf_pred_train, rf_pred_test = train_random_forest(
            preprocessor, X_train, X_test, y_train, y_test
        )

        pipe_lasso, lasso_pred_train, lasso_pred_test = train_lasso(
            preprocessor, X_train, X_test, y_train, y_test
        )

        pipe_ridge, ridge_pred_train, ridge_pred_test = train_ridge(
            preprocessor, X_train, X_test, y_train, y_test
        )

        # 7. Visualizar resultados
        plot_predictions(y_train, y_test, rf_pred_train, rf_pred_test)

        print("\n" + "="*60)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("Los modelos han sido entrenados y evaluados.")
        print("Se han generado gráficos de predicciones vs valores reales.")

        return pipe_rf, pipe_lasso, pipe_ridge

    except FileNotFoundError:
        print("❌ ERROR: No se encontró el archivo 'dataset_unificado.csv'")
        print("Asegúrate de que el archivo esté en el mismo directorio que este script.")
    except Exception as e:
        print(f"❌ ERROR inesperado: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    modelos = main()
