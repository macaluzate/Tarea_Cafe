# Café tostado

Este proyecto contiene scripts de Python para la limpieza y preprocesamiento de datasets relacionados con control de calidad y tostión de café.



## 🚀 Requisitos Previos

Asegúrate de tener instalado Python 3.7 o superior. Instala las dependencias usando:

```bash
pip install -r requirements.txt
```

O instala manualmente las librerías principales:

```bash
pip install pandas openpyxl scikit-learn numpy matplotlib
```

## 📊 Ejecución de los Scripts

### 1. Limpieza y Unificación de Datos (`cleanning_coffe.py`)

Este script procesa y unifica los datasets de control de calidad del café desde archivos Excel.

```bash
python cleanning_coffe.py
```

**Funcionalidades:**
- Carga archivos Excel de múltiples hojas (CC FT 17 y CC FT 18)
- Limpia nombres de columnas y elimina filas vacías
- Unifica datasets con mapeo de variedades de café
- Calcula tiempos de tueste y correlaciones
- Genera el archivo `dataset_unificado.csv`

### 2. Preprocesamiento y Modelado ML (`preprocess.py`)

Este script realiza machine learning sobre el dataset unificado para predecir puntajes de calidad.

```bash
python preprocess.py
```

**Funcionalidades:**
- Carga y limpia el dataset unificado
- Preprocesamiento con imputación y escalado
- Entrena 3 modelos: Random Forest, Lasso y Ridge
- Evaluación con métricas R² y MAE
- Visualización de predicciones vs valores reales

## ⚙️ Configuración de Rutas

### Cambiar la ruta de los datos de entrada

Si tus archivos de datos están en una ubicación diferente, puedes modificar las rutas en los scripts:

#### En `cleanning_coffe.py`:
```python
# Busca estas líneas en la función cargar_datasets() y modifica las rutas:
file_17 = '/content/CC FT 17   Formato de Control de Calidad Café de Trillado (1).xlsx'
file_18 = '/content/CC FT 18  Formato de  Tostión (1).xlsx'

# Cambia por tus rutas locales, por ejemplo:
file_17 = 'sample_data/CC FT 17 Formato de Control de Calidad.xlsx'
file_18 = 'sample_data/CC FT 18 Formato de Tostión.xlsx'
```

#### En `preprocess.py`:
```python
# Modifica la ruta del dataset unificado:
dataset_path = "dataset_unificado.csv"  # Cambia esta ruta si es necesario
```

### Configuración alternativa con variables de entorno

También puedes usar variables de entorno para definir las rutas:

```bash
export DATA_PATH="/ruta/a/tus/datos"
python cleaning_coffe.py
```

## 📈 Resultados Esperados

Después de ejecutar los scripts obtendrás:

### Desde `cleanning_coffe.py`:
1. **Dataset unificado** (`dataset_unificado.csv`) con todas las muestras procesadas
2. **Información de carga** mostrando el número de filas y columnas procesadas
3. **Matriz de correlación** calculada entre variables numéricas

### Desde `preprocess.py`:
1. **Modelos entrenados**: Random Forest, Lasso y Ridge
2. **Métricas de evaluación**: R², MAE para train/test
3. **Gráficos de análisis**:
   - Predicciones vs valores reales
   - Distribución de residuos
4. **Cross-validation** para Random Forest


