# Proyecto: Procesamiento y Modelado de Datos de Café

Pipeline completo para:

1. **Procesar datos crudos** (control de calidad y tostión) desde varios *sheets* de Excel.
2. **Generar un dataset limpio y unificado**.
3. **Entrenar modelos de ML** (Random Forest, Lasso, Ridge) para predecir `PUNTAJE`.


## 1. Instalar dependencias:
pip install -r requirements.txt

## 2. Ejecutar el procesamiento:
  python scripts/procesar_cafe.py

## 3. Entrenar un Modelo: 
python scripts/modelado_cafe.py --data data/processed/dataset_final_procesado.csv --modelo rf

Cambiar --modelo a lasso o ridge según necesidad.

## 4. Comparar Modelos:
ejecuta los tres:
python scripts/modelado_cafe.py --data data/processed/dataset_final_procesado.csv --modelo rf
python scripts/modelado_cafe.py --data data/processed/dataset_final_procesado.csv --modelo lasso
python scripts/modelado_cafe.py --data data/processed/dataset_final_procesado.csv --modelo ridge




