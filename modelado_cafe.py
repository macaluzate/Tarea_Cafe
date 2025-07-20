 
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta al CSV procesado")
    parser.add_argument("--modelo", choices=["rf", "lasso", "ridge"], default="rf")
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.data)

    numeric_cols = ['%H', 'TIEMPO_TUESTE_MEDIAN', 'CANTIDAD']
    categorical_cols = ['ORIGEN', 'RESPONSABLE', 'VARIEDAD_ESTANDAR', 'PROCESO', 'BENEFICIO']
    target_col = 'PUNTAJE'

    X = df[numeric_cols + categorical_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])

    if args.modelo == "rf":
        model = RandomForestRegressor(random_state=42)
    elif args.modelo == "lasso":
        model = Lasso(alpha=0.1, random_state=42)
    else:  # ridge
        model = Ridge(alpha=0.1, random_state=42)

    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"Modelo: {args.modelo}")
    print(f"R2:  {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

if __name__ == "__main__":
    main()
