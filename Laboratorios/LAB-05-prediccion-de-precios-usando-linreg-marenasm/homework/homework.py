#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error


# ============================================================
# 1. Cargar data
# ============================================================
def load_data(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, compression="zip")


# ============================================================
# 2. Limpiar data
# ============================================================
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Crear Age
    df["Age"] = 2021 - df["Year"]

    # Eliminar columnas
    df = df.drop(columns=["Year", "Car_Name"])

    # Quitar NA
    df = df.dropna()

    return df


# ============================================================
# 3. Split
# ============================================================
def split_data(data_train, data_test):
    data_train = data_train.copy()
    data_test = data_test.copy()

    X_train = data_train.drop(columns="Present_Price")
    y_train = data_train["Present_Price"]

    X_test = data_test.drop(columns="Present_Price")
    y_test = data_test["Present_Price"]

    return X_train, y_train, X_test, y_test


# ============================================================
# 4. Pipeline (sin remainder="passthrough")
# ============================================================
def modelo() -> Pipeline:
    cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    num_cols = ["Selling_Price", "Driven_kms", "Age", "Owner",]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ]
    )

    pipe = Pipeline([
        ("preprocesador", preprocesador),
        ("select", SelectKBest(score_func=f_regression)),
        ("modelo", LinearRegression())
    ])

    return pipe


# ============================================================
# 5. GridSearchCV
# ============================================================
def hiperparametros(modelo, x_train, y_train):
    grid = GridSearchCV(
        estimator=modelo,
        param_grid={"select__k": range(1, 15)},
        cv=10,
        scoring="neg_mean_absolute_error",
        refit=True,
    )

    grid.fit(x_train, y_train)
    return grid


# ============================================================
# 6. Métricas
# ============================================================
def metricas(modelo, X_train, y_train, X_test, y_test):
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    met_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, y_pred_train),
        "mse": mean_squared_error(y_train, y_pred_train),
        "mad": median_absolute_error(y_train, y_pred_train),
    }

    met_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, y_pred_test),
        "mse": mean_squared_error(y_test, y_pred_test),
        "mad": median_absolute_error(y_test, y_pred_test),
    }

    return met_train, met_test


# ============================================================
# 7. Guardar modelo
# ============================================================
def guardar_modelo(modelo, path="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(modelo, f)


# ============================================================
# 8. Guardar métricas
# ============================================================
def guardar_metricas(metricas_list, path="files/output/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for m in metricas_list:
            f.write(json.dumps(m) + "\n")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    train = load_data("files/input/train_data.csv.zip")
    test = load_data("files/input/test_data.csv.zip")

    train = limpiar_datos(train)
    test = limpiar_datos(test)

    X_train, y_train, X_test, y_test = split_data(train, test)

    pipe = modelo()

    grid = hiperparametros(pipe, X_train, y_train)

    guardar_modelo(grid, "files/models/model.pkl.gz")

    met_train, met_test = metricas(grid, X_train, y_train, X_test, y_test)

    guardar_metricas([met_train, met_test], "files/output/metrics.json")


if __name__ == "__main__":
    main()
