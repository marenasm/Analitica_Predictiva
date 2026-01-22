# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
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
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

#
#import os
#import gzip
#import json
#import pickle
#
#import pandas as pd
#from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import (
#    balanced_accuracy_score,
#    precision_score,
#    recall_score,
#    f1_score,
#    confusion_matrix,
#)
#from sklearn.model_selection import GridSearchCV
#
## ...existing code...
#
#
#def pregunta01():
#    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
#    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
#
#    def cleanse(df):
#        df = df.copy()
#        df.rename(columns={"default payment next month": "default"}, inplace=True)
#        df.drop(columns=["ID"], inplace=True)
#        df.dropna(inplace=True)
#        # eliminar registros con valores indicativos de "no disponible"
#        df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
#        # agrupar EDUCATION > 4 en categoria "others" (4)
#        df["EDUCATION"] = df["EDUCATION"].apply(lambda value: 4 if value > 4 else value)
#        return df
#
#    train_data = cleanse(train_data)
#    test_data = cleanse(test_data)
#
#    x_train = train_data.drop(columns=["default"])
#    y_train = train_data["default"]
#    x_test = test_data.drop(columns=["default"])
#    y_test = test_data["default"]
#
#    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
#    numeric_features = [c for c in x_train.columns if c not in categorical_features]
#
#    def make_pipeline(categorical_cols, numeric_cols):
#        # One-hot para categóricas, MinMax scaler para numéricas
#        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#        numeric_transformer = MinMaxScaler()
#        preprocessor = ColumnTransformer(
#            transformers=[
#                ("cat", categorical_transformer, categorical_cols),
#                ("num", numeric_transformer, numeric_cols),
#            ],
#            remainder="drop",
#        )
#        pipeline = Pipeline(
#            steps=[
#                ("preprocessor", preprocessor),
#                ("select", SelectKBest(score_func=f_classif)),
#                ("classifier", LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")),
#                # usar 'saga' para permitir l1/l2 y mayor max_iter para convergencia
#                ("classifier", LogisticRegression(max_iter=5000, random_state=42, solver="saga")),
#            ]
#        )
#        return pipeline
#
#    pipeline = make_pipeline(categorical_features, numeric_features)
#
#    def optimize(pipeline, x_train, y_train):
#        param_grid = {
#            # Selección de K mejores características
#            "select__k": [10, 20, 30, "all"],
#            # Hiperparámetros de regresión logística
#            "classifier__C": [0.01, 0.1, 1, 10],
#            "classifier__penalty": ["l2"],
#        }
#        param_grid = {
#            "select__k": [10, 20, 30, "all"],
#            "classifier__C": [0.01, 0.1, 1, 10, 100],
#            "classifier__penalty": ["l1", "l2"],
#            "classifier__class_weight": [None, "balanced"],
#        }
#        search = GridSearchCV(
#             estimator=pipeline,
#             param_grid=param_grid,
#             cv=10,
#             scoring="balanced_accuracy",
#             n_jobs=-1,
#             refit=True,
#         )
#        return search
#
#    grid_search = optimize(pipeline, x_train, y_train)
#    grid_search.fit(x_train, y_train)
#
#    model_dir = "files/models"
#    os.makedirs(model_dir, exist_ok=True)
#
#    def save_model(path, estimator):
#        # guardar el estimador final (best_estimator_) comprimido
#        to_save = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
#        with gzip.open(path, "wb") as handle:
#            pickle.dump(to_save, handle)
#        # guardar el estimador recibido (GridSearchCV) comprimido
#        with gzip.open(path, "wb") as handle:
#            pickle.dump(estimator, handle)
#        # Guardar el objeto recibido (GridSearchCV) comprimido, las pruebas esperan GridSearchCV
#        with gzip.open(path, "wb") as handle:
#            pickle.dump(estimator, handle)
#
#    save_model(os.path.join(model_dir, "model.pkl.gz"), grid_search)
#
#    def metrics_calc(y_true, y_pred, dataset):
#        return {
#            "type": "metrics",
#            "dataset": dataset,
#            "precision": precision_score(y_true, y_pred, zero_division=0),
#            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
#            "recall": recall_score(y_true, y_pred, zero_division=0),
#            "f1_score": f1_score(y_true, y_pred, zero_division=0),
#        }
#
#    def matrix_calc(y_true, y_pred, dataset):
#        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#        return {
#            "type": "cm_matrix",
#            "dataset": dataset,
#            "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
#            "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
#        }
#
#    pred_train = grid_search.predict(x_train)
#    pred_test = grid_search.predict(x_test)
#
#    metrics = [
#        metrics_calc(y_train, pred_train, "train"),
#        metrics_calc(y_test, pred_test, "test"),
#        matrix_calc(y_train, pred_train, "train"),
#        matrix_calc(y_test, pred_test, "test"),
#    ]
#
#    output_dir = "files/output"
#    os.makedirs(output_dir, exist_ok=True)
#    metrics_path = os.path.join(output_dir, "metrics.json")
#    with open(metrics_path, "w", encoding="utf-8") as handle:
#        for record in metrics:
#            handle.write(json.dumps(record) + "\n")
#
#
#if __name__ == "__main__":
#    pregunta01()
#
import gzip
import json
import os
import pickle
from typing import Dict, Any, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets de entrenamiento y prueba."""
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return train, test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Renombrar columna objetivo
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    # Eliminar ID
    df.drop(columns=["ID"], inplace=True)
    # Filtrar EDUCATION y MARRIAGE no disponibles
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    # EDUCATION > 4 -> 4 (otros)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    # Eliminar filas con NaN
    df = df.dropna()
    return df


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X e y."""
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


def build_pipeline() -> Pipeline:
    """Construye el pipeline de preprocesamiento + selección + clasificación."""
    cat = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), cat),
        ],
        remainder=MinMaxScaler(),
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    return pipe


def train_model(
    pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Ajusta el modelo usando GridSearchCV y retorna el objeto ya entrenado."""
    param_grid = {
        "feature_selection__k": range(1, x_train.shape[1] + 1),
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        refit=True,
        n_jobs=-1,
    )

    search.fit(x_train, y_train)
    return search


def save_model(estimator: Any, path: str) -> None:
    """Guarda el modelo comprimido con gzip."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def compute_metrics(name: str, y_true, y_pred) -> Dict[str, Any]:
    """Calcula métricas de clasificación."""
    return {
        "type": "metrics",
        "dataset": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion(name: str, y_true, y_pred) -> Dict[str, Any]:
    """Calcula la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


def save_metrics(results: List[Dict[str, Any]], path: str) -> None:
    """Guarda métricas y matrices de confusión"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    # 1. Cargar datos
    train_df, test_df = load_data()

    # 2. Limpiar datos
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # 3. Dividir en X e y
    x_train, y_train = split(train_df)
    x_test, y_test = split(test_df)

    # 4. Pipeline + entrenamiento
    pipeline = build_pipeline()
    model = train_model(pipeline, x_train, y_train)

    # 5. Guardar modelo
    save_model(model, "files/models/model.pkl.gz")

    # 6. Predicciones
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # 7. Métricas y matrices de confusión
    metrics = [
        compute_metrics("train", y_train, y_pred_train),
        compute_metrics("test", y_test, y_pred_test),
        compute_confusion("train", y_train, y_pred_train),
        compute_confusion("test", y_test, y_pred_test),
    ]

    # 8. Guardar métricas
    save_metrics(metrics, "files/output/metrics.json")


if __name__ == "__main__":
    main()